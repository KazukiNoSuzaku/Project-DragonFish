"""Starting and stopping command trees, per operating system.

A command often starts children (a shell runs `pytest`, which runs Python). Stopping it on a
timeout or Ctrl+C must stop the whole tree, and background commands must outlive DragonFish.

- POSIX: each command gets its own session (process group); one signal reaches the group.
- Windows: each foreground command is put in a Job Object. Every process it starts joins the
  job too, so ending the job ends the whole tree. (taskkill /T alone is not enough: Git Bash's
  fork emulation breaks the parent links it follows.)

(`sys.platform` checks are used here rather than `detect.current()` so type checkers see
which OS-only functions each branch may call.)
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from typing import Any

_CREATE_SUSPENDED = 0x00000004  # Win32 process creation flag (not exported by subprocess)


def spawn(argv: list[str], *, background: bool, **popen_kwargs: Any) -> subprocess.Popen[bytes]:
    """`subprocess.Popen` with the command in its own process tree.

    Foreground commands can be stopped as a tree with `kill_tree`; call `release` once they
    have finished. Background commands keep running after DragonFish exits.
    """
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        argv, **popen_kwargs, **_spawn_kwargs(background=background)
    )
    if sys.platform == "win32" and not background:
        # Started suspended, so it cannot start children before it is in the job.
        _jobs.adopt(proc.pid)
        if not _jobs.resume(proc.pid):
            proc.kill()
            raise OSError("could not start the command (resuming the process failed)")
    return proc


def _spawn_kwargs(*, background: bool) -> dict[str, Any]:
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        if not background:
            flags |= _CREATE_SUSPENDED  # resumed by `spawn` once it is in its job
        if background:
            # Its own hidden console: keeps running after DragonFish (and its terminal) exit.
            flags |= subprocess.CREATE_NO_WINDOW
        return {"creationflags": flags}
    # A new session is a new process group: one signal reaches every child, and the group
    # is not tied to our terminal, so background jobs survive DragonFish exiting.
    return {"start_new_session": True}


def kill_tree(proc: subprocess.Popen[Any]) -> None:
    """Stop a process started with `spawn` together with all of its children."""
    if sys.platform == "win32":
        if not _jobs.terminate(proc.pid):
            _taskkill(proc.pid)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2)
        _jobs.release(proc.pid)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=2)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)


def release(proc: subprocess.Popen[Any]) -> None:
    """Forget a finished process (frees its Windows job; leftover children keep running)."""
    if sys.platform == "win32":
        _jobs.release(proc.pid)


def kill_pid_tree(pid: int) -> None:
    """Force-stop a process tree known only by its id (e.g. a background job)."""
    if sys.platform == "win32":
        _taskkill(pid)
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


def _taskkill(pid: int) -> None:
    # /T follows parent links; /F forces the stop.
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


class _WindowsJobs:
    """Job Object handles for running foreground commands, by process id."""

    def __init__(self) -> None:
        self._handles: dict[int, int] = {}
        self._k32: Any = None

    def _kernel32(self) -> Any:
        if self._k32 is None:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k32.AssignProcessToJobObject.restype = wintypes.BOOL
            k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k32.TerminateJobObject.restype = wintypes.BOOL
            k32.QueryInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.c_void_p,
            ]
            k32.QueryInformationJobObject.restype = wintypes.BOOL
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            k32.CloseHandle.restype = wintypes.BOOL
            self._k32 = k32
        return self._k32

    def adopt(self, pid: int) -> None:
        """Put a just-started process in a new job (best effort: falls back to taskkill)."""
        k32 = self._kernel32()
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return
        process_set_quota, process_terminate = 0x0100, 0x0001
        handle = k32.OpenProcess(process_set_quota | process_terminate, False, pid)
        if handle and k32.AssignProcessToJobObject(job, handle):
            self._handles[pid] = job
        else:
            k32.CloseHandle(job)
        if handle:
            k32.CloseHandle(handle)

    def resume(self, pid: int) -> bool:
        """Resume a process created suspended (Popen does not keep its thread handle)."""
        import ctypes
        from ctypes import wintypes

        k32 = self._kernel32()
        ntdll = ctypes.WinDLL("ntdll")  # type: ignore[attr-defined]
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        process_suspend_resume = 0x0800
        handle = k32.OpenProcess(process_suspend_resume, False, pid)
        if not handle:
            return False
        try:
            return bool(ntdll.NtResumeProcess(handle) == 0)  # STATUS_SUCCESS
        finally:
            k32.CloseHandle(handle)

    def terminate(self, pid: int, timeout: float = 3.0) -> bool:
        """End every process in the job, and wait until they are gone.

        Termination is asynchronous, and closing the job handle too early can leave some
        processes (observed with Git Bash) running, so wait for the job to empty first.
        """
        import time

        job = self._handles.get(pid)
        if not job or not self._kernel32().TerminateJobObject(job, 1):
            return False
        deadline = time.monotonic() + timeout
        while self._active_processes(job) and time.monotonic() < deadline:
            time.sleep(0.02)
        return True

    def _active_processes(self, job: int) -> int:
        import ctypes
        from ctypes import wintypes

        class Accounting(ctypes.Structure):  # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
            _fields_ = [
                ("times", ctypes.c_int64 * 4),
                ("total_page_faults", wintypes.DWORD),
                ("total_processes", wintypes.DWORD),
                ("active_processes", wintypes.DWORD),
                ("total_terminated", wintypes.DWORD),
            ]

        info = Accounting()
        k32 = self._kernel32()
        ok = k32.QueryInformationJobObject(
            job, 1, ctypes.byref(info), ctypes.sizeof(info), None
        )
        return int(info.active_processes) if ok else 0

    def release(self, pid: int) -> None:
        job = self._handles.pop(pid, None)
        if job:
            self._kernel32().CloseHandle(job)


_jobs = _WindowsJobs()
