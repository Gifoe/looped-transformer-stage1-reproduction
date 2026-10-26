"""Task-process-only Windows affinity, using pointer-sized masks."""
import ctypes
from ctypes import wintypes
import os


def kernel():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.GetCurrentProcess.restype = wintypes.HANDLE
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    api.GetProcessAffinityMask.restype = wintypes.BOOL
    api.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    api.SetProcessAffinityMask.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    return api


def get_mask(pid=None):
    api = kernel()
    handle = api.GetCurrentProcess() if pid is None else api.OpenProcess(0x0400, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not api.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        return process.value, system.value
    finally:
        if pid is not None:
            api.CloseHandle(handle)


def set_self(mask):
    value = int(mask, 0) if isinstance(mask, str) else mask
    before, system = get_mask()
    assert value > 0 and value & system == value
    api = kernel()
    if not api.SetProcessAffinityMask(api.GetCurrentProcess(), value):
        raise ctypes.WinError(ctypes.get_last_error())
    assert get_mask()[0] == value
    return dict(pid=os.getpid(), before_mask=hex(before), mask=hex(value),
                scope='This process and subsequently inherited child processes only')
