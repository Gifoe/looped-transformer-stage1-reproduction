"""Grade generated code with EvalPlus in a restricted disposable child process."""
import argparse
import builtins
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import sysconfig
import tempfile

from evalplus import eval as checker
from evalplus import evaluate
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
from evalplus.sanitize import sanitize


LIB = ctypes.CDLL("libseccomp.so.2")
PRCTL = ctypes.CDLL(None, use_errno=True).prctl
LIB.seccomp_init.argtypes = [ctypes.c_uint32]
LIB.seccomp_init.restype = ctypes.c_void_p
LIB.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
LIB.seccomp_syscall_resolve_name.restype = ctypes.c_int
LIB.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
LIB.seccomp_load.argtypes = [ctypes.c_void_p]
LIB.seccomp_release.argtypes = [ctypes.c_void_p]
ALLOW = """
read write close fstat newfstatat stat lstat statx openat lseek pread64 readv writev fcntl
getdents64 access faccessat faccessat2 readlink readlinkat close_range
mmap mprotect munmap brk mremap madvise futex clock_gettime clock_getres nanosleep clock_nanosleep
getrandom getuid geteuid getgid getegid getpid gettid getpgrp getppid statfs getcwd chdir umask
mkdir mkdirat rmdir unlink unlinkat rename renameat renameat2 chmod fchmod fchmodat
getrlimit setrlimit prlimit64 rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack getitimer
setitimer gettimeofday times uname sched_yield sched_getaffinity restart_syscall exit exit_group
""".split()
EXECUTE = checker.unsafe_execute


def preload():
    # These common extensions may link against the parent Python distribution's lib directory.
    for name in ("array", "_bisect", "_collections", "_csv", "_decimal", "_hashlib", "_heapq",
                 "_json", "_opcode", "_pickle", "_random", "_struct", "binascii", "cmath",
                 "math", "select", "unicodedata", "zlib", "fractions", "operator", "statistics",
                 "decimal", "itertools", "queue", "typing", "socket", "ssl", "re"):
        try:
            builtins.__import__(name)
        except ImportError:
            continue


def jail_path(root):
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    destination = root / stdlib.relative_to("/")
    if not destination.is_dir():
        shutil.copytree(stdlib, destination, ignore=shutil.ignore_patterns(
            "site-packages", "__pycache__", "test", "tests", "ensurepip", "tkinter", "idlelib", "turtledemo"
        ))
    scratch = root / "tmp"
    scratch.mkdir(exist_ok=True)
    scratch.chmod(0o1777)


def restrict(jail):
    pid = os.getpid()
    fd = os.open("/dev/null", os.O_RDWR)
    for target in (0, 1, 2):
        os.dup2(fd, target)
    for name in os.listdir("/proc/self/fd"):
        number = int(name)
        if number >= 3:
            try:
                os.close(number)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
    os.umask(0o077)
    os.environ.clear()
    os.environ.update(TMPDIR="/tmp", EVALPLUS_MAX_MEMORY_BYTES=str(1536 << 20))
    tempfile.tempdir = "/tmp"
    os.chroot(jail)
    os.chdir("/")
    uid = 200000 + pid
    os.setgroups([])
    os.setresgid(uid, uid, uid)
    os.setresuid(uid, uid, uid)
    resource.setrlimit(resource.RLIMIT_CPU, (9, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 << 20, 4 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if PRCTL(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
    ctx = LIB.seccomp_init(0x00050000 | errno.EPERM)
    if not ctx:
        raise RuntimeError("seccomp_init failed")
    for item in ALLOW:
        call = LIB.seccomp_syscall_resolve_name(item.encode())
        if call >= 0 and LIB.seccomp_rule_add(ctx, 0x7FFF0000, call, 0) != 0:
            raise RuntimeError(f"Could not register {item}")
    if LIB.seccomp_load(ctx) != 0:
        raise RuntimeError("seccomp_load failed")
    LIB.seccomp_release(ctx)


def patched_runner(jail):
    def guarded(*args):
        restrict(jail)
        # EvalPlus normally reads host /proc/meminfo while setting this limit.
        checker.query_maximum_memory_bytes = lambda: 1536 << 20
        return EXECUTE(*args)

    return guarded


def smoke():
    code = "def f(x):\n    import math\n    return math.floor(x) + 1\n"
    status, _ = checker.untrusted_check("humaneval", code, [[2.5]], "f", [3], 0., [0.001], fast_check=True)
    if status != "pass":
        raise RuntimeError(f"Restricted worker smoke test failed: {status}")
    code = (
        "def f(x):\n"
        "    import socket\n"
        "    try:\n"
        "        socket.socket()\n"
        "    except PermissionError:\n"
        "        return 3\n"
        "    return 4\n"
    )
    status, _ = checker.untrusted_check("humaneval", code, [[2.5]], "f", [3], 0., [0.001], fast_check=True)
    if status != "pass":
        raise RuntimeError(f"Network isolation smoke test failed: {status}")
    print("SANDBOX_SMOKE PASS", flush=True)


def grade(dataset, run, args):
    if dataset == "humaneval":
        problems, checksum = get_human_eval_plus(), get_human_eval_plus_hash()
        special = ()
    else:
        problems, checksum = get_mbpp_plus(), get_mbpp_plus_hash()
        special = checker.MBPP_OUTPUT_NOT_NONE_TASKS
    oracle = evaluate.get_groundtruth(problems, checksum, special)
    (run / "EVALPLUS_DATASET.json").write_text(json.dumps(dict(dataset=dataset, official_evalplus_hash=checksum,
           questions=len(problems), version="0.3.1",
           source_id_to_official="Prefix numeric ID with Mbpp/" if dataset == "mbpp" else "Identity"),
           indent=2) + "\n", encoding="utf-8")
    if args.prepare:
        return
    for arm in ("native", "fixed", "adaptive", "feedback"):
        source, output = run / f"{arm}.jsonl", run / f"{arm}.evalplus.jsonl"
        if output.exists():
            continue
        rows = [json.loads(s) for s in source.read_text(encoding="utf-8").splitlines()]
        def official_id(key):
            return f"Mbpp/{key}" if dataset == "mbpp" else key

        if {official_id(p["id"]) for p in rows} != problems.keys() or len(rows) != len(problems):
            raise ValueError("The full, official EvalPlus task IDs must match")
        tmp = output.with_suffix(".partial")
        with tmp.open("w", encoding="utf-8") as stream:
            for idx, row in enumerate(rows):
                key = row["id"]
                official = official_id(key)
                p, g = problems[official], oracle[official]
                response = row["generated"]
                if dataset == "humaneval":
                    response = p["prompt"] + response
                solution = sanitize(response, entrypoint=p["entry_point"])
                metrics = {}
                for part in ("base", "plus"):
                    if part == "plus" and metrics["base"] != "pass":
                        metrics[part] = "skipped_after_base_failure"
                        continue
                    status, _ = checker.untrusted_check(dataset, solution, p[f"{part}_input"], p["entry_point"],
                                                        g[part], p["atol"], g[f"{part}_time"], fast_check=True)
                    metrics[part] = status
                stream.write(json.dumps(dict(id=key, evalplus_id=official, sanitized=solution, base=metrics["base"],
                    plus=metrics["plus"], source_sha256=hashlib.sha256(row["generated"].encode()).hexdigest()),
                    ensure_ascii=False, separators=(",", ":")) + "\n")
                if idx % 25 == 0:
                    stream.flush()
                    print("GRADE", dataset, arm, idx + 1, flush=True)
        tmp.replace(output)
        print("GRADE_COMPLETE", dataset, arm, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jail", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--task", choices=("humaneval", "mbpp"))
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    jail_path(a.jail)
    preload()
    if not a.prepare:
        checker.unsafe_execute = patched_runner(a.jail)
        smoke()
    if a.smoke_only:
        sys.exit(0)
    run = a.results / a.task
    run.mkdir(parents=True, exist_ok=True)
    grade(a.task, run, a)
