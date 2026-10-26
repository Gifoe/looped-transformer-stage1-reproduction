"""Task-only SSH transport. Reads password once from stdin; never persists it.

Subsequent stdin lines are paths to JSON request files. Supports PowerShell
commands, SFTP upload, and SFTP download. Host key must already be trusted.
"""
import base64
import getpass
import gzip
import json
from pathlib import Path
import sys
import time

import paramiko

client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.RejectPolicy())
print("PASSWORD_INPUT", flush=True)
secret = getpass.getpass("SSH password: ") if sys.stdin.isatty() else sys.stdin.readline().rstrip("\r\n")
client.connect("100.89.214.121", port=22, username="fyl412", password=secret,
               look_for_keys=False, allow_agent=False, timeout=20)
del secret
print("CONNECTED", flush=True)
for request_path in sys.stdin:
    request_path = request_path.strip()
    if request_path == "QUIT":
        break
    try:
        request = json.loads(Path(request_path).read_text(encoding="utf-8-sig"))
        if request.get("upload"):
            with client.open_sftp() as sftp:
                sftp.get_channel().settimeout(90)
                for local, remote in request.get("upload", []):
                    sftp.put(local, remote)
        for remote, local in request.get("download", []):
            # Windows OpenSSH SFTP stalled on larger artifacts even without
            # prefetch. A bounded compressed stdout transfer avoids that path.
            quoted = remote.replace("'", "''")
            code="(Get-Item -LiteralPath '"+quoted+"').Length"
            encoded=base64.b64encode(code.encode('utf-16-le')).decode()
            _,out,err=client.exec_command('powershell.exe -NoProfile -NonInteractive -EncodedCommand '+encoded,timeout=60)
            size=int(out.read().strip())
            if out.channel.recv_exit_status():raise RuntimeError(err.read().decode(errors='replace'))
            destination=Path(local);destination.parent.mkdir(parents=True,exist_ok=True)
            partial=destination.with_suffix(destination.suffix+'.download')
            # Bound every stdout payload below the Windows SSH channel window.
            with partial.open('wb') as target:
                for offset in range(0,size,256*1024):
                    count=min(256*1024,size-offset)
                    code="$f=[IO.File]::OpenRead('"+quoted+"');[void]$f.Seek("+str(offset)+",[IO.SeekOrigin]::Begin);$b=New-Object byte[] "+str(count)+";$n=$f.Read($b,0,$b.Length);$f.Close();$m=New-Object IO.MemoryStream;$g=New-Object IO.Compression.GZipStream($m,[IO.Compression.CompressionMode]::Compress);$g.Write($b,0,$n);$g.Close();[Convert]::ToBase64String($m.ToArray())"
                    encoded=base64.b64encode(code.encode('utf-16-le')).decode()
                    _,out,err=client.exec_command('powershell.exe -NoProfile -NonInteractive -EncodedCommand '+encoded,timeout=60)
                    payload=out.read().strip()
                    if out.channel.recv_exit_status():raise RuntimeError(err.read().decode(errors='replace'))
                    decoded=gzip.decompress(base64.b64decode(payload));assert len(decoded)==count
                    target.write(decoded);target.flush()
                    print(json.dumps(dict(download=remote,bytes_completed=offset+count,total_bytes=size)),flush=True)
            partial.replace(destination)
        if "powershell" in request:
            encoded = base64.b64encode(request["powershell"].encode("utf-16-le")).decode()
            start = time.monotonic()
            _, stdout, stderr = client.exec_command(
                "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand " + encoded,
                timeout=request.get("timeout", 120))
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            result = dict(request=request_path, exit_code=stdout.channel.recv_exit_status(),
                          elapsed_s=time.monotonic()-start, stdout=out, stderr=err)
            if request.get("save_output"):
                Path(request["save_output"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=True), flush=True)
        else:
            print(json.dumps(dict(request=request_path, transferred=True)), flush=True)
    except Exception as exc:
        print(json.dumps(dict(request=request_path, error=repr(exc))), flush=True)
client.close()
