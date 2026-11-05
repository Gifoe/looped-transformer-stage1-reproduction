"""Task transport to the user-designated replacement host; password from stdin only."""
import getpass
import os
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import paramiko

ROOT = Path(__file__).resolve().parents[1]
known = ROOT / 'transport/server_known_hosts'
if not known.exists():
    sock = socket.create_connection((os.environ['EXPERIMENT_SSH_HOST'], 22), timeout=15)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=15)
    key = transport.get_remote_server_key()
    print(json.dumps({'first_connection_host_key_sha256': hashlib.sha256(key.asbytes()).hexdigest()}), flush=True)
    keys = paramiko.HostKeys()
    keys.add(os.environ['EXPERIMENT_SSH_HOST'], key.get_name(), key)
    keys.save(str(known))
    transport.close()
client = paramiko.SSHClient()
client.load_host_keys(str(known))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
print('PASSWORD_INPUT', flush=True)
secret = getpass.getpass('SSH password: ') if sys.stdin.isatty() else sys.stdin.readline().rstrip('\r\n')
client.connect(os.environ['EXPERIMENT_SSH_HOST'], username=os.environ['EXPERIMENT_SSH_USER'], port=22, password=secret,
    look_for_keys=False, allow_agent=False, timeout=20, banner_timeout=30, auth_timeout=30)
del secret
client.get_transport().set_keepalive(15)
print('CONNECTED', flush=True)
for line in sys.stdin:
    path = line.strip()
    if path == 'QUIT':
        break
    try:
        request = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(120)
            for local, remote in request.get('upload', []):
                sftp.put(local, remote)
            for remote, local in request.get('download', []):
                p = Path(local); p.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(p), prefetch=False)
        result = {'request': path}
        if request.get('command'):
            begin = time.monotonic()
            _, out, err = client.exec_command(request['command'], timeout=request.get('timeout', 120))
            result.update(stdout=out.read().decode('utf8', errors='replace'),
                stderr=err.read().decode('utf8', errors='replace'),
                exit_code=out.channel.recv_exit_status(), elapsed_seconds=time.monotonic()-begin)
        if request.get('save_output'):
            Path(request['save_output']).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
        print(json.dumps(result, ensure_ascii=True), flush=True)
    except Exception as exc:
        print(json.dumps({'request': path, 'error': repr(exc)}), flush=True)
client.close()

