"""Standard lossless CPU-only vector container compression and streamed SHA replay."""
from pathlib import Path
import lzma,hashlib,json,shutil,time
O=Path(__file__).resolve().parent

def main():
 start=time.monotonic();src=O/'VECTORS.npz';dst=O/'VECTORS.npz.xz';assert not dst.exists();original=hashlib.sha256(src.read_bytes()).hexdigest();assert original==json.loads((O/'COMPLETE.json').read_text())['VECTORS_sha256']
 with src.open('rb') as i,lzma.open(dst,'xb',preset=9) as o:shutil.copyfileobj(i,o,1024*1024)
 h=hashlib.sha256();n=0
 with lzma.open(dst,'rb') as f:
  while True:
   b=f.read(1024*1024)
   if not b:break
   h.update(b);n+=len(b)
 assert h.hexdigest()==original and n==src.stat().st_size
 index=dict(status='standard_xz_lossless_CPU_streamed_roundtrip_exact',archive=dst.name,archive_bytes=dst.stat().st_size,archive_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),restored_name=src.name,restored_bytes=n,restored_sha256=original,preset=9,CPU_seconds=time.monotonic()-start,GPU_seconds=0,recovery='Verify compressed SHA; standard xz-decompress to VECTORS.npz; verify raw SHA/bytes. No tensor transforms or quantization.')
 (O/'VECTORS_COMPRESSED_INDEX.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf8');print(json.dumps(index))
if __name__=='__main__':main()
