"""Fixed4MiB binary parts of one already-verified standard xz stream."""
from pathlib import Path
import hashlib,json
O=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
def main():
 meta=json.loads((O/'VECTORS_COMPRESSED_INDEX.json').read_text(encoding='utf8'));src=O/meta['archive'];assert H(src.read_bytes())==meta['archive_sha256'];parts=[];h=hashlib.sha256()
 with src.open('rb') as f:
  i=0
  while True:
   b=f.read(4*1024**2)
   if not b:break
   name=src.name+'.part'+str(i).zfill(3);p=O/name
   with p.open('xb') as o:o.write(b)
   parts.append(dict(path=name,bytes=len(b),sha256=H(b)));h.update(b);i+=1
 assert h.hexdigest()==meta['archive_sha256'];index=dict(status='ordered_fixed_binary_parts_no_data_transform',part_bytes=4*1024**2,parts=parts,whole_name=src.name,whole_bytes=meta['archive_bytes'],whole_sha256=meta['archive_sha256'],restored_name=meta['restored_name'],restored_bytes=meta['restored_bytes'],restored_sha256=meta['restored_sha256'],recovery='Concatenate parts in listed order, verify whole SHA, standard xz-decompress and verify original NPZ SHA. restore_delivery.py implements safe streamed verification/restoration.')
 (O/'VECTORS_PARTS_INDEX.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(parts=len(parts),whole_bytes=meta['archive_bytes'],whole_sha256=meta['archive_sha256'])))
if __name__=='__main__':main()
