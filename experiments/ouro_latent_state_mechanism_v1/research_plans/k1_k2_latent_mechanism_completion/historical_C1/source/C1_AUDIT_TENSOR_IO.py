"""Independent read-side verification of raw or losslessly gzipped tensor archives."""
import gzip
import hashlib
from pathlib import Path

def digest(data):return hashlib.sha256(data).hexdigest()
def archive_receipt(raw_path, metadata):
    raw_path=Path(raw_path)
    path=raw_path if raw_path.exists() else raw_path.with_name(raw_path.name+'.gz')
    assert path.exists(),path
    archived=path.read_bytes()
    compressed=path.suffix=='.gz'
    expected=metadata['tensor_archive_sha256'] if compressed else metadata['tensor_sha256']
    assert digest(archived)==expected,(path,'archive checksum')
    return {'file':str(path),'archive_sha256':digest(archived),
            'raw_tensor_sha256':metadata['tensor_sha256'],'gzip':compressed},archived

def load_checked(raw_path,metadata):
    from safetensors.torch import load
    receipt,archived=archive_receipt(raw_path,metadata)
    raw=gzip.decompress(archived) if receipt['gzip'] else archived
    assert digest(raw)==metadata['tensor_sha256'],(raw_path,'decompressed tensor checksum')
    return load(raw),receipt
