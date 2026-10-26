"""Compare task-owned cached bytecode with fresh compilation without execution."""
import hashlib
import importlib.util
import json
import marshal
from pathlib import Path
import shutil
import struct
import types

root = Path(__file__).resolve().parents[1]
base = root.parents[1]
out = root/'logs/native26_bytecode_audit'
assert not out.exists(), 'Preserve existing audit before a separate diagnostic'
out.mkdir()

def canonical(value):
    if isinstance(value, types.CodeType):
        fields = ['co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals',
                  'co_stacksize', 'co_flags', 'co_code', 'co_consts', 'co_names',
                  'co_varnames', 'co_freevars', 'co_cellvars', 'co_exceptiontable']
        return {name: canonical(getattr(value, name)) for name in fields}
    if isinstance(value, bytes):
        return {'bytes': value.hex()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, frozenset):
        return {'frozenset': sorted([canonical(v) for v in value], key=repr)}
    if value is Ellipsis:
        return {'ellipsis': True}
    if isinstance(value, complex):
        return {'complex': [value.real, value.imag]}
    return value

results = []
sources = [root/'scripts/common.py', root/'scripts/methods.py', root/'scripts/generate_math.py',
           base/'scripts/qwen_recipe.py',
           root/'cache/huggingface/modules/transformers_modules/Ouro_hyphen_2_dot_6B/modeling_ouro.py']
for index, source in enumerate(sources):
    print(json.dumps(dict(checking=str(source))), flush=True)
    if not source.exists():
        results.append(dict(source=str(source), source_missing=True))
        continue
    data = source.read_bytes()
    cache = Path(importlib.util.cache_from_source(str(source)))
    entry = dict(source=str(source), source_sha256=hashlib.sha256(data).hexdigest(),
                 cache=str(cache), cache_exists=cache.exists())
    if cache.exists():
        payload = cache.read_bytes()
        shutil.copyfile(cache, out/f'{index}_{cache.name}')
        entry['cache_sha256'] = hashlib.sha256(payload).hexdigest()
        entry['magic_matches'] = payload[:4] == importlib.util.MAGIC_NUMBER
        if entry['magic_matches']:
            flags = struct.unpack_from('<I', payload, 4)[0]
            entry['header_flags'] = flags
            entry['source_header_matches'] = (payload[8:16] == importlib.util.source_hash(data)
                if flags & 1 else struct.unpack_from('<II', payload, 8) == (int(source.stat().st_mtime) & 0xffffffff, len(data)))
            cached = canonical(marshal.loads(payload[16:]))
            fresh = canonical(compile(data, str(source), 'exec', dont_inherit=True, optimize=0))
            entry['semantic_code_matches'] = cached == fresh
            for name, value in [('cached', cached), ('fresh', fresh)]:
                entry[name+'_semantic_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    results.append(entry)
result = dict(results=results, semantic_mismatches=sum(r.get('semantic_code_matches') is False for r in results),
              scope='Read-only bytecode/source comparison and preserved cache bytes. No cached code executed, no cache deleted, no runtime stability claim.')
(out/'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result), flush=True)
