"""Actual training-free BF16 weight edits and exact reproducible replacement patches.

A: selected MLP down_proj input columns are attenuated by (1-alpha).
B: W' = BF16(FP32(W) - alpha U(U.T FP32(W))), optionally on selected columns.
Only actual physical output-projection parameters may be patched; every recurrent
invocation therefore uses the same edited tensor. No hooks/adapters/optimizers.
All persistence is restricted to this V2 experiment directory; V1 is read-only.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import torch

ROOT = Path(__file__).resolve().parents[1]
REVISION = '574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
ALLOWED = re.compile(r'^model\.layers\.(\d+)\.(mlp\.down_proj|self_attn\.o_proj)\.weight$')


def tensor_sha256(tensor):
    """SHA256 of contiguous CPU raw bytes; shape/dtype are recorded separately."""
    value = tensor.detach().to('cpu').contiguous()
    # Singleton trailing dimensions can be "contiguous" with a non-unit final
    # stride (e.g. QR's [d,1] output); dtype reinterpretation requires stride one.
    flat = torch.empty(value.numel(), dtype=value.dtype, device='cpu')
    flat.copy_(value.reshape(-1))
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _validate_name(name):
    match = ALLOWED.fullmatch(name)
    if match is None:
        raise ValueError('Only physical Ouro MLP down_proj or attention o_proj weights may change: '+name)
    return int(match[1]), match[2]


def _base(name, weight):
    _validate_name(name)
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError('Source must be the actual evaluated BF16 [output,input] matrix')
    value = weight.detach().to('cpu').contiguous().clone()
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite source weight')
    return value


def _support(columns, size):
    if columns is None:
        return None
    result = tuple(int(x) for x in columns)
    if not result or len(result) != len(set(result)) or min(result)<0 or max(result)>=size:
        raise ValueError('Columns must be nonempty, unique, in-range input-column indices')
    return result


def _strength(alpha):
    alpha = float(alpha)
    if not torch.isfinite(torch.tensor(alpha)) or alpha < 0:
        raise ValueError('Require finite nonnegative alpha')
    return alpha


@dataclass
class WeightEdit:
    name: str
    base: torch.Tensor                 # exact CPU BF16 original
    edited: torch.Tensor               # exact CPU BF16 materialized candidate
    specification: dict
    factors: dict = None               # optional small exact FP32 direction factors

    @property
    def stats(self):
        return edit_statistics(self)


@torch.no_grad()
def sparse_column_attenuation(name, weight, columns, alpha, provenance=None):
    """Family A. This scales selected MLP columns, NOT the whole output matrix."""
    _, branch = _validate_name(name)
    if branch != 'mlp.down_proj':
        raise ValueError('Family A is restricted to MLP down_proj columns')
    base = _base(name, weight)
    columns = _support(columns, base.shape[1])
    if columns is None:
        raise ValueError('Sparse editing needs explicit columns')
    alpha = _strength(alpha)
    if alpha > 1:
        raise ValueError('Column attenuation uses alpha in [0,1]; do not silently reverse signs')
    edited = base.clone()
    if alpha:
        edited[:, list(columns)] = (base[:, list(columns)].float() * (1-alpha)).to(torch.bfloat16)
    return WeightEdit(name, base, edited, dict(family='A_sparse_column_attenuation',
        formula='selected W[:,S] <- BF16((1-alpha)*FP32(W[:,S]))',
        alpha=alpha, columns=list(columns), column_count=len(columns),
        provenance=dict(provenance or {})))


@torch.no_grad()
def low_rank_projection(name, weight, U, alpha, columns=None, provenance=None):
    """Family B. U has one or four orthonormal output-space columns.

    Optional columns implements the MIT-like sparse directional variant. It must
    not be mislabeled as uniform selected-column attenuation (family A).
    """
    base = _base(name, weight)
    basis = U.detach().to(device='cpu', dtype=torch.float32).contiguous()
    if basis.ndim == 1:
        basis = basis[:, None]
    if basis.ndim != 2 or basis.shape[0] != base.shape[0] or basis.shape[1] not in (1,4):
        raise ValueError('Require U shape [output_dim,1] or [output_dim,4]')
    if not torch.isfinite(basis).all():
        raise ValueError('Nonfinite basis')
    gram = basis.T @ basis
    if not torch.allclose(gram, torch.eye(basis.shape[1]), atol=1e-4, rtol=1e-4):
        raise ValueError('U is not orthonormal; construct/record the basis explicitly')
    columns = _support(columns, base.shape[1]);alpha = _strength(alpha)
    original = base.float();target = original.clone()
    if alpha:
        if columns is None:
            target -= alpha * (basis @ (basis.T @ original))
        else:
            sub = original[:, list(columns)]
            target[:, list(columns)] = sub - alpha*(basis @ (basis.T @ sub))
    edited = target.to(torch.bfloat16)
    if not torch.isfinite(edited).all():
        raise ValueError('Candidate became nonfinite after BF16 materialization')
    return WeightEdit(name, base, edited, dict(family='B_low_rank_projection' if columns is None else 'A_sparse_directional_projection',
        formula='W <- BF16(FP32(W)-alpha*U*(U.T*FP32(W))*D_S)',
        alpha=alpha, intended_update_rank=basis.shape[1], columns=None if columns is None else list(columns),
        basis_sha256=tensor_sha256(basis), basis_dtype='torch.float32',
        provenance=dict(provenance or {})), factors={'U':basis.clone()})


@torch.no_grad()
def edit_statistics(edit, total_unique_parameters=None):
    base, final = edit.base, edit.edited
    if base.dtype != torch.bfloat16 or final.dtype != torch.bfloat16 or base.shape != final.shape:
        raise ValueError('Exact BF16 original/edited matrices required')
    changed = base.view(torch.int16) != final.view(torch.int16)
    delta = final.float()-base.float()
    columns = edit.specification.get('columns')
    support = base.numel() if columns is None else base.shape[0]*len(columns)
    changed_count = int(changed.sum())
    if columns is not None:
        mask = torch.ones(base.shape[1], dtype=torch.bool);mask[columns] = False
        if changed[:,mask].any():
            raise ValueError('Sparse edit changed a coordinate outside its declared support')
    base_norm = float(torch.linalg.vector_norm(base.float()))
    delta_norm = float(torch.linalg.vector_norm(delta))
    return dict(tensor_name=edit.name, shape=list(base.shape), evaluated_dtype='torch.bfloat16',
        base_sha256=tensor_sha256(base), edited_sha256=tensor_sha256(final), hash_semantics='contiguous_cpu_tensor_raw_bytes',
        intended_support_coordinates=support, actual_changed_coordinates=changed_count,
        numerically_changed_coordinates=int(torch.count_nonzero(delta)),
        unchanged_coordinates_in_support=support-changed_count,
        changed_fraction_of_tensor=changed_count/base.numel(), total_unique_parameters=total_unique_parameters,
        changed_fraction_of_model=None if total_unique_parameters is None else changed_count/total_unique_parameters,
        base_frobenius=base_norm, actual_delta_frobenius=delta_norm,
        relative_delta_frobenius=delta_norm/max(base_norm,1e-30), max_abs_delta=float(delta.abs().max()),
        note='Intended low rank may increase after BF16 rounding; changed coordinates count actual BF16 bit differences.')


def summarize_edits(edits, total_unique_parameters):
    edits = list(edits)
    if not edits or len({e.name for e in edits}) != len(edits):
        raise ValueError('Nonempty candidate with unique tensor names required')
    layers = {_validate_name(e.name)[0] for e in edits}
    if len(layers)>2:
        raise ValueError('Initial V2 candidate may modify at most two physical layers')
    if total_unique_parameters<=0:
        raise ValueError('Positive unique-parameter denominator required')
    details = [dict(**edit_statistics(e,total_unique_parameters), specification=e.specification) for e in edits]
    changed = sum(d['actual_changed_coordinates'] for d in details)
    if changed<=0:
        raise ValueError('Edited candidate has no actual BF16 changes; baseline is not final-selection eligible')
    return dict(model_revision=REVISION, total_unique_parameters=total_unique_parameters,
        actual_changed_parameters=changed, actual_changed_fraction_of_model=changed/total_unique_parameters,
        actual_delta_frobenius=sum(d['actual_delta_frobenius']**2 for d in details)**.5,
        edited_physical_layers=sorted(layers), edits=details, training=False,
        all_recurrent_invocations_share_edited_parameters=True,
        embeddings_tokenizer_head_and_norm_weights_unchanged=True)


@contextmanager
def applied_edits(model, edits, invalidate=None):
    """Copy materialized BF16 weights into REAL parameters; restore exact clones.

    Usage: with applied_edits(model, candidate, invalidate=scorer.invalidate): ...
    Every result/readout in the block comes from the edited model's own trajectory.
    Never use baseline cached states inside the block. Optional invalidate callback
    runs after application AND restoration; cached paths remain independently gated.
    Restoration runs on normal return or exception, with exact hash verification.
    """
    edits=list(edits)
    if any(m.training for m in model.modules()):
        raise ValueError('Require model.eval() before weight editing')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Require model.requires_grad_(False); no autograd graph is allowed')
    parameters=dict(model.named_parameters())
    total=sum(p.numel() for p in model.parameters())
    summary=summarize_edits(edits,total)
    selected=[]
    for edit in edits:
        _validate_name(edit.name)
        if edit.name not in parameters:
            raise KeyError('Physical parameter not found: '+edit.name)
        p=parameters[edit.name]
        if p.dtype!=torch.bfloat16 or p.shape!=edit.base.shape or tensor_sha256(p)!=tensor_sha256(edit.base):
            raise ValueError('Base mismatch; candidate must be applied to the preserved original checkpoint: '+edit.name)
        selected.append((p,edit))
    if len({id(p) for p,e in selected})!=len(selected):
        raise ValueError('Candidate contains aliased physical tensors')
    backups=[]
    with torch.no_grad():
        try:
            for p,edit in selected:
                backups.append((p,p.detach().clone(),tensor_sha256(p)))
                p.copy_(edit.edited.to(device=p.device))
                if tensor_sha256(p)!=tensor_sha256(edit.edited):
                    raise RuntimeError('Edited BF16 parameter hash mismatch after copy')
            if invalidate is not None:invalidate()
            yield summary
        finally:
            try:
                for p,original,expected in backups:
                    p.copy_(original)
                    if tensor_sha256(p)!=expected:
                        raise RuntimeError('Failed exact original-weight restoration')
            finally:
                if invalidate is not None:invalidate()


def _v2_destination(directory):
    destination=Path(directory).resolve()
    if not destination.is_relative_to(ROOT.resolve()):
        raise ValueError('Write destination must remain inside this V2 experiment')
    return destination


@torch.no_grad()
def save_edit_bundle(directory, edits, total_unique_parameters, metadata=None):
    """Save exact replacement patch, not a rounded recomputation of low-rank factors.

    Sparse payloads store changed indices + target BF16 values; dense payloads store
    the complete target BF16 matrix when smaller. Source hash prevents application
    to any other checkpoint. Both formats reconstruct evaluated weights bit-for-bit.
    """
    edits=list(edits);summary=summarize_edits(edits,total_unique_parameters)
    destination=_v2_destination(directory)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Preserve existing edit bundle: '+str(destination))
    destination.mkdir(parents=True,exist_ok=True)
    manifest=dict(format='ouro_v2_exact_bf16_replacement_patch_v1',**summary,metadata=dict(metadata or {}))
    for i,(edit,info) in enumerate(zip(edits,manifest['edits'])):
        target_path=destination/f'edited_tensor_{i:02d}.pt'
        torch.save(edit.edited,target_path)
        info.update(full_edited_tensor_file=target_path.name,
                    full_edited_tensor_file_sha256=file_sha256(target_path))
        if edit.factors is not None:
            factors_path=destination/f'factors_{i:02d}.pt'
            torch.save(edit.factors,factors_path)
            info.update(factors_file=factors_path.name,factors_file_sha256=file_sha256(factors_path))
        changed=(edit.base.view(torch.int16)!=edit.edited.view(torch.int16)).reshape(-1)
        indices=changed.nonzero(as_tuple=False).reshape(-1).to(torch.int64)
        if indices.numel()*10 < edit.edited.numel()*2:
            payload=dict(format='sparse_replacements',indices=indices,
                         values=edit.edited.reshape(-1).index_select(0,indices).clone())
        else:
            payload=dict(format='dense_target',edited=edit.edited.clone())
        path=destination/f'edit_{i:02d}.pt'
        torch.save(payload,path)
        info.update(payload_file=path.name,payload_sha256=file_sha256(path),payload_format=payload['format'])
    manifest_path=destination/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    return manifest


@torch.no_grad()
def load_edit_bundle(directory, model):
    """Reconstruct exact edited CPU tensors against verified current model bases."""
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf8'))
    if manifest['format']!='ouro_v2_exact_bf16_replacement_patch_v1' or manifest['model_revision']!=REVISION:
        raise ValueError('Unrecognized model/patch format')
    parameters=dict(model.named_parameters());edits=[]
    for info in manifest['edits']:
        name=info['tensor_name'];_validate_name(name)
        base=_base(name,parameters[name])
        if list(base.shape)!=info['shape'] or tensor_sha256(base)!=info['base_sha256']:
            raise ValueError('Patch base tensor hash/shape mismatch: '+name)
        path=(directory/info['payload_file']).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha256(path)!=info['payload_sha256']:
            raise ValueError('Patch payload path or checksum mismatch')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['format']=='sparse_replacements':
            idx,values=payload['indices'],payload['values']
            if idx.dtype!=torch.int64 or values.dtype!=torch.bfloat16 or idx.numel()!=values.numel():
                raise ValueError('Malformed sparse BF16 patch')
            if idx.ndim!=1 or values.ndim!=1 or idx.numel()==0 or int(idx.min())<0 or int(idx.max())>=base.numel() or idx.unique().numel()!=idx.numel():
                raise ValueError('Invalid patch support')
            edited=base.clone();edited.reshape(-1)[idx]=values
        elif payload['format']=='dense_target':
            edited=payload['edited'].contiguous()
            if edited.dtype!=torch.bfloat16 or edited.shape!=base.shape:
                raise ValueError('Malformed dense BF16 patch')
        else:raise ValueError('Unknown patch payload format')
        if tensor_sha256(edited)!=info['edited_sha256']:
            raise ValueError('Exact edited tensor could not be reproduced')
        target_path=(directory/info['full_edited_tensor_file']).resolve()
        if not target_path.is_relative_to(directory.resolve()) or file_sha256(target_path)!=info['full_edited_tensor_file_sha256']:
            raise ValueError('Full BF16 target file checksum/path mismatch')
        full_target=torch.load(target_path,map_location='cpu',weights_only=True)
        if full_target.dtype!=torch.bfloat16 or full_target.shape!=base.shape or tensor_sha256(full_target)!=info['edited_sha256']:
            raise ValueError('Full BF16 target disagrees with exact patch')
        factors=None
        if 'factors_file' in info:
            factors_path=(directory/info['factors_file']).resolve()
            if not factors_path.is_relative_to(directory.resolve()) or file_sha256(factors_path)!=info['factors_file_sha256']:
                raise ValueError('Factor checksum/path mismatch')
            factors=torch.load(factors_path,map_location='cpu',weights_only=True)
            if tensor_sha256(factors['U'])!=info['specification']['basis_sha256']:
                raise ValueError('Direction-factor checksum mismatch')
        edit=WeightEdit(name,base,edited,info['specification'],factors)
        if edit.stats['actual_changed_coordinates']!=info['actual_changed_coordinates']:
            raise ValueError('Changed-coordinate audit mismatch')
        edits.append(edit)
    summary=summarize_edits(edits,sum(p.numel() for p in model.parameters()))
    if summary['actual_changed_parameters']!=manifest['actual_changed_parameters'] or summary['total_unique_parameters']!=manifest['total_unique_parameters']:
        raise ValueError('Model denominator/aggregate changed count mismatch')
    return edits,manifest
