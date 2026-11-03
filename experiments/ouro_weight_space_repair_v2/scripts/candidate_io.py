"""Materialize a frozen V2 candidate; no model load, fitting, or device policy."""
import json
from pathlib import Path
import torch
from weight_edits import (ROOT, file_sha256, tensor_sha256,
                          low_rank_projection, sparse_column_attenuation)


def build_candidate(candidate_spec, model):
    """Return actual BF16 WeightEdits. Apply using weight_edits.applied_edits.

    The caller must invalidate all reusable caches on apply AND restore. Persist
    the selected full BF16 tensors with save_edit_bundle for bit-exact replay.
    """
    spec = candidate_spec
    path = (ROOT / spec['directions_file']).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Directions must be inside V2')
    if file_sha256(path) != spec['directions_sha256']:
        raise ValueError('Frozen direction file hash mismatch')
    bases = torch.load(path, map_location='cpu', weights_only=True)
    parameters = dict(model.named_parameters())
    edits = []
    for item in spec['edits']:
        name = item['name']; weight = parameters[name]
        if tensor_sha256(weight) != item['base_tensor_sha256']:
            raise ValueError('Candidate source parameter mismatch: '+name)
        provenance = dict(candidate_id=spec['id'], basis_key=item['basis_key'],
                          directions_sha256=spec['directions_sha256'],
                          discovery_provenance_sha256=spec['discovery_provenance_sha256'])
        if item['mode'] == 'projection':
            basis = bases[item['basis_key']]
            if tensor_sha256(basis) != item['basis_sha256']:
                raise ValueError('Frozen basis tensor mismatch')
            edit = low_rank_projection(name, weight, basis, item['alpha'],
                                       columns=item['columns'], provenance=provenance)
        elif item['mode'] == 'attenuation':
            edit = sparse_column_attenuation(name, weight, item['columns'],
                                              item['alpha'], provenance=provenance)
        else:
            raise ValueError('Unknown edit mode')
        edits.append(edit)
    if not 1 <= len(edits) <= 2 or len({e.name for e in edits}) != len(edits):
        raise ValueError('Candidate must contain one or two distinct physical edits')
    return edits
