"""Causal current/history-only exits and exact native BF16 threshold semantics."""
import numpy as np

LOWER={'entropy','choice_entropy','full_vocab_kl','choice_kl','full_vocab_logit_l2','choice_logit_l2','hidden_absolute_l2','hidden_relative_l2'}
METRICS=['confidence','entropy','logit_margin','full_vocab_kl','full_vocab_logit_l2','hidden_absolute_l2','hidden_relative_l2','choice_confidence','choice_entropy','choice_logit_margin','choice_kl','choice_logit_l2','native_gate_cdf']

def bf16_scalar(value):
    # Round-to-nearest, ties-to-even, matching torch.tensor(value, bfloat16).
    bits=np.asarray([value],dtype=np.float32).view(np.uint32)
    bits=(bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)
    return float(bits.view(np.float32)[0])

def exits(value,metric,threshold):
    if value is None:return False
    threshold=bf16_scalar(threshold) if metric=='native_gate_cdf' else threshold
    return value<=threshold if metric in LOWER else value>=threshold

def exit_index(trajectory,metric,threshold):
    for index,state in enumerate(trajectory[:-1]):
        if exits(state.get(metric),metric,threshold):return index
    return len(trajectory)-1
