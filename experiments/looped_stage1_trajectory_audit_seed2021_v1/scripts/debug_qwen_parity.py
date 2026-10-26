from qwen_audit import load,parity,Budget
budget=Budget()
try:
    tok,m,recipe=load()
    parity(tok,m,recipe)
finally:budget.close('qwen_parity_diagnostic')
