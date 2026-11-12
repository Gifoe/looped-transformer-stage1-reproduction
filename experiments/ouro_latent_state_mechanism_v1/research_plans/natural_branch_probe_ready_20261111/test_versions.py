import torch
with torch.no_grad():m=torch.nn.Linear(4,4).to('cpu').eval()
versions=[p._version for p in m.parameters()];assert len(versions)==2
with torch.inference_mode():
 m(torch.ones(1,4));assert [p._version for p in m.parameters()]==versions
with torch.inference_mode():bad=torch.ones(1)
try:bad._version
except RuntimeError:pass
else:raise AssertionError('inference tensor unexpectedly versioned')
print('MODEL_LOADER_VERSION_CPU_PASS; NOT GPU')
