"""CPU mock test of loop/layer indexing and intervention, NOT an Ouro test."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from latent_hooks import LayerRecorder,Patch
from core import target_log_likelihoods

class MockLayer(nn.Module):
    def __init__(self,h):
        super().__init__();self.weight=nn.Parameter(torch.randn(h)*.01)
    def forward(self,x):
        return x+torch.tanh(x+self.weight)*.01

class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Embedding(32,8);self.layers=nn.ModuleList(MockLayer(8) for _ in range(24))
    def forward(self,input_ids,use_cache=False):
        h=self.embed(input_ids)
        states=[]
        for _ in range(4):
            for layer in self.layers:h=layer(h)
            states.append(h)
        return None,states,None

class MockLM(nn.Module):
    def __init__(self):
        super().__init__();self.model=MockBackbone();self.lm_head=nn.Linear(8,32,bias=False)

class TestLoopHooks(unittest.TestCase):
    def test_capture_and_identity_patch(self):
        torch.manual_seed(1);m=MockLM().eval();x=torch.tensor([[2,3,4,5,6]])
        with torch.inference_mode():
            with LayerRecorder(m,[0,2,4],True) as rec:
                _,states,_=m.model(x)
            arr=rec.raw_uint16()
            self.assertEqual(arr.shape,(4,24,3,8))
            self.assertEqual(rec.calls,[4]*24)
            stats=rec.statistics()
            self.assertEqual(len(stats['relative_l2_by_transition_and_layer']),3)
            with LayerRecorder(m,[0,2,4],True,Patch(12,1,2,1.0)) as ident:
                _,same,_=m.model(x)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(states,same)))
            with LayerRecorder(m,[0,2,4],True,Patch(12,1,2,0.0)) as patch:
                _,modified,_=m.model(x)
            self.assertFalse(torch.equal(states[-1],modified[-1]))
            scores,heads=target_log_likelihoods(m,states,[4],[9])
            self.assertEqual(len(scores),4)
            self.assertEqual(heads.shape,(4,32))

if __name__=='__main__': unittest.main()
