"""Two continuous phases of the checkpoint's loop-end RMSNorm."""
import math
import torch


def radial(x,t,epsilon):
    log_e=math.log(epsilon)
    fraction=t if log_e==0 else math.expm1(t*log_e)/math.expm1(log_e)
    return x*torch.rsqrt(math.exp(t*log_e)+x.square().mean(-1,keepdim=True)*fraction)


class NormFlow:
    def __init__(self,norm):
        self.epsilon=float(norm.variance_epsilon)
        self.gamma=norm.weight.detach().float()
        assert self.epsilon>0 and torch.all(self.gamma>0), "Positive gamma is required for this chart"

    def at(self,x,time):
        h=x.float() if x.dtype!=torch.float64 else x
        if time<=1:
            return radial(h,time,self.epsilon)
        return radial(h,1.,self.epsilon)*torch.exp((time-1)*self.gamma.to(h.dtype).log())

    def __call__(self,x):
        return self.at(x,2.)
