"""Equal-width summaries of aligned full-width K1 token states."""
import numpy as np


def decode_bf16(bits):
    return (bits.astype(np.uint32)<<16).view(np.float32)


def token_weights(base, ids, selected, bridge):
    """Return four masked normalized pooling heads over a valid span."""
    site=np.asarray(selected)
    x=base[site]
    n=len(site)
    mean=np.ones(n,dtype=np.float32)/n
    position=np.linspace(-1,1,n,dtype=np.float32)
    position=(1+.75*position)/(1+.75*position).sum()
    frequency=np.bincount(ids)
    local=np.sqrt(frequency[ids[site]])
    repeat=local/local.sum()
    anchor=base[bridge].mean(0)
    affinity=(x@anchor)/(np.linalg.norm(x,axis=1)*max(np.linalg.norm(anchor),1e-6)+1e-6)
    affinity=np.exp(2*affinity-np.max(2*affinity))
    affinity=affinity/affinity.sum()
    return np.stack((mean,position,repeat,affinity))


def pool(vec, idx, weights=None):
    selected=vec[idx]
    if weights is None:
        return selected.mean(0)
    return np.concatenate([w @ part for w,part in zip(weights,np.split(selected,4,axis=-1))])


def summarize(base, other_s, other_m, ids, locs, source, mapping, token_aware):
    def weights(where):
        if not token_aware:
            return None
        return token_weights(base,ids,where,locs[1])
    s,m=weights(source),weights(mapping)
    patch=np.stack((pool(other_s-base,source,s),pool(other_m-base,mapping,m)))
    centered=np.stack([pool(base,site,weights(site)) for site in locs])
    state=np.stack((pool(base,source,s),pool(base,mapping,m),centered[4]))
    return patch,state,centered


def masked_context(raw):
    return raw.reshape(*raw.shape[:-2],-1)
