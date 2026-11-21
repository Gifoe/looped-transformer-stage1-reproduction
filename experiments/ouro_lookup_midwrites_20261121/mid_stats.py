"""Paired exact endpoint effects for the fixed middle layer replay."""

import numpy as np

from edge_stats import upper_binomial
from onset_deck import DOMAINS

MODES = {"attention": ("a_write",), "mlp": ("m_write",), "both": ("a_write", "m_write")}
DIRECTIONS = ("rescue", "damage")
ARMS = ("authentic", "geometry", "unrelated")
FAMILY = len(DOMAINS) * len(MODES) * (len(ARMS) - 1)


def contrast(a, b):
    delta = np.array(a, dtype=np.int8) - np.array(b, dtype=np.int8)
    extra, loss = int((delta > 0).sum()), int((delta < 0).sum())
    rng = np.random.default_rng(261127189)
    scores = np.concatenate([delta[rng.integers(len(delta), size=(1000, len(delta)))].mean(axis=1) for _ in range(40)])
    alpha = .05 / (2 * FAMILY)
    lo, hi = np.quantile(scores, [alpha, 1-alpha]).tolist()
    return dict(extra=extra, loss=loss, mean=float(delta.mean()), simultaneous=[lo, hi],
                upper_97_5=upper_binomial(extra, len(delta)),
                upper_simultaneous=upper_binomial(extra, len(delta), alpha=.025 / FAMILY))


def summarize(rows):
    domains = {}
    for domain in DOMAINS:
        q = [r for r in rows if r["domain"] == domain]
        modes = {}
        for mode in MODES:
            counts, both = {}, {}
            for a in ARMS:
                rescue = [r["pred"][mode]["rescue"][a]["number"] == r["target"] for r in q]
                damage = [r["pred"][mode]["damage"][a]["number"] == r["old"] for r in q]
                still = [r["pred"][mode]["damage"][a]["number"] == r["target"] for r in q]
                both[a] = [x and y for x,y in zip(rescue, damage, strict=True)]
                counts[a] = dict(rescue=sum(rescue), damage=sum(damage), dual=sum(both[a]), damage_still_correct=sum(still))
            tests = {a: contrast(both["authentic"], both[a]) for a in ARMS[1:]}
            supported = counts["authentic"]["dual"] >= 8 and all(x["simultaneous"][0] > 0 for x in tests.values())
            modes[mode] = dict(n=len(q), counts=counts, contrasts=tests, supported=supported)
        domains[domain] = modes
    supports = [m for m in MODES if all(domains[d][m]["supported"] for d in DOMAINS)]
    largest = max(c["upper_simultaneous"] for d in domains.values() for m in d.values() for c in m["contrasts"].values())
    audit = [r["audit"] for r in rows if "audit" in r]
    verify = dict(n=len(audit), checks=sum(a["checks"] for a in audit),
                  max_identity=max(a["identity"] for a in audit), max_suffix=max(a["suffix"] for a in audit))
    return dict(domains=domains, supported=supports, stop=largest < .10, max_upper=largest, audit=verify)
