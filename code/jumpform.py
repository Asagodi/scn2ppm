"""Do the jump sizes have a closed form in the decoders?

A reset is y -> y - D_j.  If two sides of a word boundary differ by which
neuron fires, the landing points on B_0 differ by the tangential component of
a decoder, or of a difference of decoders.  Compare the measured jumps with
every such candidate.
"""
import os, sys, itertools
import numpy as np

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn

here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, 'k3_chaotic.npz'))

for i in range(3):
    D = z['D'][i]
    meas = np.unique(np.round(np.load(os.path.join(here, f'jumps_net{i}.npz'))['bj'], 9))
    print(f"\n=== net {i} ===")
    print("  measured distinct |jump|:", np.array2string(meas, precision=6))
    cands = {}
    for b in range(3):                       # candidates on each bound B_b
        E = scn.tangent_frame(D[b])[:, 0]
        for j in range(3):
            cands[f"|E_{b}.D_{j}|"] = abs(E @ D[j])
        for j, k in itertools.combinations(range(3), 2):
            cands[f"|E_{b}.(D_{j}-D_{k})|"] = abs(E @ (D[j] - D[k]))
            cands[f"|E_{b}.(D_{j}+D_{k})|"] = abs(E @ (D[j] + D[k]))
    for m in meas:
        hit = [(k, v) for k, v in cands.items() if abs(v - m) < 1e-6]
        near = sorted(cands.items(), key=lambda kv: abs(kv[1] - m))[:3]
        if hit:
            print(f"  jump {m:.6f}  ==  " + ", ".join(f"{k} = {v:.6f}" for k, v in hit))
        else:
            print(f"  jump {m:.6f}  no exact match; nearest: "
                  + ", ".join(f"{k}={v:.6f}" for k, v in near))
