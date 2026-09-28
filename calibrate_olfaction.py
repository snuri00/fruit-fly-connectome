"""
Calibrating the antennal lobe against an independent physiological target.

In the published model every synapse has the same strength. Inside the
antennal lobe (AL) that makes the network run away: excitatory synapses
between AL neurons (671,312 synapses, mostly from projection neurons and
cholinergic local neurons) outnumber inhibitory ones (226,478) three to one,
and stimulating the receptor neurons of a single glomerulus at 30 Hz drives
81 % of all projection neurons and 68 % of the Kenyon cells, whatever the
stimulus strength. In flies an odour activates about 5-10 % of the Kenyon
cells (Turner et al. 2008; Honegger et al. 2011).

One parameter is calibrated: a common gain for the excitatory synapses
between AL neurons (cell classes ALPN, ALLN, ALIN, ALON). The connectome is
not changed. The gain is chosen so that 20 calibration odours activate on
average 7.5 % of the Kenyon cells. These odours are generated from a seed that
the learning experiments never use; the chosen gain is written to
data/olfaction_calibration.json and then kept fixed.

    python3 calibrate_olfaction.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "olfaction_calibration.json")
GAINS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.5, 1.0]
TARGET_KC = 0.075
CALIBRATION_SEED = 777
N_ODOURS = 20
STIM_MS = 300
AL_CLASSES = ["ALPN", "ALLN", "ALIN", "ALON"]


def glomeruli(brain, side="left"):
    """ORN groups per glomerulus: {glomerulus: neuron indices}."""
    ct = brain.annot["cell_type"].fillna("").to_numpy().astype(str)
    sd = brain.annot["side"].fillna("").to_numpy()
    out = {}
    for t in sorted(set(t for t in ct if t.startswith("ORN_"))):
        idx = np.flatnonzero((ct == t) & (sd == side))
        if len(idx):
            out[t[4:]] = idx
    return out


def make_odours(glom_names, seed, n, k_range=(3, 8), rate_range=(20.0, 80.0)):
    """Random odours: each activates k glomeruli with ORN rates in rate_range."""
    rng = np.random.default_rng(seed)
    odours = []
    for _ in range(n):
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        names = rng.choice(glom_names, k, replace=False)
        odours.append({str(g): float(rng.uniform(*rate_range)) for g in names})
    return odours


def odour_drive(glom, odour):
    return [(glom[g], hz) for g, hz in odour.items()]


def al_excitatory_positions(brain):
    """Positions in W.data of excitatory synapses between AL neurons."""
    cls = brain.annot["cell_class"].fillna("").to_numpy()
    al = np.isin(cls, AL_CLASSES)
    W = brain.W
    rows = np.repeat(np.arange(brain.n), np.diff(W.indptr))
    return np.flatnonzero(al[rows] & al[W.indices] & (W.data > 0))


def apply_gain(brain, gain, pos=None, w0=None):
    """Scale the AL-internal excitatory synapses; returns (pos, w0) to undo."""
    if pos is None:
        pos = al_excitatory_positions(brain)
        w0 = brain.W.data[pos].copy()
    brain.W.data[pos] = w0 * gain
    return pos, w0


def measure(brain, glom, odours, kc, pn, ms=STIM_MS):
    kc_frac, pn_frac = [], []
    for od in odours:
        brain.reset()
        brain.set_input(odour_drive(glom, od))
        c0 = brain.spike_count.copy()
        brain.run(ms)
        d = brain.spike_count - c0
        kc_frac.append(float(np.mean(d[kc] > 0)))
        pn_frac.append(float(np.mean(d[pn] > 0)))
    brain.set_input([])
    return np.array(kc_frac), np.array(pn_frac)


def main():
    sys.path.insert(0, HERE)
    from fly_brain import FlyBrain
    b = FlyBrain(seed=CALIBRATION_SEED)
    b.run(1)
    cls = b.annot["cell_class"].fillna("").to_numpy()
    sd = b.annot["side"].fillna("").to_numpy()
    kc = np.flatnonzero((cls == "Kenyon_Cell") & (sd == "left"))
    pn = np.flatnonzero((cls == "ALPN") & (sd == "left"))
    glom = glomeruli(b)
    odours = make_odours(sorted(glom), CALIBRATION_SEED, N_ODOURS)
    pos, w0 = apply_gain(b, 1.0)
    print(f"{len(glom)} glomeruli, {len(pos):,} AL-internal excitatory connections, "
          f"{N_ODOURS} calibration odours of 3-8 glomeruli at 20-80 Hz\n")
    print(f"{'gain':>5s}  {'KCs active (mean)':>18s}  {'odours with 5-10 %':>19s}  {'PNs active':>11s}")
    rows = []
    for g in GAINS:
        apply_gain(b, g, pos, w0)
        kcf, pnf = measure(b, glom, odours, kc, pn)
        in_range = int(np.sum((kcf >= 0.05) & (kcf <= 0.10)))
        rows.append((g, kcf.mean(), in_range, pnf.mean()))
        print(f"{g:5.2f}  {100 * kcf.mean():17.1f}%  {in_range:>12d} / {N_ODOURS}  {100 * pnf.mean():10.1f}%", flush=True)
    best = min(rows, key=lambda r: abs(r[1] - TARGET_KC))
    apply_gain(b, best[0], pos, w0)
    dose = []
    name = "DM1" if "DM1" in glom else sorted(glom)[0]
    for hz in (10, 20, 40, 80):
        b.reset()
        b.set_input([(glom[name], hz)])
        c0 = b.spike_count.copy()
        b.run(STIM_MS)
        d = b.spike_count - c0
        ct = b.annot["cell_type"].fillna("").to_numpy().astype(str)
        own = np.flatnonzero((cls == "ALPN") & np.char.startswith(ct, name) & (sd == "left"))
        dose.append((hz, float(d[own].mean() / (STIM_MS / 1000)) if len(own) else float("nan")))
    print(f"\nchosen gain {best[0]} (KCs {100 * best[1]:.1f} %, target {100 * TARGET_KC:.1f} %)")
    print(f"dose-response check, glomerulus {name}: " + ", ".join(f"ORN {hz} Hz -> PN {r:.0f} Hz" for hz, r in dose))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"al_excitatory_gain": best[0], "target_kc_fraction": TARGET_KC,
                   "measured_kc_fraction": best[1], "calibration_seed": CALIBRATION_SEED,
                   "scan": [{"gain": g, "kc_fraction": k, "odours_in_5_10pct": n, "pn_fraction": p}
                            for g, k, n, p in rows],
                   "dose_response": dose}, f, indent=1)
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
