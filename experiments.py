"""
Stimulus -> response experiments on the fly brain, the way the model is tested
in Shiu et al. 2024: drive a group of sensory neurons with Poisson input,
record every neuron for a second and read out the neurons that matter for
behaviour (motor neurons and descending neurons, which carry commands from the
brain to the body).

    python3 experiments.py              # all experiments
    python3 experiments.py feeding      # only the ones whose name contains "feeding"
"""
import sys
import time

import numpy as np

from fly_brain import FlyBrain

TRIALS = 5
T_MS = 1000
RATE = 150


def cells(b, cell_type, side=None):
    return b.neurons(cell_type=cell_type, side=side)


def experiments(b):
    """(name, drive, silenced, readouts). Readouts are {label: indices};
    their mean rate per neuron is reported."""
    sugar_L = cells(b, "LB3", "left")
    bitter_L = b.neurons(cell_type=["LB1a,LB1d", "LB1b", "LB1c", "LB1e"], side="left")
    loom_L = np.concatenate([cells(b, "LC4", "left"), cells(b, "LPLC2", "left")])
    loom_R = np.concatenate([cells(b, "LC4", "right"), cells(b, "LPLC2", "right")])
    jo_ce = b.neurons(cell_type=["JO-CM", "JO-CL", "JO-CA1", "JO-CA2",
                                 "JO-EV1", "JO-EV2", "JO-EV3", "JO-EV4",
                                 "JO-EV5", "JO-EV6", "JO-ED1", "JO-ED2_a",
                                 "JO-ED2_b", "JO-ED2_c"], side="left")

    feeding = {"MN9 (CB0701)": cells(b, "CB0701")}
    escape = {"giant fiber L": cells(b, "DNp01", "left"),
              "giant fiber R": cells(b, "DNp01", "right")}
    steer = {"DNa02 L": cells(b, "DNa02", "left"), "DNa02 R": cells(b, "DNa02", "right"),
             "DNa01 L": cells(b, "DNa01", "left"), "DNa01 R": cells(b, "DNa01", "right"),
             "MDN (backward)": cells(b, "MDN")}

    return [
        ("control: no stimulus", [], [], {**feeding, **escape}),
        ("feeding: sugar GRNs (LB3, left)", [(sugar_L, RATE)], [], feeding),
        ("feeding: sugar + bitter GRNs", [(sugar_L, RATE), (bitter_L, RATE)], [], feeding),
        ("feeding: bitter GRNs alone", [(bitter_L, RATE)], [], feeding),
        ("escape: looming, left eye (LC4+LPLC2)", [(loom_L, RATE)], [], {**escape, **steer}),
        ("escape: looming, both eyes", [(loom_L, RATE), (loom_R, RATE)], [], escape),
        ("escape: looming both eyes, LC4 silenced (only LPLC2 left)",
         [(loom_L, RATE), (loom_R, RATE)], cells(b, "LC4"), escape),
        ("grooming: antennal JO-C/E, left", [(jo_ce, RATE)], [], steer),
    ]


def run(b, drive, silenced, trials=TRIALS, ms=T_MS):
    b.set_silenced(silenced)
    rates = np.zeros(b.n)
    for _ in range(trials):
        b.reset()
        b.set_input(drive)
        b.run(ms)
        rates += b.spike_count / (ms * 1e-3)
    b.set_silenced([])
    return rates / trials


def main():
    t0 = time.time()
    b = FlyBrain(seed=1)
    b.run(1)
    print(f"brain: {b.n:,} neurons, {b.n_connections:,} connections, "
          f"{b.n_synapses:,} synapses, loaded in {time.time() - t0:.1f}s\n")

    desc = b.neurons(super_class="descending")
    names = b.annot["cell_type"].fillna("?").to_numpy()
    sides = b.annot["side"].fillna("?").str[0].str.upper().to_numpy()
    wanted = sys.argv[1] if len(sys.argv) > 1 else ""

    for name, drive, silenced, readouts in experiments(b):
        if wanted not in name:
            continue
        t = time.time()
        r = run(b, drive, silenced)
        n_drive = sum(len(d) for d, _ in drive)
        print(f"== {name}")
        print(f"   driven {n_drive} neurons, silenced {len(silenced)}, "
              f"active {(r > 0).sum()} neurons, {time.time() - t:.1f}s")
        for label, idx in readouts.items():
            print(f"   {label:18s} {r[idx].mean():6.1f} Hz  (n={len(idx)})")
        top = desc[np.argsort(r[desc])[::-1][:6]]
        top = [f"{names[i]}{sides[i]} {r[i]:.0f}" for i in top if r[i] > 0]
        print(f"   top descending:   {', '.join(top) or '-'}\n")


if __name__ == "__main__":
    main()
