"""
Does the fly brain model learn? A differential conditioning experiment, the
model version of aversive conditioning in a T-maze (Tully & Quinn 1985) and of
MBON recordings before and after pairing (Hige et al. 2015).

Two visual stimuli reach the mushroom body through different visual projection
neurons of the left hemisphere (A: aMe12, aMe20, LTe16; B: MTe30, MTe32, MTe40,
LTe25). They activate about 30 and 50 Kenyon cells with an overlap of 7
(Jaccard 0.08). During training the conditioned stimulus CS+ is presented
together with the punishment dopamine neurons PPL1 (the model's electric
shock); CS- is presented alone.

Readout: the mean membrane depolarisation of every MBON during a 1 s stimulus,
the model counterpart of calcium or voltage imaging. The MBONs that respond
(more than 0.3 mV to either stimulus) are chosen in a separate selection test
before the pre-test, so that choosing them does not bias the baseline
(regression to the mean). Each test is the mean of 3 alternating presentations of A and B.
The memory score of a fly is

    score = sum over those MBONs of (change to CS-) - (change to CS+)

in mV, positive when the response to CS+ is specifically reduced, and also as a
percentage of the pre-test response to CS+.

Behaviour, the counterpart of a T-maze choice: an MBON's valence is opposite
to that of the dopamine in its compartment (Aso et al. 2014), so an MBON
whose dopamine input comes from PPL1 (punishment) promotes approach and one
driven by PAM (reward) promotes avoidance. Its weight is its PPL1 share minus
its PAM share of DAN->MBON synapses. The approach drive of a stimulus is the
weighted sum of all MBON responses to it, and at each test the fly "walks
into" the arm whose stimulus has the higher approach drive. Preference index
PI = 100 x (flies choosing CS- - flies choosing CS+) / flies.

Nothing in the model is tuned for this test. The neuron and synapse
parameters are the published ones (Shiu et al. 2024). The learning rule and its
constants are those of fly_learning.MushroomBody as used for Fly DOOM
(eta 0.02, eligibility 1.5 s, recovery 60 s); the dopamine neurons' fast
synapses are removed because dopamine acts as a modulator.

Conditions (each fly: pre-test, 6 training cycles, tests 8 s, 90 s and 240 s
after training; half of the flies have A as CS+, half B):

    paired         CS+ with PPL1, CS- alone
    unpaired       CS+ alone, PPL1 alone 8 s later, CS- alone
    cs_only        CS+ and CS- alone
    da_only        PPL1 alone, no stimulus
    no_plasticity  paired, learning rate 0
    reversal       paired, then 6 cycles with the roles swapped

    python3 conditioning.py --flies 16 --procs 6
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np

CS_TYPES = {"A": ["aMe12", "aMe20", "LTe16"], "B": ["MTe32", "MTe30", "LTe25", "MTe40"]}
CS_RATE = 120.0
DA_RATE = 150.0
STIM_MS = 1000
GAP_MS = 8000
CYCLES = 6
RESPONSIVE_MV = 0.3
TEST_REPEATS = 3
TEST_DELAYS_S = (8, 90, 240)
CONDITIONS = ["paired", "unpaired", "cs_only", "da_only", "no_plasticity", "reversal"]

_fly = None


class Fly:
    def __init__(self):
        from fly_brain import FlyBrain
        from fly_learning import MushroomBody
        self.brain = FlyBrain(seed=0)
        self.brain.run(1)
        self.learner = MushroomBody(self.brain)
        b = self.brain
        cls = b.annot["cell_class"].to_numpy()
        self.mbon = np.flatnonzero(cls == "MBON")
        self.cs = {k: b.neurons(cell_type=v, side="left") for k, v in CS_TYPES.items()}
        self.ppl1 = self.learner.ppl1
        self.eta0 = self.learner.eta
        n_pam = len(self.learner.pam)
        comp = self.learner.comp
        self.valence = comp[n_pam:].sum(0) - comp[:n_pam].sum(0)

    def new_fly(self, seed, plastic=True):
        self.brain.rng = np.random.default_rng(seed)
        self.learner.reset_weights()
        self.learner.eta = self.eta0 if plastic else 0.0
        self.brain.reset()

    def rest(self, ms):
        """Silence: the model has no spontaneous activity, so only the
        eligibility traces decay and depressed synapses recover."""
        L = self.learner
        ms = max(ms, 0.0)
        L.elig *= math.exp(-ms / L.tau_elig)
        W = self.brain.W
        w = W.data[L.pos]
        W.data[L.pos] = L.w0 + (w - L.w0) * math.exp(-ms / L.tau_recover)
        self.brain.reset()

    def present(self, cs=None, dopamine=False, ms=STIM_MS):
        b = self.brain
        drive = []
        if cs:
            drive.append((self.cs[cs], CS_RATE))
        if dopamine:
            drive.append((self.ppl1, DA_RATE))
        b.set_input(drive)
        vm = np.zeros(len(self.mbon))
        for _ in range(ms // 10):
            counts = np.zeros(b.n)
            for _ in range(10):
                sp = b.run(1, record=True)
                if len(sp):
                    np.add.at(counts, sp[:, 1], 1)
                vm += b.v[self.mbon] - b.p["v_0"]
            self.learner.update(counts, 10.0)
        b.set_input([])
        return vm / ms

    def test(self):
        """Mean response to each stimulus over TEST_REPEATS alternating presentations."""
        out = {"A": 0.0, "B": 0.0}
        for _ in range(TEST_REPEATS):
            for cs in ("A", "B"):
                out[cs] = out[cs] + self.present(cs) / TEST_REPEATS
                self.rest(GAP_MS)
        return out

    def train(self, cs_plus, cs_minus, condition):
        for _ in range(CYCLES):
            if condition in ("paired", "no_plasticity", "reversal"):
                self.present(cs_plus, dopamine=True)
                self.rest(GAP_MS)
            elif condition == "unpaired":
                self.present(cs_plus)
                self.rest(GAP_MS)
                self.present(None, dopamine=True)
                self.rest(GAP_MS)
            elif condition == "cs_only":
                self.present(cs_plus)
                self.rest(GAP_MS)
            elif condition == "da_only":
                self.present(None, dopamine=True)
                self.rest(GAP_MS)
            if condition != "da_only":
                self.present(cs_minus)
                self.rest(GAP_MS)


def score(pre, post, cs_plus, cs_minus, responsive):
    d_plus = (post[cs_plus] - pre[cs_plus])[responsive].sum()
    d_minus = (post[cs_minus] - pre[cs_minus])[responsive].sum()
    base = max(pre[cs_plus][responsive].sum(), 1e-6)
    return float(d_minus - d_plus), float(100 * (d_minus - d_plus) / base)


def choice(fly, resp, cs_plus, cs_minus):
    """+1 if the fly would walk towards CS-, -1 towards CS+."""
    a_minus = float(resp[cs_minus] @ fly.valence)
    a_plus = float(resp[cs_plus] @ fly.valence)
    return 1 if a_minus > a_plus else -1


def as_lists(resp):
    return {k: [round(float(x), 4) for x in v] for k, v in resp.items()}


def run_fly(job):
    global _fly
    if _fly is None:
        _fly = Fly()
    condition, seed, cs_plus = job
    cs_minus = "B" if cs_plus == "A" else "A"
    fly = _fly
    fly.new_fly(seed, plastic=(condition != "no_plasticity"))
    t0 = time.time()
    select = fly.test()
    responsive = (select["A"] > RESPONSIVE_MV) | (select["B"] > RESPONSIVE_MV)
    pre = fly.test()
    fly.train(cs_plus, cs_minus, condition)
    result = {"condition": condition, "seed": seed, "cs_plus": cs_plus,
              "n_responsive": int(responsive.sum()), "scores": {}, "choices": {},
              "choice_pre": choice(fly, pre, cs_plus, cs_minus), "pre": as_lists(pre), "post": {},
              "depressed": fly.learner.stats()["depressed"]}
    since = GAP_MS / 1000
    for delay in TEST_DELAYS_S:
        fly.rest((delay - since) * 1000)
        post = fly.test()
        since = delay + 2 * TEST_REPEATS * (STIM_MS + GAP_MS) / 1000
        result["scores"][delay] = score(pre, post, cs_plus, cs_minus, responsive)
        result["choices"][delay] = choice(fly, post, cs_plus, cs_minus)
        result["post"][delay] = as_lists(post)
        if delay == TEST_DELAYS_S[0]:
            first_post = post
    if condition == "reversal":
        fly.train(cs_minus, cs_plus, "paired")
        fly.rest(TEST_DELAYS_S[0] * 1000 - GAP_MS)
        post = fly.test()
        result["reversed"] = score(first_post, post, cs_plus, cs_minus, responsive)
        result["choice_reversed"] = choice(fly, post, cs_plus, cs_minus)
    result["wall_s"] = round(time.time() - t0, 1)
    return result


def summarise(results):
    from scipy import stats
    print(f"\n{'condition':15s} {'n':>3s}  " + "  ".join(f"{'score +' + str(d) + ' s':>18s}" for d in TEST_DELAYS_S)
          + "   p (+8 s, >0)")
    first = {}
    for cond in CONDITIONS:
        rows = [r for r in results if r["condition"] == cond]
        if not rows:
            continue
        cols = []
        for d in TEST_DELAYS_S:
            mv = np.array([r["scores"][str(d)][0] if str(d) in r["scores"] else r["scores"][d][0] for r in rows])
            pc = np.array([r["scores"][str(d)][1] if str(d) in r["scores"] else r["scores"][d][1] for r in rows])
            cols.append(f"{mv.mean():6.2f} mV {pc.mean():5.0f}%")
            if d == TEST_DELAYS_S[0]:
                first[cond] = mv
        p = stats.wilcoxon(first[cond], alternative="greater").pvalue if np.any(first[cond] != 0) else 1.0
        print(f"{cond:15s} {len(rows):3d}  " + "  ".join(f"{c:>18s}" for c in cols) + f"   {p:.2g}")
    if "paired" in first:
        print("\npaired vs each control at +8 s (Mann-Whitney, one-sided):")
        for cond in CONDITIONS[1:5]:
            if cond in first:
                p = stats.mannwhitneyu(first["paired"], first[cond], alternative="greater").pvalue
                print(f"   paired > {cond:14s} p = {p:.2g}")
    print(f"\n{'behaviour':15s} {'PI before':>10s} {'PI +8 s':>9s} {'PI +90 s':>9s}   p (+8 s, sign test)")
    d0 = str(TEST_DELAYS_S[0])
    for cond in CONDITIONS:
        rows = [r for r in results if r["condition"] == cond]
        if not rows:
            continue
        pre_pi = 100 * np.mean([r["choice_pre"] for r in rows])
        post = [r["choices"][d0] for r in rows]
        pis = [100 * np.mean([r["choices"][str(d)] for r in rows]) for d in TEST_DELAYS_S[:2]]
        p = stats.binomtest(sum(c > 0 for c in post), len(post), 0.5, alternative="greater").pvalue
        print(f"{cond:15s} {pre_pi:10.0f} {pis[0]:9.0f} {pis[1]:9.0f}   {p:.2g}")
    rev = [r for r in results if r["condition"] == "reversal" and "reversed" in r]
    if rev:
        before = np.array([r["scores"][str(TEST_DELAYS_S[0])][0] if str(TEST_DELAYS_S[0]) in r["scores"]
                           else r["scores"][TEST_DELAYS_S[0]][0] for r in rev])
        after = np.array([r["reversed"][0] for r in rev])
        p = stats.wilcoxon(after, alternative="less").pvalue
        print(f"\nreversal: score for the original CS+ after first training {before.mean():+.2f} mV, "
              f"after reversal training {after.mean():+.2f} mV (relative to the first post-test; <0 = reversed, p = {p:.2g})")
        pi_rev = 100 * np.mean([r["choice_reversed"] for r in rev])
        print(f"reversal: PI for the original CS- after reversal {pi_rev:.0f} (negative = the flies now avoid it)")


def main():
    ap = argparse.ArgumentParser(description="differential conditioning of the fly brain model")
    ap.add_argument("--flies", type=int, default=16, help="flies per condition (even)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "recordings", "conditioning.json"))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    args = ap.parse_args()
    jobs = [(c, 1000 + i, "A" if i % 2 == 0 else "B")
            for c in args.conditions.split(",") for i in range(args.flies)]
    t0 = time.time()
    results = []
    with Pool(args.procs) as pool:
        for r in pool.imap_unordered(run_fly, jobs):
            results.append(r)
            s = r["scores"][TEST_DELAYS_S[0]]
            print(f"{r['condition']:14s} fly {r['seed']} CS+={r['cs_plus']}  score +8 s {s[0]:+6.2f} mV ({s[1]:+4.0f}%)"
                  f"  responsive MBONs {r['n_responsive']}  depressed synapses {r['depressed']}  {r['wall_s']} s",
                  flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f)
    results = json.load(open(args.out))
    summarise(results)
    print(f"({time.time() - t0:.0f} s, results in {args.out})")


if __name__ == "__main__":
    main()
