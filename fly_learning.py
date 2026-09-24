"""
Dopamine-gated plasticity for the fly brain. Rewards and punishments are
delivered the way they reach a real fly: by activating dopaminergic neurons.
PAM neurons signal reward, PPL1 neurons punishment. Both learning rules below
read the dopamine signal from the spikes of those neurons in the model.

The Shiu et al. model treats dopamine like a fast excitatory transmitter. That
makes every reward excite thousands of Kenyon cells directly, so everything
looks "active before the reward" and is unlearned at once. Dopamine is a slow
modulator, so `make_dopamine_modulatory` removes the fast synapses of all
dopaminergic neurons; their spikes still carry the teaching signal.

MushroomBody (the fly's own learning rule)
    Kenyon cells (KCs) code the sensory context sparsely. A KC that was active
    shortly before dopamine arrives in an MBON's compartment has its synapse
    onto that MBON depressed (Hige et al. 2015; Handler et al. 2019):

        dw = -eta * eligibility(KC) * dopamine(MBON compartment) * w0

    The eligibility is a low-pass trace of the KC's spikes (tau 1.5 s).
    Compartments come from the connectome: an MBON's dopamine is the spikes of
    the DANs that synapse onto it, weighted by synapse count. Depressed weights
    slowly recover (tau 60 s). Both PAM and PPL1 depress; which behaviour
    changes depends on which MBONs lie in their compartments.

SteeringSynapses (control, not known to exist in flies)
    Excitatory synapses onto the steering neurons DNa01/DNa02 carry a Hebbian
    eligibility (pre and post spiking in the same 28.6 ms game tic, decaying
    with tau 0.5 s). Reward strengthens and punishment weakens eligible
    synapses:

        dw = eta * (PAM - PPL1) * eligibility * w0,   0 <= w <= 3 w0
"""
import math

import numpy as np


def dan_groups(brain):
    ct = brain.annot["cell_type"].fillna("").to_numpy().astype(str)
    return (np.flatnonzero(np.char.startswith(ct, "PAM")),
            np.flatnonzero(np.char.startswith(ct, "PPL1")))


def make_dopamine_modulatory(brain):
    """Zero the outgoing synapses of every dopaminergic neuron (cell class DAN;
    the predicted transmitter is not used, since it labels most Kenyon cells
    as dopaminergic too). The original
    weights are kept in brain.dopamine_synapses (position -> weight), since
    they still define which compartments each neuron reaches. Idempotent."""
    if getattr(brain, "dopamine_synapses", None) is None:
        rows = np.flatnonzero(brain.annot["cell_class"].to_numpy() == "DAN")
        pos = np.concatenate([np.arange(brain.W.indptr[i], brain.W.indptr[i + 1]) for i in rows])
        brain.dopamine_synapses = dict(zip(pos.tolist(), brain.W.data[pos].tolist()))
    brain.W.data[list(brain.dopamine_synapses)] = 0.0


def rows_to_cols(W, rows, cols):
    """Positions in W.data of the synapses from `rows` onto `cols`."""
    starts, ends = W.indptr[rows], W.indptr[rows + 1]
    lens = ends - starts
    pos = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
    return pos[np.isin(W.indices[pos], cols)]


def row_of(W, pos):
    return np.searchsorted(W.indptr, pos, side="right") - 1


class MushroomBody:
    name = "mushroom body"

    def __init__(self, brain, eta=0.02, tau_elig=1500.0, tau_recover=60000.0):
        self.brain = brain
        make_dopamine_modulatory(brain)
        W = brain.W
        cls = brain.annot["cell_class"].to_numpy()
        self.kc = np.flatnonzero(cls == "Kenyon_Cell")
        self.mbon = np.flatnonzero(cls == "MBON")
        self.pam, self.ppl1 = dan_groups(brain)
        self.dan = np.concatenate([self.pam, self.ppl1])

        self.pos = rows_to_cols(W, self.kc, self.mbon)
        kc_of = {k: i for i, k in enumerate(self.kc)}
        mbon_of = {m: i for i, m in enumerate(self.mbon)}
        self.pre = np.array([kc_of[k] for k in row_of(W, self.pos)])
        self.post = np.array([mbon_of[m] for m in W.indices[self.pos]])
        self.w0 = W.data[self.pos].copy()

        comp = np.zeros((len(self.dan), len(self.mbon)))
        dan_pos = rows_to_cols(W, self.dan, self.mbon)
        dan_of = {d: i for i, d in enumerate(self.dan)}
        for p in dan_pos:
            comp[dan_of[row_of(W, p)], mbon_of[W.indices[p]]] += abs(brain.dopamine_synapses.get(int(p), 0.0))
        total = comp.sum(axis=0)
        self.comp = np.divide(comp, total, out=np.zeros_like(comp), where=total > 0)

        self.eta, self.tau_elig, self.tau_recover = eta, tau_elig, tau_recover
        self.elig = np.zeros(len(self.kc))
        self.last_da = np.zeros(len(self.mbon))

    def reset_weights(self):
        self.brain.W.data[self.pos] = self.w0
        self.elig[:] = 0

    def update(self, counts, ms):
        W = self.brain.W
        self.elig *= math.exp(-ms / self.tau_elig)
        self.elig += counts[self.kc]
        da = counts[self.dan] @ self.comp
        self.last_da = da
        w = W.data[self.pos]
        if da.any():
            w = w - self.eta * self.elig[self.pre] * da[self.post] * self.w0
        w = w + (self.w0 - w) * (ms / self.tau_recover)
        W.data[self.pos] = np.clip(w, 0.0, self.w0)

    def stats(self):
        w = self.brain.W.data[self.pos]
        return {"synapses": len(self.pos), "mean_w": float(w.sum() / self.w0.sum()),
                "depressed": int(np.sum(w < 0.9 * self.w0)),
                "kc_active": int(np.sum(self.elig > 0.05))}


class SteeringSynapses:
    name = "steering synapses"

    def __init__(self, brain, eta=0.004, tau_elig=500.0, w_max=3.0):
        self.brain = brain
        make_dopamine_modulatory(brain)
        W = brain.W
        self.targets = np.concatenate([brain.neurons(cell_type=t) for t in ("DNa01", "DNa02")])
        pos = np.flatnonzero(np.isin(W.indices, self.targets))
        pos = pos[W.data[pos] > 0]
        self.pos = pos
        self.pre = row_of(W, pos)
        self.post = W.indices[pos]
        self.w0 = W.data[pos].copy()
        self.pam, self.ppl1 = dan_groups(brain)
        self.eta, self.tau_elig, self.w_max = eta, tau_elig, w_max
        self.elig = np.zeros(len(pos))
        self.last_da = 0.0

    def reset_weights(self):
        self.brain.W.data[self.pos] = self.w0
        self.elig[:] = 0

    def update(self, counts, ms):
        W = self.brain.W
        self.elig *= math.exp(-ms / self.tau_elig)
        self.elig += np.minimum(counts[self.pre], 3) * np.minimum(counts[self.post], 3)
        da = counts[self.pam].mean() - counts[self.ppl1].mean()
        self.last_da = da
        if da != 0:
            w = W.data[self.pos] + self.eta * da * self.elig * self.w0
            W.data[self.pos] = np.clip(w, 0.0, self.w_max * self.w0)

    def stats(self):
        w = self.brain.W.data[self.pos]
        return {"synapses": len(self.pos), "mean_w": float(w.sum() / self.w0.sum()),
                "changed": int(np.sum(np.abs(w - self.w0) > 0.1 * self.w0))}


LEARNERS = {"mb": MushroomBody, "steer": SteeringSynapses}
