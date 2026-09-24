"""
A leaky integrate-and-fire model of the whole adult Drosophila brain
(FlyWire v783 connectome, 138,639 neurons, 15 million connections).

The neuron model, its constants and the synapse rule are the ones of
Shiu et al. 2024 (Nature 634, 210-219), reimplemented with numpy so the brain
can be stepped tick by tick inside a game loop instead of running whole trials
in Brian2:

    dv/dt = (v_0 - v + g) / t_mbr        membrane potential
    dg/dt = -g / tau                      synaptic conductance (as a voltage)

A neuron whose v crosses v_th spikes, is reset (v = v_rst, g = 0) and is
refractory for t_rfc. After a delay t_dly each spike adds
w_syn x (signed synapse count) to the g of every postsynaptic neuron.
As in Brian2, input that reaches a neuron while it is refractory is lost.
The sign comes from the predicted neurotransmitter (Eckstein et al. 2024):
acetylcholine excites, GABA and glutamate inhibit.

Stimulated neurons receive Poisson input strong enough that every input event
triggers a spike, like optogenetic activation in the paper.
"""
import os
from collections import deque

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "Drosophila_brain_model")
DEFAULT_COMP = os.path.join(MODEL_DIR, "Completeness_783.csv")
DEFAULT_CON = os.path.join(MODEL_DIR, "Connectivity_783.parquet")
DEFAULT_ANNOT = os.path.join(HERE, "data", "neuron_annotations.tsv")
ANNOT_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
             "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv")

PARAMS = {
    "v_0": -52.0,
    "v_rst": -52.0,
    "v_th": -45.0,
    "t_mbr": 20.0,
    "tau": 5.0,
    "t_rfc": 2.2,
    "t_dly": 1.8,
    "w_syn": 0.275,
    "f_poi": 250,
}


@njit(cache=True)
def _integrate(t, v, g, ref_until, v_0, v_th, e_m, e_s, c_vg, fired):
    """Exact linear update of v and g for non-refractory neurons, then threshold.
    Writes spiking indices into `fired` and returns how many there are."""
    k = 0
    for i in range(v.shape[0]):
        if ref_until[i] >= t:
            continue
        vi = v_0 + (v[i] - v_0) * e_m + g[i] * c_vg
        g[i] *= e_s
        v[i] = vi
        if vi > v_th:
            fired[k] = i
            k += 1
    return k


@njit(cache=True)
def _deliver(t, pre, indptr, indices, data, g, ref_until):
    """g[post] += w for every outgoing synapse of every neuron in `pre`.
    Refractory targets drop the input, which is what Brian2 does with
    variables marked "(unless refractory)" in Shiu et al.'s model."""
    for p in pre:
        for s in range(indptr[p], indptr[p + 1]):
            j = indices[s]
            if ref_until[j] < t:
                g[j] += data[s]


@njit(cache=True)
def _reset(t, fired, v, g, ref_until, v_rst, rfc_steps, driven):
    for i in fired:
        v[i] = v_rst
        g[i] = 0.0
        ref_until[i] = t if driven[i] else t + rfc_steps - 1


class FlyBrain:
    def __init__(self, dt=0.1, params=None, comp_path=DEFAULT_COMP,
                 con_path=DEFAULT_CON, annot_path=DEFAULT_ANNOT, seed=None):
        self.p = dict(PARAMS, **(params or {}))
        p = self.p
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        self.root_ids = pd.read_csv(comp_path, index_col=0).index.to_numpy()
        n = self.n = len(self.root_ids)
        self.idx = {r: i for i, r in enumerate(self.root_ids)}

        con = pd.read_parquet(con_path, columns=[
            "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        self.W = sp.csr_matrix(
            (con["Excitatory x Connectivity"].to_numpy(np.float64) * p["w_syn"],
             (con["Presynaptic_Index"].to_numpy(), con["Postsynaptic_Index"].to_numpy())),
            shape=(n, n))
        self.n_connections = self.W.nnz
        self.n_synapses = int(np.abs(con["Excitatory x Connectivity"]).sum())
        self._w_intact = None
        self.silenced = np.empty(0, dtype=np.int64)
        del con

        self.annot = self._load_annotations(annot_path)

        e_m = np.exp(-dt / p["t_mbr"])
        e_s = np.exp(-dt / p["tau"])
        self._e_m, self._e_s = e_m, e_s
        self._c_vg = p["tau"] / (p["tau"] - p["t_mbr"]) * (e_s - e_m)
        self._rfc_steps = int(round(p["t_rfc"] / dt))
        self._dly_steps = int(round(p["t_dly"] / dt))
        self._w_poi = p["f_poi"] * p["w_syn"]
        self.reset()

    def _load_annotations(self, path):
        if not os.path.exists(path):
            import urllib.request
            print(f"downloading FlyWire cell-type annotations (31 MB) to {path} ...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(ANNOT_URL, path + ".part")
            os.replace(path + ".part", path)
        cols = ["root_id", "super_class", "cell_class", "cell_sub_class",
                "cell_type", "side", "top_nt", "pos_x", "pos_y", "pos_z",
                "soma_x", "soma_y", "soma_z"]
        a = pd.read_csv(path, sep="\t", usecols=cols, dtype={"root_id": "int64"},
                        low_memory=False)
        a = a.drop_duplicates("root_id").set_index("root_id")
        return a.reindex(self.root_ids)

    def neurons(self, side=None, **match):
        """Indices of neurons whose annotation columns equal the given values,
        e.g. neurons(cell_type="DNp01", side="left")."""
        mask = np.ones(self.n, dtype=bool)
        for col, val in match.items():
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            mask &= self.annot[col].isin(vals).to_numpy()
        if side is not None:
            mask &= (self.annot["side"] == side).to_numpy()
        return np.flatnonzero(mask)

    def positions(self):
        """(n, 3) FAFB coordinates in nm: the soma if known, else a point on
        the neuron. NaN for the few neurons without an annotation."""
        a = self.annot
        xyz = a[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
        missing = np.isnan(xyz[:, 0])
        xyz[missing] = a[["pos_x", "pos_y", "pos_z"]].to_numpy(float)[missing]
        return xyz * np.array([4.0, 4.0, 40.0])

    def set_silenced(self, idx):
        """Silence neurons (optogenetic inhibition in the paper): all their
        outgoing synapses are set to zero. Replaces the previous set."""
        if self._w_intact is None:
            self._w_intact = self.W.data.copy()
        else:
            self.W.data[:] = self._w_intact
        self.silenced = np.unique(np.asarray(idx, dtype=np.int64))
        ptr = self.W.indptr
        for i in self.silenced:
            self.W.data[ptr[i]:ptr[i + 1]] = 0.0

    def ids_to_idx(self, root_ids):
        return np.array([self.idx[r] for r in root_ids if r in self.idx], dtype=np.int64)

    def reset(self):
        n = self.n
        self.v = np.full(n, self.p["v_0"], dtype=np.float64)
        self.g = np.zeros(n, dtype=np.float64)
        self.ref_until = np.full(n, -1, dtype=np.int64)
        self.t = 0
        self._in_flight = deque([np.empty(0, np.int64)] * self._dly_steps)
        self.fired = np.empty(0, dtype=np.int64)
        self.spike_count = np.zeros(n, dtype=np.int64)
        self._fired_buf = np.empty(n, dtype=np.int64)
        self.set_input([])

    def set_input(self, drive):
        """Poisson drive: a list of (neuron indices, rate in Hz) pairs.
        It replaces the previous drive. Driven neurons have no refractory period."""
        idx, prob = [], []
        for neu, hz in drive:
            neu = np.asarray(neu, dtype=np.int64)
            idx.append(neu)
            prob.append(np.full(len(neu), hz * self.dt * 1e-3, dtype=np.float32))
        self._poi_idx = np.concatenate(idx) if idx else np.empty(0, np.int64)
        self._poi_p = np.concatenate(prob) if prob else np.empty(0, np.float32)
        self._driven = np.zeros(self.n, dtype=np.bool_)
        self._driven[self._poi_idx] = True

    def step(self):
        """Advance one dt. Order follows Brian2's schedule:
        integrate -> threshold -> synapses (+ Poisson input) -> reset."""
        p = self.p
        t = self.t
        k = _integrate(t, self.v, self.g, self.ref_until, p["v_0"], p["v_th"],
                       self._e_m, self._e_s, self._c_vg, self._fired_buf)
        fired = self._fired_buf[:k].copy()

        arriving = self._in_flight.popleft()
        self._in_flight.append(fired)
        if len(arriving):
            _deliver(t, arriving, self.W.indptr, self.W.indices, self.W.data,
                     self.g, self.ref_until)
        if len(self._poi_idx):
            hit = self._poi_idx[self.rng.random(len(self._poi_idx)) < self._poi_p]
            np.add.at(self.v, hit, self._w_poi)

        _reset(t, fired, self.v, self.g, self.ref_until, p["v_rst"],
               self._rfc_steps, self._driven)

        self.fired = fired
        self.spike_count[fired] += 1
        self.t += 1
        return fired

    def run(self, ms, record=False):
        """Run for `ms` milliseconds. Returns (step, neuron) spike pairs if record."""
        steps, neus = [], []
        for _ in range(int(round(ms / self.dt))):
            f = self.step()
            if record and len(f):
                steps.append(np.full(len(f), self.t, dtype=np.int64))
                neus.append(f)
        if record:
            if not steps:
                return np.empty((0, 2), dtype=np.int64)
            return np.column_stack([np.concatenate(steps), np.concatenate(neus)])
        return None

    def rates(self, since_count, ms):
        """Firing rate in Hz of every neuron since a spike_count snapshot."""
        return (self.spike_count - since_count) / (ms * 1e-3)


if __name__ == "__main__":
    import time

    t0 = time.time()
    brain = FlyBrain(seed=0)
    print(f"{brain.n:,} neurons, {brain.n_connections:,} connections, "
          f"loaded in {time.time() - t0:.1f}s")
    sugar = brain.neurons(cell_type="LB3", side="left")
    mn9 = brain.neurons(cell_type="CB0701")

    window = 100
    for t in range(0, 2000, window):
        if t == 0:
            brain.set_input([(sugar, 150)])
            print(f"\n0 ms: sugar on ({len(sugar)} sugar-sensing neurons at 150 Hz)")
        elif t == 1000:
            brain.set_input([])
            print("1000 ms: sugar off")
        before = brain.spike_count.copy()
        brain.run(window)
        r = brain.rates(before, window)
        bar = "#" * int(r[mn9].mean() / 5)
        print(f"{t:5d} ms  active {int((r > 0).sum()):4d} neurons   "
              f"MN9 {r[mn9].mean():5.0f} Hz {bar}")
