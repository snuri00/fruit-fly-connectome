"""
The fly's compound eyes: from a camera image to the rest of the brain.

The retina and the optic lobes up to the motion detectors are simulated with
flyvis (Lappalainen et al., Nature 2024): a connectome-constrained model of
721 columns per eye and 65 cell types whose neurons are graded (non-spiking)
and whose time constants, resting potentials and synapse strengths were
trained on optic flow. Its T4/T5 cells reproduce the direction tuning measured
in real flies, which the spiking whole-brain model with one strength for
every synapse does not.

flyvis's lobula and lobula plate cell types (T2, T3, T4a-d, T5a-d, Tm, TmY)
then drive the same cell types in the FlyWire whole-brain model, column by
column, with Poisson input. The column of every FlyWire neuron comes from the
Codex column assignment (Matsliah et al., Nature 2024). The two hexagonal
lattices are aligned from the connectome itself: for each T4 subtype the
offset from its Mi4 to its Mi9 inputs points along its preferred direction,
in both models; the rotation that makes these four directions agree is 180°.

The FlyWire neurons are driven by how far each flyvis cell is above its
adapted level, a running mean of its own activity. Tm, T2 and T3 respond
tonically to brightness and adapt with a 0.5 s time constant. T4/T5 adapt
more slowly (2 s), like motion adaptation in flies: they also sit above
their grey-screen level in a still, textured scene, which kept the giant
fiber firing, while a faster adaptation would erase motion too quickly.
A still scene fades out and only change reaches the brain.

Camera image: the left half is seen by the left eye, the right half by the
right eye, as if the fly looked at the screen; each eye gets the square next
to the midline, so shapes keep their proportions. Each half is resampled onto
the 721 ommatidia so that T4a/T5a respond to front-to-back motion and T4c/T5c
to upward motion, as in the fly.
"""
import os

import numpy as np
import pandas as pd
import torch

from flyvis import NetworkView, results_dir
from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = os.path.join(HERE, "data", "column_assignment.csv.gz")
MODEL = "flow/0000/000"
DT = 0.01
EYES = ("left", "right")
OUTPUT_PREFIXES = ("T2", "T3", "T4", "T5", "Tm")
GAIN = 400.0
RATE_MAX = 250.0
ADAPT_TAU = 0.5
ADAPT_TAU_MOTION = 2.0
WIDE_FIELD = ("Am", "CT1", "Lawf")
STAGES = ["photoreceptors", "lamina", "medulla", "motion (T4/T5)", "lobula"]


def stage_of(cell_type):
    """Processing stage of a flyvis cell type, or None for wide-field cells."""
    t = str(cell_type)
    if t.startswith(WIDE_FIELD):
        return None
    if t.startswith("R"):
        return "photoreceptors"
    if t.startswith("L") and t[1:].isdigit():
        return "lamina"
    if t.startswith(("T4", "T5")):
        return "motion (T4/T5)"
    if t.startswith(("T2", "T3", "TmY")):
        return "lobula"
    return "medulla"


def fw_pixel(p, q):
    return np.stack([(q - p) / 2.0, (p + q) * np.sqrt(3) / 2], -1)


class FlyEyes:
    def __init__(self, brain, model=MODEL, dt=DT, sample_px=96):
        self.brain = brain
        self.dt = dt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = NetworkView(results_dir / model).init_network()
        self.net.eval()
        nodes = self.net.connectome.nodes
        self.node_type = nodes.type[:].astype(str)
        self.node_uv = np.stack([nodes.u[:], nodes.v[:]], -1)
        self.u, self.v = get_hex_coords(15)
        self.hx, self.hy = hex_to_pixel(self.u, self.v)
        self.n_hex = len(self.u)
        self.types = sorted(set(self.node_type))

        self._label = self._hex_labels(sample_px)
        self.sample_px = sample_px
        with torch.no_grad():
            self.params = self.net._param_api()
            self.state = self.net.steady_state(t_pre=1.0, dt=dt, batch_size=2, value=0.5)
        self.baseline = self.state.nodes.activity.detach().cpu().numpy().copy()
        self.activity = self.baseline.copy()
        self.adapted = self.baseline.copy()
        self.photo = np.full((2, self.n_hex), 0.5, dtype=np.float32)
        self._type_hex = {}
        self._scale = {}
        self._map_flywire()
        self._map_visual()
        self.node_stage = np.array([stage_of(t) for t in self.node_type], dtype=object)

    def _hex_labels(self, n):
        """For an n x n patch, the ommatidium each pixel belongs to."""
        ext_x, ext_y = np.abs(self.hx).max() + 1.5, np.abs(self.hy).max() + 1.5
        ext = max(ext_x, ext_y)
        gx, gy = np.meshgrid(np.linspace(-ext, ext, n), np.linspace(ext, -ext, n))
        d = (gx[..., None] - self.hx) ** 2 + (gy[..., None] - self.hy) ** 2
        lab = d.argmin(-1)
        lab[d.min(-1) > 3.0] = -1
        return lab

    def _map_flywire(self):
        """flyvis node index for every FlyWire neuron of an output type."""
        col = pd.read_csv(COLUMNS)
        col = col[col.root_id.isin(self.brain.idx) & col.type.str.startswith(OUTPUT_PREFIXES)]
        col = col[col.type.isin(self.types)]
        fv = -np.sqrt(3) * fw_pixel(col.p.to_numpy(), col.q.to_numpy())
        d = (fv[:, None, 0] - self.hx) ** 2 + (fv[:, None, 1] - self.hy) ** 2
        hexal = d.argmin(1)
        inside = d.min(1) < 1.0
        key = {(t, int(u), int(v)): i for i, (t, (u, v)) in enumerate(zip(self.node_type, self.node_uv))}
        fw, eye, node = [], [], []
        for (rid, hemi, t), h, ok in zip(col[["root_id", "hemisphere", "type"]].itertuples(index=False),
                                         hexal, inside):
            k = key.get((t, int(self.u[h]), int(self.v[h])))
            if ok and k is not None:
                fw.append(self.brain.idx[rid])
                eye.append(EYES.index(hemi))
                node.append(k)
        self.fw_idx = np.array(fw, dtype=np.int64)
        self.fw_eye = np.array(eye)
        self.fw_node = np.array(node)
        self.fw_type = self.node_type[self.fw_node]

    def _map_visual(self):
        """flyvis node for every columnar FlyWire visual neuron, for display.
        Types in the Codex column table use its columns. Others, most of all
        R1-6 (which the table leaves out, as OpticLobe.jl notes), get the
        synapse-weighted mean position of their partners that have a column."""
        b = self.brain
        col = pd.read_csv(COLUMNS)
        col = col[col.root_id.isin(b.idx)]
        pix = np.full((b.n, 2), np.nan)
        known = np.array([b.idx[r] for r in col.root_id])
        pix[known] = fw_pixel(col.p.to_numpy(), col.q.to_numpy())
        hemi = np.full(b.n, "", dtype=object)
        hemi[known] = col.hemisphere.to_numpy()
        ftype = np.full(b.n, "", dtype=object)
        ftype[known] = col.type.to_numpy()

        ct = b.annot["cell_type"].fillna("").to_numpy().astype(str)
        side = b.annot["side"].fillna("").to_numpy()
        wanted = set(t for t in self.types if stage_of(t)) | {"R1-6"}
        missing = np.flatnonzero(np.isin(ct, list(wanted)) & np.isnan(pix[:, 0]))
        if len(missing):
            A = abs(b.W[missing]) + abs(b.W[:, missing]).T
            k = np.flatnonzero(~np.isnan(pix[:, 0]))
            Ak = A.tocsc()[:, k]
            wsum = np.asarray(Ak.sum(axis=1)).ravel()
            est = (Ak @ pix[k]) / np.maximum(wsum, 1e-9)[:, None]
            ok = wsum > 0
            pix[missing[ok]] = est[ok]
            hemi[missing[ok]] = side[missing[ok]]
            ftype[missing[ok]] = ct[missing[ok]]
        self.inferred = int(len(missing) and ok.sum())

        use = np.flatnonzero(~np.isnan(pix[:, 0]) & np.isin(hemi, EYES))
        fv = -np.sqrt(3) * pix[use]
        d = (fv[:, None, 0] - self.hx) ** 2 + (fv[:, None, 1] - self.hy) ** 2
        hexal, inside = d.argmin(1), d.min(1) < 1.5
        key = {(t, int(u), int(v)): i for i, (t, (u, v)) in enumerate(zip(self.node_type, self.node_uv))}
        vis, eye, node = [], [], []
        for i, h, ok_ in zip(use, hexal, inside):
            t = "R1" if ftype[i] == "R1-6" else ftype[i]
            k_ = key.get((t, int(self.u[h]), int(self.v[h])))
            if ok_ and k_ is not None:
                vis.append(i)
                eye.append(EYES.index(hemi[i]))
                node.append(k_)
        self.vis_idx = np.array(vis, dtype=np.int64)
        self.vis_eye = np.array(eye)
        self.vis_node = np.array(node)
        self.vis_stage = np.array([stage_of(t) for t in self.node_type[self.vis_node]], dtype=object)

    def visual_levels(self, gain=3.0):
        """0..1 display level of every mapped FlyWire visual neuron: how far its
        flyvis counterpart is from its adapted level (either sign)."""
        dev = self.activity[self.vis_eye, self.vis_node] - self.adapted[self.vis_eye, self.vis_node]
        return np.clip(np.abs(dev) * gain, 0.0, 1.0)

    def stage_activity(self):
        """Mean absolute deviation from the adapted level per stage (flyvis units)."""
        dev = np.abs(self.activity - self.adapted)
        return {st: float(dev[:, self.node_stage == st].mean()) for st in STAGES}

    def sample(self, gray):
        """gray: H x W float image in [0, 1]. Returns (2, 721) photoreceptor input."""
        import cv2
        h, w = gray.shape
        out = np.empty((2, self.n_hex), dtype=np.float32)
        side = min(w // 2, h)
        top = (h - side) // 2
        halves = {"left": gray[top:top + side, w // 2 - side: w // 2],
                  "right": gray[top:top + side, w // 2: w // 2 + side]}
        for e, name in enumerate(EYES):
            patch = cv2.resize(halves[name], (self.sample_px, self.sample_px), interpolation=cv2.INTER_AREA)
            if name == "right":
                patch = patch[:, ::-1]
            lab = self._label.ravel()
            ok = lab >= 0
            s = np.bincount(lab[ok], weights=patch.ravel()[ok], minlength=self.n_hex)
            c = np.bincount(lab[ok], minlength=self.n_hex)
            out[e] = s / np.maximum(c, 1)
        self.photo = out
        return out

    def step(self, photo=None, frames=1):
        """Advance both eyes by frames x dt with the given photoreceptor input."""
        photo = self.photo if photo is None else photo
        x = torch.tensor(photo, dtype=torch.float32, device=self.device)[:, None, None, :]
        x = x.expand(2, frames, 1, self.n_hex)
        with torch.no_grad():
            self.net.stimulus.zero(2, frames)
            self.net.stimulus.add_input(x)
            stim = self.net.stimulus()
            for i in range(frames):
                self.state = self.net._next_state(self.params, self.state, stim[:, i], self.dt)
        self.activity = self.state.nodes.activity.detach().cpu().numpy()
        self.adapted += self._adapt_k(frames) * (self.activity - self.adapted)
        return self.activity

    def _adapt_k(self, frames):
        if getattr(self, "_k_cache", (None,))[0] != frames:
            motion = np.char.startswith(self.node_type.astype(str), "T4") | \
                np.char.startswith(self.node_type.astype(str), "T5")
            tau = np.where(motion, ADAPT_TAU_MOTION, ADAPT_TAU)
            self._k_cache = (frames, 1 - np.exp(-frames * self.dt / tau))
        return self._k_cache[1]

    def reset(self):
        with torch.no_grad():
            self.state = self.net.steady_state(t_pre=1.0, dt=self.dt, batch_size=2, value=0.5)
        self.activity = self.baseline.copy()
        self.adapted = self.baseline.copy()

    def rates(self):
        """Poisson rate for every mapped FlyWire neuron (Hz)."""
        act = self.activity[self.fw_eye, self.fw_node] - self.adapted[self.fw_eye, self.fw_node]
        return np.clip(GAIN * act, 0.0, RATE_MAX)

    def drive(self):
        return [(self.fw_idx, self.rates())]

    def _hex_index(self, t):
        """flyvis node of cell type t in each ommatidium (-1 where missing)."""
        if t not in self._type_hex:
            pos = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(self.u, self.v))}
            out = np.full(self.n_hex, -1)
            for n in np.flatnonzero(self.node_type == t):
                out[pos[tuple(int(c) for c in self.node_uv[n])]] = n
            self._type_hex[t] = out
        return self._type_hex[t]

    def type_map(self, eye, t):
        idx = self._hex_index(t)
        a = self.activity[eye, idx] - self.adapted[eye, idx]
        return np.where(idx >= 0, a, 0.0)

    def motion_map(self, eye):
        """Per ommatidium: (rightward-in-image, upward) motion from T4/T5 a-d.
        Each subtype is divided by a running mean of its response size, so
        that subtypes with larger responses do not dominate the direction."""
        m = {}
        for t in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"):
            a = np.maximum(self.type_map(eye, t), 0.0)
            size = self._scale.get(t, 0.0)
            size += 0.02 * (a.mean() - size)
            self._scale[t] = size
            m[t] = a / max(size, 1e-3)
        front_to_back = m["T4a"] + m["T5a"] - m["T4b"] - m["T5b"]
        up = m["T4c"] + m["T5c"] - m["T4d"] - m["T5d"]
        rightward = -front_to_back if EYES[eye] == "left" else front_to_back
        return rightward, up
