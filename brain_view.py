"""
Live picture of the whole brain: every neuron at its soma position (frontal
view of the FlyWire volume), lit up in its cell-class colour when it spikes and
fading afterwards, like a calcium-imaging movie. Used by the console and the game.
"""
import numpy as np
import pygame

CLASS_COLORS = {
    "optic": (70, 120, 255),
    "visual_projection": (80, 205, 255),
    "visual_centrifugal": (120, 150, 255),
    "central": (255, 184, 80),
    "sensory": (110, 255, 170),
    "sensory_ascending": (140, 240, 200),
    "ascending": (200, 150, 255),
    "descending": (255, 96, 200),
    "motor": (255, 80, 80),
    "endocrine": (255, 250, 140),
}
UNKNOWN_CLASS = (170, 170, 170)
LEGEND = ["sensory", "optic", "visual_projection", "central", "descending", "motor"]


class Glow:
    """Per-neuron light level: 1 right after a spike, decaying with tau (ms)."""

    def __init__(self, n, tau=150.0):
        self.level = np.zeros(n, dtype=np.float32)
        self.tau = tau

    def update(self, fired, ms):
        self.level *= np.exp(-ms / self.tau)
        self.level[fired] = 1.0

    def clear(self):
        self.level[:] = 0.0


class BrainMap:
    def __init__(self, brain, max_size, background=(16, 19, 25),
                 density_color=(58, 70, 92)):
        xyz = brain.positions()
        self.ok = ~np.isnan(xyz[:, 0])
        x, y = xyz[:, 0], xyz[:, 1]
        x0, x1 = np.nanmin(x), np.nanmax(x)
        y0, y1 = np.nanmin(y), np.nanmax(y)
        scale = min(max_size[0] / (x1 - x0), max_size[1] / (y1 - y0))
        self.w = int((x1 - x0) * scale) + 1
        self.h = int((y1 - y0) * scale) + 1
        self.um_per_px = 1e-3 / scale
        self.px = np.clip(np.nan_to_num((x - x0) * scale, nan=0), 0, self.w - 1).astype(np.int64)
        self.py = np.clip(np.nan_to_num((y - y0) * scale, nan=0), 0, self.h - 1).astype(np.int64)

        cls = brain.annot["super_class"].fillna("?").to_numpy()
        self.color = np.array([CLASS_COLORS.get(c, UNKNOWN_CLASS) for c in cls],
                              dtype=np.float32)

        flat = self.px[self.ok] * self.h + self.py[self.ok]
        dens = np.bincount(flat, minlength=self.w * self.h)
        dens = np.log1p(dens.reshape(self.w, self.h).astype(np.float32))
        dens /= dens.max()
        lo, hi = np.array(background, np.float32), np.array(density_color, np.float32)
        self.bg = lo + dens[..., None] ** 0.8 * (hi - lo)
        self.surface = pygame.Surface((self.w, self.h))

        left = (brain.annot["side"] == "left").to_numpy() & self.ok
        self.left_on_screen_left = self.px[left].mean() < self.w / 2

    def render(self, glow, spot=True, gain=1.0):
        """Draw the neurons with glow > 0.02 onto the density image. Lower
        `gain` for small maps, where many neurons share a pixel."""
        img = self.bg.copy()
        act = np.flatnonzero((glow > 0.02) & self.ok)
        if len(act):
            light = self.color[act] * (gain * glow[act] ** 1.3)[:, None]
            px, py = self.px[act], self.py[act]
            offsets = [(0, 0, 1.0)]
            if spot:
                offsets += [(1, 0, 0.5), (-1, 0, 0.5), (0, 1, 0.5), (0, -1, 0.5)]
            for dx, dy, w in offsets:
                np.add.at(img, (np.clip(px + dx, 0, self.w - 1),
                                np.clip(py + dy, 0, self.h - 1)), light * w)
        np.clip(img, 0, 255, out=img)
        pygame.surfarray.blit_array(self.surface, img.astype(np.uint8))
        return self.surface

    def point(self, i, origin):
        return (origin[0] + int(self.px[i]), origin[1] + int(self.py[i]))

    def nearest(self, pos, origin, candidates, radius=12):
        """Index of the candidate neuron nearest to a screen position, or None."""
        cand = candidates[self.ok[candidates]]
        if not len(cand):
            return None
        d = (self.px[cand] + origin[0] - pos[0]) ** 2 + (self.py[cand] + origin[1] - pos[1]) ** 2
        j = int(np.argmin(d))
        return int(cand[j]) if d[j] < radius ** 2 else None
