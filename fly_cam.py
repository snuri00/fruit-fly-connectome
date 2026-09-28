"""
FLY EYE CAM - see yourself through a fruit fly's eyes, and watch its brain react.

Your camera image (mirrored, like a mirror) is split in two: the left half is
the fly's left eye, the right half its right eye. Each is resampled onto 721
ommatidia and run through flyvis (retina to motion detectors, trained on
optic flow); its lobula outputs then drive the FlyWire whole-brain model.

Panels: the camera; what each eye sees (hexagonal mosaics); the motion that
its T4/T5 cells report (colour = direction, brightness = strength); the live
brain; looming detectors, rotation-sensing HS/H2 cells, giant fibers and the
steering neurons.

    python3 fly_cam.py                 camera 0
    python3 fly_cam.py --camera 2      another camera
    python3 fly_cam.py --video clip.mp4
    python3 fly_cam.py --demo loom     synthetic looming disc (no camera needed)

Keys: SPACE pause   F11 / F full screen   ESC quit
The window can also be resized; the picture scales and keeps its proportions.
"""
import argparse
import math
import os
import threading
import time
from collections import deque

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

CANVAS = (1600, 960)
CAM = (20, 72, 560, 420)
DETECTOR = (20, 504, 560, 432)
EYES_RECT = (600, 72, 500, 420)
MOTION_RECT = (600, 504, 500, 432)
SIDE_X = 1120
BRAIN_RECT = (SIDE_X, 72, 460, 330)
FLOW_RECT = (SIDE_X, 414, 460, 522)

BG = (11, 12, 15)
PANEL = (20, 21, 26)
BORDER = (44, 46, 54)
TEXT = (230, 231, 235)
MUTED = (140, 142, 152)
FAINT = (80, 82, 92)
OFF = (38, 40, 48)
LOOM = (255, 110, 90)
ROT = (110, 190, 255)
GF = (255, 80, 80)
STEER = (120, 220, 150)
STAGE_COLORS = {
    "photoreceptors": (255, 226, 110),
    "lamina": (255, 150, 70),
    "medulla": (120, 225, 130),
    "motion (T4/T5)": (80, 160, 255),
    "lobula": (180, 130, 255),
}
POP_COLORS = {"visual projection": (80, 215, 235), "central brain": (255, 184, 80),
              "descending": (255, 96, 200)}
TRACE_S = 6.0
BRAIN_STEP_MS = 10.0
FRAME_BUDGET = 0.03


def wheel(angle, k=1.0):
    """Colour of a direction on the colour wheel (0 = right, pi/2 = up)."""
    h = (angle / (2 * math.pi)) % 1.0
    return tuple(int(12 + 243 * k * (0.5 + 0.5 * math.cos(2 * math.pi * (h - o)))) for o in (0.0, 1 / 3, 2 / 3))


def mix(c1, c2, k):
    k = min(max(k, 0.0), 1.0)
    return tuple(int(a + (b - a) * k) for a, b in zip(c1, c2))


class CameraSource:
    """Grabs frames on a background thread so the UI never waits for them."""

    def __init__(self, index=0, path=None):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(path if path else index)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {'video ' + path if path else f'camera {index}'}")
        self.is_file = path is not None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                if self.is_file:
                    self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue
            f = self.cv2.cvtColor(f, self.cv2.COLOR_BGR2RGB)[:, ::-1]
            with self.lock:
                self.frame = f
            if self.is_file:
                time.sleep(1.0 / self.fps)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        self.running = False
        self.cap.release()


class DemoSource:
    """Synthetic scenes: a dark disc looming towards one eye, or a rotating world."""

    def __init__(self, kind="loom"):
        self.kind = kind
        self.t0 = time.time()
        self.yy, self.xx = np.mgrid[0:480, 0:640]

    def read(self):
        t = (time.time() - self.t0) % 4.0
        img = np.full((480, 640), 0.8, np.float32)
        if self.kind == "loom":
            if 1.0 < t < 2.2:
                r = 6 + 240 * ((t - 1.0) / 1.2) ** 3
                img[(self.xx - 470) ** 2 + (self.yy - 240) ** 2 < r * r] = 0.05
        else:
            d = 1 if t < 2 else -1
            img = 0.5 + 0.45 * np.sign(np.sin(2 * np.pi * (self.xx - d * 300 * t) / 80))
        rgb = (np.repeat(img[..., None], 3, -1) * 255).astype(np.uint8)
        return rgb

    def close(self):
        pass


class App:
    def __init__(self, source):
        """Opens the window at once and loads the models on a background thread,
        so the window keeps answering the desktop while they load (~15 s)."""
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("Fly Eye Cam")
        self.window = pygame.display.set_mode(CANVAS, pygame.RESIZABLE)
        self.screen = pygame.Surface(CANVAS)
        self.fullscreen = False
        self.clock = pygame.time.Clock()
        f = pygame.font.SysFont
        self.f_title = f("dejavusans", 20, bold=True)
        self.f = f("dejavusans", 13)
        self.f_small = f("dejavusans", 11)
        self.f_mono = f("dejavusansmono", 13)
        self.f_big = f("dejavusans", 46, bold=True)
        self.source = source
        self.loading = "starting"
        self.load_error = None
        self.load_t0 = time.time()
        self.running = True
        loader = threading.Thread(target=self._load_models, daemon=True)
        loader.start()
        while loader.is_alive() and self.running:
            self._pump_while_loading()
        if not self.running:
            source.close()
            raise SystemExit(0)
        if self.load_error:
            raise self.load_error
        self._setup_views()

    def _pump_while_loading(self):
        pg = self.pg
        for ev in pg.event.get():
            if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                self.running = False
        self.screen.fill(BG)
        s = self.f_title.render(f"loading {self.loading} ...", True, MUTED)
        self.screen.blit(s, s.get_rect(center=(CANVAS[0] // 2, CANVAS[1] // 2)))
        t = self.f_small.render(f"{time.time() - self.load_t0:.0f} s", True, FAINT)
        self.screen.blit(t, t.get_rect(center=(CANVAS[0] // 2, CANVAS[1] // 2 + 30)))
        self.present()
        self.clock.tick(20)

    def _load_models(self):
        try:
            from fly_brain import FlyBrain
            from fly_circuits import Circuits
            from fly_eye import FlyEyes
            self.loading = "the FlyWire brain (138,639 neurons)"
            self.brain = FlyBrain(seed=int(time.time()))
            self.brain.run(0.2)
            self.loading = "the compound eyes (flyvis) and their FlyWire columns"
            self.eyes = FlyEyes(self.brain)
            self.circ = Circuits(self.brain)
        except BaseException as err:
            self.load_error = err

    def _setup_views(self):
        """Runs on the main thread. flyvis sets torch's default device when it is
        imported, and that setting is per thread, so it is repeated here."""
        import flyvis
        import torch
        from brain_view import BrainMap, Glow
        torch.set_default_device(flyvis.device)
        b = self.brain
        self.groups = {
            "LC4": {s: b.neurons(cell_type="LC4", side=s) for s in ("left", "right")},
            "LPLC2": {s: b.neurons(cell_type="LPLC2", side=s) for s in ("left", "right")},
            "HS": {s: b.neurons(cell_type=["HSE", "HSN", "HSS"], side=s) for s in ("left", "right")},
            "H2": {s: b.neurons(cell_type="H2", side=s) for s in ("left", "right")},
            "GF": {s: b.neurons(cell_type="DNp01", side=s) for s in ("left", "right")},
            "DNa": {s: b.neurons(cell_type=["DNa01", "DNa02"], side=s) for s in ("left", "right")},
        }
        self.rate = {g: {"left": 0.0, "right": 0.0} for g in self.groups}
        self.map = BrainMap(b, (440, 214), background=PANEL)
        for st, col in STAGE_COLORS.items():
            self.map.color[self.eyes.vis_idx[self.eyes.vis_stage == st]] = col
        self.glow = Glow(b.n, 200.0)
        sc = b.annot["super_class"].fillna("").to_numpy()
        self.pops = {"visual projection": np.flatnonzero(sc == "visual_projection"),
                     "central brain": np.flatnonzero(sc == "central"),
                     "descending": np.flatnonzero(sc == "descending")}
        self.stage_n = {st: int((self.eyes.vis_stage == st).sum()) for st in STAGE_COLORS}
        self.flow = deque()
        self.regions = self._region_labels()
        self.hex_px = self._hex_layout()
        self.motion = np.zeros((2, 2, self.eyes.n_hex))
        self.trace = deque()
        self.paused = False
        self.speed = 0.0
        self.last_frame = None

    def _region_labels(self):
        """Map coordinates of a few landmarks, from the neurons that lie there."""
        b, m = self.brain, self.map
        a = b.annot
        ct = a["cell_type"].fillna("").to_numpy()
        cc = a["cell_class"].fillna("").to_numpy()
        side = a["side"].fillna("").to_numpy()
        near = "left" if m.left_on_screen_left else "right"
        groups = [("lamina", (ct == "L1") & (side == near)),
                  ("medulla", (ct == "Mi1") & (side == near)),
                  ("lobula plate", (ct == "T4a") & (side == near)),
                  ("lobula", (ct == "LC4") & (side == near)),
                  ("mushroom body", (cc == "Kenyon_Cell") & (side != near)),
                  ("antennal lobe", (cc == "ALLN") & (side != near)),
                  ("central complex", cc == "CX"),
                  ("SEZ", cc == "gustatory")]
        out = []
        for name, mask in groups:
            idx = np.flatnonzero(mask & m.ok)
            if len(idx):
                out.append((name, (int(np.median(m.px[idx])), int(np.median(m.py[idx])))))
        return out

    def _hex_layout(self):
        """Screen offsets of the ommatidia: flyvis pixel coordinates are a flat-top
        hexagonal grid of hexagon radius 1, with y pointing up."""
        size = 5.4
        return self.eyes.hx * size, -self.eyes.hy * size, size

    def present(self):
        """Scale the fixed-size canvas into the window, keeping its proportions."""
        pg = self.pg
        ww, wh = self.window.get_size()
        k = min(ww / CANVAS[0], wh / CANVAS[1])
        size = (int(CANVAS[0] * k), int(CANVAS[1] * k))
        self.window.fill((0, 0, 0))
        img = self.screen if size == CANVAS else pg.transform.smoothscale(self.screen, size)
        self.window.blit(img, ((ww - size[0]) // 2, (wh - size[1]) // 2))
        pg.display.flip()

    def toggle_fullscreen(self):
        pg = self.pg
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.window = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        else:
            self.window = pg.display.set_mode(CANVAS, pg.RESIZABLE)

    def run(self):
        pg = self.pg
        running = True
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE:
                    if self.fullscreen:
                        self.toggle_fullscreen()
                    else:
                        running = False
                elif ev.type == pg.KEYDOWN and ev.key in (pg.K_F11, pg.K_f):
                    self.toggle_fullscreen()
                elif ev.type == pg.KEYDOWN and ev.key == pg.K_SPACE:
                    self.paused = not self.paused
            if not self.paused:
                self.advance()
            self.draw()
            self.clock.tick(60)
        self.source.close()
        pg.quit()

    def advance(self):
        frame = self.source.read()
        if frame is None:
            return
        self.last_frame = frame
        gray = frame.mean(-1).astype(np.float32) / 255.0
        self.eyes.sample(gray)
        t0 = time.perf_counter()
        fired, steps = [], 0
        before = self.brain.spike_count.copy()
        while time.perf_counter() - t0 < FRAME_BUDGET and steps < 4:
            self.eyes.step()
            self.brain.set_input(self.eyes.drive())
            sp = self.brain.run(BRAIN_STEP_MS, record=True)
            if len(sp):
                fired.append(sp[:, 1])
            steps += 1
        ms = steps * BRAIN_STEP_MS
        wall = time.perf_counter() - t0
        self.speed = 0.8 * self.speed + 0.2 * (ms / 1000.0) / max(wall, 1e-3)
        delta = self.brain.spike_count - before
        self.glow.update(np.concatenate(fired) if fired else np.empty(0, np.int64), ms)
        km = 1 - math.exp(-ms / 150.0)
        for e in range(2):
            self.motion[e] += km * (np.array(self.eyes.motion_map(e)) - self.motion[e])
        k = 1 - math.exp(-ms / 80.0)
        for g, sides in self.groups.items():
            for s, idx in sides.items():
                hz = delta[idx].mean() / (ms * 1e-3) if len(idx) else 0.0
                self.rate[g][s] += k * (hz - self.rate[g][s])
        self.glow.level[self.eyes.vis_idx] = np.maximum(self.glow.level[self.eyes.vis_idx],
                                                        self.eyes.visual_levels())
        now = time.time()
        sa = self.eyes.stage_activity()
        pops = [delta[idx].sum() / len(idx) / (ms * 1e-3) for idx in self.pops.values()]
        self.flow.append((now, [sa[st] for st in STAGE_COLORS] + pops))
        while self.flow and now - self.flow[0][0] > TRACE_S:
            self.flow.popleft()
        gf_spikes = int(delta[np.concatenate(list(self.groups["GF"].values()))].sum())
        self.trace.append((now, self.motion[0][0].mean(), self.motion[1][0].mean(), self.rotation(),
                           self.rate["LC4"]["left"] + self.rate["LC4"]["right"], gf_spikes))
        while self.trace and now - self.trace[0][0] > TRACE_S:
            self.trace.popleft()

    def rotation(self):
        """World rotation reported by HS/H2 (positive: the world moves right)."""
        r = self.rate
        return (r["HS"]["right"] + r["H2"]["left"]) - (r["HS"]["left"] + r["H2"]["right"])

    def text(self, txt, pos, color=TEXT, f=None, anchor="topleft"):
        s = (f or self.f).render(str(txt), True, color)
        r = s.get_rect(**{anchor: pos})
        self.screen.blit(s, r)
        return r

    def panel(self, rect, title, subtitle=""):
        pg = self.pg
        rect = pg.Rect(rect)
        pg.draw.rect(self.screen, PANEL, rect, border_radius=6)
        pg.draw.rect(self.screen, BORDER, rect, 1, border_radius=6)
        r = self.text(title, (rect.x + 12, rect.y + 9), TEXT, self.f_mono)
        if subtitle:
            self.text(subtitle, (r.right + 8, rect.y + 11), MUTED, self.f_small)
        return rect

    def draw(self):
        self.screen.fill(BG)
        self.text("FLY EYE CAM", (20, 16), TEXT, self.f_title)
        self.text("your camera through a fruit fly's compound eyes  ·  flyvis retina & optic lobe  →  "
                  "FlyWire whole brain", (160, 22), MUTED, self.f_small)
        self.text(f"brain speed {self.speed:.2f}x real",
                  (CANVAS[0] - 20, 22), MUTED, self.f_small, "topright")
        self.draw_camera()
        self.draw_detector()
        self.draw_mosaics()
        self.draw_motion()
        self.draw_side()
        self.text("SPACE pause   ·   F11 full screen   ·   ESC quit   ·   "
                  "stay still and the fly's brain goes quiet; move and watch it react",
                  (20, CANVAS[1] - 20), FAINT, self.f_small)
        self.present()

    def draw_camera(self):
        pg = self.pg
        r = self.panel(CAM, "CAMERA", "mirrored · left half = left eye, right half = right eye")
        if self.last_frame is None:
            self.text("waiting for the camera ...", (r.x + 14, r.y + 40), FAINT, self.f)
            return
        view = pg.Rect(r.x + 10, r.y + 32, r.w - 20, r.h - 42)
        surf = pg.surfarray.make_surface(np.transpose(self.last_frame, (1, 0, 2)))
        self.screen.blit(pg.transform.smoothscale(surf, view.size), view)
        pg.draw.line(self.screen, (255, 255, 255), (view.centerx, view.y), (view.centerx, view.bottom), 1)
        self.text("LEFT EYE", (view.x + 8, view.y + 6), (255, 255, 255), self.f_small)
        self.text("RIGHT EYE", (view.right - 8, view.y + 6), (255, 255, 255), self.f_small, "topright")

    def draw_detector(self):
        """Population motion of each eye as a compass, the world rotation that
        HS/H2 report, looming and giant fibers, and a 6 s recording of them."""
        pg = self.pg
        r = self.panel(DETECTOR, "MOTION DETECTOR", "what the eyes and brain report")
        for e, name in enumerate(("left", "right")):
            cx, cy, rad = r.x + 90 + e * 170, r.y + 112, 58
            pg.draw.circle(self.screen, OFF, (cx, cy), rad, 2)
            for a in range(0, 360, 45):
                x = cx + (rad - 6) * math.cos(math.radians(a))
                y = cy - (rad - 6) * math.sin(math.radians(a))
                pg.draw.circle(self.screen, FAINT, (x, y), 2)
            right, up = self.motion[e].mean(axis=1)
            k = min(math.hypot(right, up) / 1.5, 1.0)
            if k > 0.05:
                ang = math.atan2(up, right)
                tip = (cx + rad * 0.9 * k * math.cos(ang), cy - rad * 0.9 * k * math.sin(ang))
                col = mix(FAINT, ROT, k)
                pg.draw.line(self.screen, col, (cx, cy), tip, 5)
                pg.draw.circle(self.screen, col, tip, 6)
                arrow = "→↗↑↖←↙↓↘"[int(((math.degrees(ang) + 22.5) % 360) // 45)]
                label = f"{arrow}  {100 * k:3.0f}%"
            else:
                label = "still"
            pg.draw.circle(self.screen, TEXT, (cx, cy), 3)
            self.text(f"{name} eye", (cx, cy - rad - 20), MUTED, self.f_small, "midtop")
            self.text(label, (cx, cy + rad + 6), TEXT, self.f_mono, "midtop")

        x0 = r.x + 360
        wbar = r.right - x0 - 16
        self.text("world rotation  (HS / H2)", (x0, r.y + 40), MUTED, self.f_small)
        bar = pg.Rect(x0, r.y + 60, wbar, 12)
        pg.draw.rect(self.screen, OFF, bar, border_radius=4)
        frac = max(-1.0, min(1.0, self.rotation() / 300.0))
        if abs(frac) > 0.02:
            w = abs(frac) * bar.w / 2
            pg.draw.rect(self.screen, ROT, (bar.centerx if frac > 0 else bar.centerx - w, bar.y, w, bar.h),
                         border_radius=4)
        pg.draw.line(self.screen, TEXT, (bar.centerx, bar.y - 4), (bar.centerx, bar.bottom + 4), 1)
        self.text("←", (bar.x, bar.bottom + 2), MUTED, self.f)
        self.text("→", (bar.right, bar.bottom + 2), MUTED, self.f, "topright")

        self.text("looming  (LC4)", (x0, r.y + 100), MUTED, self.f_small)
        for k2, s2 in enumerate(("left", "right")):
            v = self.rate["LC4"][s2]
            b2 = pg.Rect(x0 + k2 * ((wbar + 8) // 2), r.y + 118, (wbar - 8) // 2, 10)
            pg.draw.rect(self.screen, OFF, b2, border_radius=3)
            if v > 0.5:
                pg.draw.rect(self.screen, LOOM, (b2.x, b2.y, b2.w * min(v / 100, 1), b2.h), border_radius=3)
            self.text(f"{s2[0].upper()} {v:3.0f} Hz", (b2.x, b2.bottom + 2), FAINT, self.f_small)

        self.text("giant fibers  (DNp01)", (x0, r.y + 156), MUTED, self.f_small)
        for k2, s2 in enumerate(("left", "right")):
            v = self.rate["GF"][s2]
            c = (x0 + 12 + k2 * 90, r.y + 186)
            pg.draw.circle(self.screen, mix(OFF, GF, min(v / 60, 1)), c, 10)
            self.text(f"{s2[0].upper()} {v:3.0f} Hz", (c[0] + 16, c[1] - 7), FAINT, self.f_small)

        self.draw_traces(pg.Rect(r.x + 12, r.y + 232, r.w - 24, r.h - 244))

    def draw_traces(self, area):
        """Scrolling recording of the last TRACE_S seconds."""
        pg = self.pg
        self.text(f"RECORDING  last {TRACE_S:.0f} s", (area.x, area.y), MUTED, self.f_small)
        rows = [("eye motion  → / ←", ROT), ("world rotation", ROT), ("looming LC4", LOOM),
                ("giant fiber spikes", GF)]
        plot = pg.Rect(area.x + 118, area.y + 18, area.w - 118, area.h - 30)
        h = plot.h / len(rows)
        now = time.time()
        tr = list(self.trace)

        def X(t):
            return plot.right - (now - t) / TRACE_S * plot.w

        for i, (label, col) in enumerate(rows):
            band = pg.Rect(plot.x, plot.y + i * h, plot.w, h - 4)
            pg.draw.rect(self.screen, (16, 17, 21), band)
            self.text(label, (area.x, band.y + band.h / 2 - 7), FAINT, self.f_small)
            if len(tr) < 2:
                continue
            mid = band.centery
            if i == 0:
                pg.draw.line(self.screen, OFF, (band.x, mid), (band.right, mid))
                for j, c in ((1, (120, 170, 255)), (2, (170, 230, 255))):
                    pts = [(X(t[0]), mid - max(-1, min(1, t[j] / 1.5)) * band.h / 2) for t in tr]
                    pg.draw.lines(self.screen, c, False, pts, 2)
            elif i == 1:
                pg.draw.line(self.screen, OFF, (band.x, mid), (band.right, mid))
                pts = [(X(t[0]), mid - max(-1, min(1, t[3] / 300)) * band.h / 2) for t in tr]
                pg.draw.lines(self.screen, col, False, pts, 2)
            elif i == 2:
                pts = [(X(t[0]), band.bottom - min(t[4] / 150, 1) * band.h) for t in tr]
                pg.draw.lines(self.screen, col, False, pts, 2)
            else:
                for t in tr:
                    if t[5]:
                        x = X(t[0])
                        pg.draw.line(self.screen, col, (x, band.y + 2), (x, band.bottom - 2), 2)
        for sec in range(int(TRACE_S) + 1):
            x = plot.right - sec / TRACE_S * plot.w
            self.text(f"-{sec}s" if sec else "now", (x, plot.bottom + 1), FAINT, self.f_small, "midtop")
        self.text("left eye", (plot.right - 110, area.y), (120, 170, 255), self.f_small)
        self.text("right eye", (plot.right - 52, area.y), (170, 230, 255), self.f_small)

    def _hexes(self, center, values_rgb, flip_x=False):
        pg = self.pg
        xs, ys, size = self.hex_px
        if flip_x:
            xs = -xs
        corners = [(size * math.cos(math.pi / 3 * k), size * math.sin(math.pi / 3 * k)) for k in range(6)]
        for x, y, col in zip(xs, ys, values_rgb):
            cx, cy = center[0] + x, center[1] + y
            pg.draw.polygon(self.screen, col, [(cx + a, cy + b) for a, b in corners])

    def draw_mosaics(self):
        r = self.panel(EYES_RECT, "WHAT THE FLY SEES", "721 ommatidia per eye")
        cy = r.y + 32 + (r.h - 32) // 2
        for e, name in enumerate(("left", "right")):
            cx = r.x + r.w // 4 + e * r.w // 2
            vals = np.clip(self.eyes.photo[e], 0, 1)
            cols = [(int(v * 235) + 10,) * 3 for v in vals]
            self._hexes((cx, cy), cols, flip_x=(name == "right"))
            self.text(f"{name} eye", (cx, r.bottom - 20), MUTED, self.f_small, "midtop")

    def draw_motion(self):
        r = self.panel(MOTION_RECT, "MOTION DETECTORS  T4 / T5",
                       "colour = direction, brightness = strength (150 ms average)")
        cy = r.y + 32 + (r.h - 60) // 2
        for e, name in enumerate(("left", "right")):
            cx = r.x + r.w // 4 + e * r.w // 2
            right, up = self.motion[e]
            mag = np.clip(np.hypot(right, up) / 4.0, 0, 1)
            ang = np.arctan2(up, right)
            self._hexes((cx, cy), [wheel(a, m) for a, m in zip(ang, mag)], flip_x=(name == "right"))
            self.text(f"{name} eye", (cx, r.bottom - 44), MUTED, self.f_small, "midtop")
        lx = r.x + 14
        for label, a in (("→", 0.0), ("↑", math.pi / 2), ("←", math.pi), ("↓", -math.pi / 2)):
            self.pg.draw.circle(self.screen, wheel(a), (lx + 6, r.bottom - 14), 6)
            self.text(label, (lx + 16, r.bottom - 22), MUTED, self.f)
            lx += 50

    def draw_side(self):
        self.draw_brain()
        self.draw_flow()

    def draw_brain(self):
        pg = self.pg
        side = "fly's left on the left" if self.map.left_on_screen_left else "fly's left on the right"
        r = self.panel(BRAIN_RECT, "BRAIN", f"all 138,639 neurons · frontal view, dorsal up, {side}")
        m = self.map
        ox, oy = r.centerx - m.w // 2, r.y + 62
        self.screen.blit(m.render(self.glow.level, spot=False, gain=0.45), (ox, oy))
        top_row = ("lamina", "lobula plate", "central complex", "mushroom body")
        rows = {"top": [], "bottom": []}
        for name, (x, y) in self.regions:
            rows["top" if name in top_row else "bottom"].append((ox + x, name, (ox + x, oy + y)))
        for row, ty in (("top", oy - 6), ("bottom", oy + m.h + 6)):
            placed = sorted(rows[row])
            xs = []
            for x, name, p in placed:
                w = self.f_small.size(name)[0]
                x = max(x, (xs[-1] if xs else r.x + 12) + w / 2 + 8)
                xs.append(x + w / 2)
                q = (x, ty)
                pg.draw.circle(self.screen, (225, 225, 230), p, 3, 1)
                pg.draw.line(self.screen, FAINT, p, q, 1)
                self.text(name, q, (205, 207, 215), self.f_small, "midbottom" if row == "top" else "midtop")
        bar = 100 / m.um_per_px
        by = r.bottom - 40
        pg.draw.line(self.screen, MUTED, (r.right - 14 - bar, by), (r.right - 14, by), 2)
        self.text("100 µm", (r.right - 20 - bar, by - 7), MUTED, self.f_small, "topright")

        x = r.x + 12
        y = r.bottom - 22
        for name, col in list(STAGE_COLORS.items()) + list(POP_COLORS.items()):
            short = {"photoreceptors": "photorec.", "motion (T4/T5)": "T4/T5", "visual projection": "VPN",
                     "central brain": "central", "descending": "descend."}.get(name, name)
            pg.draw.rect(self.screen, col, (x, y + 4, 8, 8), border_radius=2)
            t = self.text(short, (x + 11, y), MUTED, self.f_small)
            x = t.right + 9

    def draw_flow(self):
        """Activity of each processing stage over the last seconds: a movement in
        front of the camera travels down the list."""
        pg = self.pg
        r = self.panel(FLOW_RECT, "SIGNAL FLOW", f"eye → brain, last {TRACE_S:.0f} s")
        names = list(STAGE_COLORS) + list(POP_COLORS)
        colors = list(STAGE_COLORS.values()) + list(POP_COLORS.values())
        counts = [self.stage_n[st] for st in STAGE_COLORS] + [len(v) for v in self.pops.values()]
        units = ["a.u."] * len(STAGE_COLORS) + ["Hz"] * len(POP_COLORS)
        top, bottom = r.y + 36, r.bottom - 44
        h = (bottom - top) / len(names)
        plot_x0, plot_x1 = r.x + 150, r.right - 64
        now = time.time()
        flow = list(self.flow)
        for i, (name, col, n, unit) in enumerate(zip(names, colors, counts, units)):
            y0 = top + i * h
            band = pg.Rect(plot_x0, y0 + 4, plot_x1 - plot_x0, h - 8)
            pg.draw.rect(self.screen, (16, 17, 21), band)
            pg.draw.rect(self.screen, col, (r.x + 12, y0 + 8, 4, h - 16), border_radius=2)
            self.text(name, (r.x + 22, y0 + 6), TEXT, self.f_small)
            src = "flyvis" if i < len(STAGE_COLORS) else "FlyWire"
            self.text(f"{n:,} n · {src}", (r.x + 22, y0 + 21), FAINT, self.f_small)
            if len(flow) < 2:
                continue
            vals = np.array([f[1][i] for f in flow])
            top_v = max(vals.max(), 0.02 if unit == "a.u." else 1.0)
            pts = [(plot_x1 - (now - f[0]) / TRACE_S * band.w, band.bottom - v / top_v * (band.h - 2))
                   for f, v in zip(flow, vals)]
            poly = [(pts[0][0], band.bottom)] + pts + [(pts[-1][0], band.bottom)]
            pg.draw.polygon(self.screen, mix(PANEL, col, 0.35), poly)
            pg.draw.lines(self.screen, col, False, pts, 2)
            cur = vals[-1]
            label = f"{cur:.2f}" if unit == "a.u." else f"{cur:.1f}"
            self.text(label, (r.right - 12, y0 + 6), TEXT, self.f_mono, "topright")
            self.text(unit, (r.right - 12, y0 + 22), FAINT, self.f_small, "topright")
        for sec in range(int(TRACE_S) + 1):
            x = plot_x1 - sec / TRACE_S * (plot_x1 - plot_x0)
            self.text(f"-{sec}s" if sec else "now", (x, bottom + 2), FAINT, self.f_small, "midtop")
        self.text("stages 1-5: flyvis retina & optic lobe (trained, graded), mean |Δ| from adapted level",
                  (r.x + 12, r.bottom - 26), FAINT, self.f_small)
        self.text("6-8: FlyWire spiking model, mean firing rate · each trace scaled to its own maximum",
                  (r.x + 12, r.bottom - 13), FAINT, self.f_small)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video")
    ap.add_argument("--demo", choices=["loom", "rotate"])
    args = ap.parse_args()
    if args.demo:
        src = DemoSource(args.demo)
    else:
        src = CameraSource(args.camera, args.video)
    App(src).run()


if __name__ == "__main__":
    main()
