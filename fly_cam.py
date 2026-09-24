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

Keys: SPACE pause   ESC quit
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
EYES_RECT = (600, 72, 580, 420)
MOTION_RECT = (600, 504, 580, 432)
SIDE_X = 1200

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
TRACE_S = 6.0
BRAIN_STEP_MS = 10.0
FRAME_BUDGET = 0.045


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
        import pygame
        from brain_view import BrainMap, Glow
        from fly_brain import FlyBrain
        from fly_circuits import Circuits
        from fly_eye import FlyEyes
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("Fly Eye Cam")
        self.screen = pygame.display.set_mode(CANVAS)
        self.clock = pygame.time.Clock()
        f = pygame.font.SysFont
        self.f_title = f("dejavusans", 20, bold=True)
        self.f = f("dejavusans", 13)
        self.f_small = f("dejavusans", 11)
        self.f_mono = f("dejavusansmono", 13)
        self.f_big = f("dejavusans", 46, bold=True)
        self.screen.fill(BG)
        s = self.f_title.render("loading the fly's eyes and brain ...", True, MUTED)
        self.screen.blit(s, s.get_rect(center=(CANVAS[0] // 2, CANVAS[1] // 2)))
        pygame.display.flip()

        self.source = source
        self.brain = FlyBrain(seed=int(time.time()))
        self.brain.run(0.2)
        self.eyes = FlyEyes(self.brain)
        self.circ = Circuits(self.brain)
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
        self.map = BrainMap(b, (360, 180), background=PANEL)
        self.glow = Glow(b.n, 200.0)
        self.hex_px = self._hex_layout()
        self.motion = np.zeros((2, 2, self.eyes.n_hex))
        self.trace = deque()
        self.paused = False
        self.speed = 0.0
        self.last_frame = None

    def _hex_layout(self):
        """Screen offsets of the ommatidia: flyvis pixel coordinates are a flat-top
        hexagonal grid of hexagon radius 1, with y pointing up."""
        size = 5.4
        return self.eyes.hx * size, -self.eyes.hy * size, size

    def run(self):
        pg = self.pg
        running = True
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                    running = False
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
        now = time.time()
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
        self.text("SPACE pause   ·   ESC quit   ·   stay still and the fly's brain goes quiet; move and watch it react",
                  (20, CANVAS[1] - 20), FAINT, self.f_small)
        self.pg.display.flip()

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
            cols = []
            for a, m in zip(ang, mag):
                h = (a / (2 * math.pi)) % 1.0
                rgb = [0.5 + 0.5 * math.cos(2 * math.pi * (h - o)) for o in (0.0, 1 / 3, 2 / 3)]
                cols.append(tuple(int(12 + 243 * m * c) for c in rgb))
            self._hexes((cx, cy), cols, flip_x=(name == "right"))
            self.text(f"{name} eye", (cx, r.bottom - 44), MUTED, self.f_small, "midtop")
        lx = r.x + 14
        for label, a in (("→", 0.0), ("↑", math.pi / 2), ("←", math.pi), ("↓", -math.pi / 2)):
            h = (a / (2 * math.pi)) % 1.0
            rgb = tuple(int(12 + 243 * (0.5 + 0.5 * math.cos(2 * math.pi * (h - o)))) for o in (0.0, 1 / 3, 2 / 3))
            self.pg.draw.circle(self.screen, rgb, (lx + 6, r.bottom - 14), 6)
            self.text(label, (lx + 16, r.bottom - 22), MUTED, self.f)
            lx += 50

    def draw_side(self):
        pg = self.pg
        width = CANVAS[0] - SIDE_X - 20
        r = self.panel((SIDE_X, 72, width, 250), "FLY BRAIN", "live")
        origin = (r.centerx - self.map.w // 2, r.y + 40)
        self.screen.blit(self.map.render(self.glow.level, spot=False, gain=0.35), origin)

        r = self.panel((SIDE_X, 334, width, 380), "WHAT THE BRAIN MAKES OF IT", "Hz, left | right")
        rows = [("LC4", "looming", LOOM, 150), ("LPLC2", "looming", LOOM, 150),
                ("HS", "rotation (front→back)", ROT, 250), ("H2", "rotation (back→front)", ROT, 250),
                ("DNa", "steering", STEER, 120), ("GF", "giant fiber: escape jump", GF, 150)]
        y = r.y + 40
        half = (width - 150) // 2
        for g, role, col, full in rows:
            self.text(g, (r.x + 12, y), TEXT, self.f_mono)
            self.text(role, (r.x + 12, y + 17), MUTED, self.f_small)
            for k, s in enumerate(("left", "right")):
                v = self.rate[g][s]
                bar = pg.Rect(r.x + 130 + k * (half + 10), y + 6, half, 10)
                pg.draw.rect(self.screen, OFF, bar, border_radius=3)
                frac = min(v / full, 1.0)
                if frac > 0.005:
                    if s == "left":
                        pg.draw.rect(self.screen, col, (bar.right - bar.w * frac, bar.y, bar.w * frac, bar.h), border_radius=3)
                    else:
                        pg.draw.rect(self.screen, col, (bar.x, bar.y, bar.w * frac, bar.h), border_radius=3)
                self.text(f"{v:3.0f}", (bar.x if s == "left" else bar.right, y + 18), FAINT, self.f_small,
                          "topleft" if s == "left" else "topright")
            y += 54

        r = self.panel((SIDE_X, 726, width, 210), "HONEST NOTES", "")
        notes = ["The eyes (flyvis) are trained and match real",
                 "motion tuning: HS/H2 respond mirror-symmetrically.",
                 "The central brain uses one strength per synapse:",
                 "LPLC2 barely responds, the giant fiber also fires",
                 "for large moving patterns, and DNa steering has a",
                 "left bias. Movements, not still scenes, drive it."]
        y = r.y + 36
        for line in notes:
            self.text(line, (r.x + 12, y), MUTED, self.f_small)
            y += 17


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
