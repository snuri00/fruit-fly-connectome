"""
FLY SWAT - try to swat a fly whose escapes are decided by its real connectome.

The swatter's image expanding on the fly's eye drives the looming detectors of
that eye; if a giant fiber spikes in time, the fly jumps. A fly that is busy
drinking sugar water reacts later. The panel on the right shows the brain live
and, after each swing, a millisecond timeline of what the fly's neurons did.

    python3 fly_swat.py                 play
    python3 fly_swat.py --headless 12   measure escape rates (12 swings per condition)

Mouse:  LMB swing   RMB sugar drop   Shift+RMB bitter drop
Keys:   1 / 2 / 3 slow / normal / fast swing    S slow motion during swings
        G cut the giant fibers    N new fly    SPACE pause    ESC quit
"""
import argparse
import math
import os
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from swat_world import (DROP_RADIUS, HOVER_Z, SWATTER, TABLE_H, TABLE_W,
                        SwatWorld)

CANVAS = (1600, 960)
SCALE = 3.85
TABLE_ORIGIN = (24, 84)
TABLE_PX = (int(TABLE_W * SCALE), int(TABLE_H * SCALE))
SIDE_X = 1204
FLY_SPRITE = 160
FLY_ZOOM = 0.48
STRIDE_MM = 1.4

BG = (12, 13, 17)
PANEL = (19, 21, 27)
BORDER = (40, 45, 55)
TEXT = (228, 231, 236)
MUTED = (134, 141, 152)
FAINT = (78, 84, 95)
OFF = (36, 40, 49)
LOOM = (110, 196, 255)
GF = (255, 96, 96)
MN9 = (246, 180, 72)
SUGAR = (220, 236, 255)
BITTER = (190, 220, 90)
GOOD = (120, 230, 150)

SLOWMO = 0.2
NORMAL_SPEED = 1.0
FRAME_BUDGET = 0.030


def to_px(x, y):
    return (TABLE_ORIGIN[0] + x * SCALE, TABLE_ORIGIN[1] + y * SCALE)


def to_mm(px, py):
    return ((px - TABLE_ORIGIN[0]) / SCALE, (py - TABLE_ORIGIN[1]) / SCALE)


def mix(c1, c2, k):
    k = min(max(k, 0.0), 1.0)
    return tuple(int(a + (b - a) * k) for a, b in zip(c1, c2))


class Game:
    def __init__(self):
        import pygame
        from brain_view import BrainMap, Glow
        from fly_sprite import GAIT_FRAMES, build_drop, build_fly, build_swatter
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("Fly Swat - connectome edition")
        self.screen = pygame.display.set_mode(CANVAS)
        self.clock = pygame.time.Clock()
        f = pygame.font.SysFont
        self.f_title = f("dejavusans", 20, bold=True)
        self.f = f("dejavusans", 13)
        self.f_small = f("dejavusans", 11)
        self.f_mono = f("dejavusansmono", 13)
        self.f_big = f("dejavusans", 44, bold=True)
        self.f_num = f("dejavusansmono", 22)

        self.screen.fill(BG)
        msg = self.f_title.render("loading the fly's brain (138,639 neurons) ...", True, MUTED)
        self.screen.blit(msg, msg.get_rect(center=(CANVAS[0] // 2, CANVAS[1] // 2)))
        pygame.display.flip()

        self.world = SwatWorld(seed=int(time.time()))
        self.map = BrainMap(self.world.brain, (360, 180), background=PANEL)
        self.glow = Glow(self.world.brain.n, 200.0)
        self.table = self._make_table()
        self.spr = build_fly(FLY_SPRITE)
        self.gait_frames = GAIT_FRAMES
        self.gait = 0.0
        self._last_fly = None
        max_side = SWATTER * SCALE * 1.35
        head, shadow, frac = build_swatter(int(max_side / (2 * 0.44)) + 1)
        self.sw_head, self.sw_shadow, self.sw_frac = head, shadow, frac
        self.drop_spr = {"sugar": build_drop(96, (214, 232, 255)),
                         "bitter": build_drop(96, (196, 222, 96))}
        self.speed_name = "normal"
        self.slowmo = True
        self.paused = False
        self.popups = []
        self.shake = 0.0
        self.gf_raster = []
        self._reported = None
        pygame.mouse.set_visible(False)

    def _make_table(self):
        """Procedural wooden table top."""
        pg = self.pg
        w, h = TABLE_PX
        rng = np.random.default_rng(3)
        y = np.arange(h)[None, :]
        x = np.arange(w)[:, None]
        grain = (np.sin(y / 9.0 + 3.0 * np.sin(x / 140.0) + rng.normal(0, 0.02, (w, 1)) * 40) * 0.5 + 0.5)
        planks = ((y // 128) % 2) * 0.06
        noise = rng.normal(0, 1, (w, h)) * 0.03
        k = np.clip(0.55 + 0.25 * grain + planks + noise, 0, 1)
        base = np.array([96, 64, 40], np.float32)
        light = np.array([150, 104, 66], np.float32)
        img = base + k[..., None] * (light - base)
        cx, cy = w / 2, h / 2
        r = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        img *= (1 - 0.35 * np.clip(r - 0.4, 0, 1))[..., None]
        seams = (y % 128 == 0)
        img[np.broadcast_to(seams, (w, h))] *= 0.6
        surf = pg.Surface((w, h))
        pg.surfarray.blit_array(surf, img.astype(np.uint8))
        return surf

    def run(self):
        pg = self.pg
        wall_dt = 1 / 30
        running = True
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    running = self.on_key(ev)
                elif ev.type == pg.MOUSEBUTTONDOWN:
                    self.on_click(ev)
            mx, my = pg.mouse.get_pos()
            x, y = to_mm(mx, my)
            self.world.move_swatter(min(max(x, 0), TABLE_W), min(max(y, 0), TABLE_H))
            if not self.paused:
                self.advance(wall_dt)
            self.draw()
            wall_dt = self.clock.tick(60) / 1000.0
        pg.quit()

    def advance(self, wall_dt):
        w = self.world
        swinging = w.swatter.phase in ("down", "impact")
        speed = SLOWMO if (self.slowmo and swinging) else NORMAL_SPEED
        target = min(speed * wall_dt * 1000.0, 50.0)
        t0 = time.perf_counter()
        fired, sim = [], 0.0
        while sim < target and time.perf_counter() - t0 < FRAME_BUDGET:
            f = w.step(1.0)
            if len(f):
                fired.append(f)
                for side, idx in w.gf_idx.items():
                    if np.isin(idx, f).any():
                        self.gf_raster.append((w.t, side))
            sim += 1.0
        self.gf_raster = [(t, s) for t, s in self.gf_raster if w.t - t < 600]
        if sim:
            self.glow.update(np.concatenate(fired) if fired else np.empty(0, np.int64), sim)
        last = w.swings[-1] if w.swings else None
        if last is not None and last.result and self._reported is not last:
            self._reported = last
            pos = to_px(w.swatter.x, w.swatter.y)
            if last.result == "hit":
                self.popup("SPLAT!", GF, pos)
                self.shake = 1.0
            elif last.result == "escaped":
                self.popup("escaped", LOOM, pos)
            else:
                self.popup("miss", MUTED, pos)
        self.shake *= 0.85

    def popup(self, text, color, pos):
        self.popups.append((time.time(), text, color, pos))
        self.popups = self.popups[-4:]

    def on_key(self, ev):
        pg, w = self.pg, self.world
        if ev.key == pg.K_ESCAPE:
            return False
        if ev.key == pg.K_SPACE:
            self.paused = not self.paused
        elif ev.key in (pg.K_1, pg.K_2, pg.K_3):
            self.speed_name = {pg.K_1: "slow", pg.K_2: "normal", pg.K_3: "fast"}[ev.key]
        elif ev.key == pg.K_s:
            self.slowmo = not self.slowmo
        elif ev.key == pg.K_g:
            w.set_gf_ablation(not w.gf_ablated)
        elif ev.key == pg.K_n:
            w.new_fly()
        return True

    def on_click(self, ev):
        pg, w = self.pg, self.world
        x, y = to_mm(*ev.pos)
        if not (0 <= x <= TABLE_W and 0 <= y <= TABLE_H):
            return
        if ev.button == 1:
            w.swing(self.speed_name)
        elif ev.button == 3:
            kind = "bitter" if pg.key.get_mods() & pg.KMOD_SHIFT else "sugar"
            w.add_drop(x, y, kind)

    def text(self, txt, pos, color=TEXT, f=None, anchor="topleft"):
        s = (f or self.f).render(str(txt), True, color)
        r = s.get_rect(**{anchor: pos})
        self.screen.blit(s, r)
        return r

    def draw(self):
        pg = self.pg
        self.screen.fill(BG)
        self.draw_header()
        ox = int(self.rng_shake())
        table_rect = pg.Rect(TABLE_ORIGIN, TABLE_PX).move(ox, 0)
        self.screen.blit(self.table, table_rect)
        pg.draw.rect(self.screen, (60, 42, 28), table_rect.inflate(8, 8), 4, border_radius=4)
        clip = self.screen.get_clip()
        self.screen.set_clip(table_rect)
        self.draw_drops(ox)
        self.draw_swatter_shadow(ox)
        self.draw_fly(ox)
        self.draw_swatter(ox)
        self.draw_popups()
        self.draw_log()
        self.screen.set_clip(clip)
        self.draw_side()
        self.draw_footer()
        pg.display.flip()

    def rng_shake(self):
        return math.sin(time.time() * 90) * 6 * self.shake

    def draw_header(self):
        w = self.world
        self.text("FLY SWAT", (24, 16), TEXT, self.f_title)
        self.text("the fly's escapes are decided by its real connectome  ·  "
                  "FlyWire whole-brain model, 138,639 neurons", (130, 22), MUTED, self.f_small)
        sc = w.score
        stats = [("SWINGS", sc["swings"]), ("HITS", sc["hits"]), ("ESCAPES", sc["escapes"]),
                 ("SWING", self.speed_name), ("SLOW-MO", "on" if self.slowmo else "off")]
        x = SIDE_X - 20
        for label, val in reversed(stats):
            val = str(val)
            wdt = max(self.f_num.size(val)[0], self.f_small.size(label)[0])
            self.text(label, (x - wdt, 12), MUTED, self.f_small)
            self.text(val, (x - wdt, 28), TEXT, self.f_num)
            x -= wdt + 30
        if w.gf_ablated:
            self.text("GIANT FIBERS CUT", (x - 170, 30), GF, self.f_mono)

    def draw_footer(self):
        self.text("LMB swing   ·   RMB sugar drop   ·   Shift+RMB bitter drop   ·   "
                  "1/2/3 swing speed   ·   S slow motion   ·   G cut giant fibers   ·   "
                  "N new fly   ·   SPACE pause   ·   ESC quit",
                  (24, CANVAS[1] - 26), FAINT, self.f_small)

    def draw_drops(self, ox):
        pg = self.pg
        for d in self.world.drops:
            if d.volume <= 0:
                continue
            x, y = to_px(d.x, d.y)
            x += ox
            r = max(3, DROP_RADIUS * math.sqrt(d.volume) * SCALE)
            size = int(r * 2 / 0.88)
            img = pg.transform.smoothscale(self.drop_spr[d.kind], (size, size))
            self.screen.blit(img, img.get_rect(center=(x, y)))
            col = SUGAR if d.kind == "sugar" else BITTER
            self.text(d.kind, (x, y + r + 5), mix(col, BG, 0.25), self.f_small, "midtop")

    def draw_fly(self, ox):
        pg = self.pg
        f = self.world.fly
        sp = self.spr
        x, y = to_px(f.x, f.y)
        x += ox
        ang = -math.degrees(f.heading)

        if self._last_fly is not None and f.state == "walk":
            self.gait += math.hypot(f.x - self._last_fly[0], f.y - self._last_fly[1]) / STRIDE_MM
        self._last_fly = (f.x, f.y)

        def put(surf, cx, cy, zoom, alpha=None):
            img = pg.transform.rotozoom(surf, ang, zoom)
            if alpha is not None:
                img.set_alpha(int(255 * min(max(alpha, 0), 1)))
            self.screen.blit(img, img.get_rect(center=(cx, cy)))

        if f.state == "dead":
            put(sp["splat"], x, y, FLY_ZOOM)
            put(sp["dead"], x, y, FLY_ZOOM)
            return
        off = 3 + f.z * 1.1
        put(sp["shadow"], x + off, y + off, FLY_ZOOM, alpha=1 - min(f.z / 80, 0.6))
        zoom = FLY_ZOOM * (1 + f.z / 90.0)
        lift = f.z * 0.3
        bx, by = x - lift, y - lift
        if f.state in ("air", "takeoff"):
            legs = sp["legs_air"]
        elif f.state == "walk":
            legs = sp["legs"][int(self.gait * self.gait_frames) % self.gait_frames]
        else:
            legs = sp["legs"][0]
        put(legs, bx, by, zoom)
        if f.proboscis > 0.05:
            put(sp["proboscis"], bx, by, zoom, alpha=f.proboscis)
        put(sp["body"], bx, by, zoom)
        if f.state == "air":
            put(sp["wings_flight"][int(self.world.t / 6) % 2], bx, by, zoom)
        elif f.state == "takeoff":
            put(sp["wings_flight"][0], bx, by, zoom)
        else:
            put(sp["wings_rest"], bx, by, zoom)

    def swatter_size(self):
        s = self.world.swatter
        k = 1 - s.z / HOVER_Z
        side = SWATTER * SCALE * (1 + 0.35 * (1 - k))
        return k, side, int(side / (2 * self.sw_frac))

    def draw_swatter_shadow(self, ox):
        pg = self.pg
        s = self.world.swatter
        if not s.visible:
            return
        k, _, px = self.swatter_size()
        x, y = to_px(s.x, s.y)
        off = 26 * (1 - k)
        img = pg.transform.smoothscale(self.sw_shadow, (px, px))
        img.set_alpha(int(70 + 150 * k))
        self.screen.blit(img, img.get_rect(center=(x + off + ox, y + off)))

    def draw_swatter(self, ox):
        pg = self.pg
        s = self.world.swatter
        if not s.visible:
            return
        k, side, px = self.swatter_size()
        x, y = to_px(s.x, s.y)
        x += ox
        img = pg.transform.smoothscale(self.sw_head, (px, px))
        img.set_alpha(int(110 + 130 * k))
        rect = img.get_rect(center=(x, y))
        neck = (x, y + px / 2 - 2)
        end = (x + 60 + 90 * (1 - k), y + px / 2 + 190)
        pg.draw.line(self.screen, (60, 60, 66), neck, end, 5)
        pg.draw.line(self.screen, (120, 40, 34), (neck[0] + (end[0] - neck[0]) * 0.25,
                                                 neck[1] + (end[1] - neck[1]) * 0.25), end, 11)
        pg.draw.line(self.screen, (200, 80, 66), (neck[0] + (end[0] - neck[0]) * 0.25 - 2,
                                                 neck[1] + (end[1] - neck[1]) * 0.25), (end[0] - 2, end[1]), 3)
        self.screen.blit(img, rect)
        pg.draw.circle(self.screen, (255, 255, 255), (x, y), 3)
        pg.draw.circle(self.screen, (30, 20, 20), (x, y), 3, 1)
        if s.phase == "hover":
            self.text(f"{self.speed_name} swing", (rect.right + 8, rect.top + 4), (255, 214, 205), self.f_small)

    def draw_popups(self):
        now = time.time()
        keep = []
        for t, text, color, pos in self.popups:
            age = now - t
            if age > 1.4:
                continue
            keep.append((t, text, color, pos))
            s = self.f_big.render(text, True, color)
            s.set_alpha(int(255 * (1 - age / 1.4)))
            self.screen.blit(s, s.get_rect(center=(pos[0], pos[1] - 40 - age * 40)))
        self.popups = keep

    def draw_log(self):
        w = self.world
        y = TABLE_ORIGIN[1] + TABLE_PX[1] - 22
        for t, text in reversed(w.events[-5:]):
            age = (w.t - t) / 1000
            col = mix(TEXT, (60, 45, 30), min(age / 8, 0.8))
            self.text(f"{t / 1000:6.2f} s  {text}", (TABLE_ORIGIN[0] + 12, y), col, self.f_small)
            y -= 16

    def panel(self, rect, title, subtitle=""):
        pg = self.pg
        pg.draw.rect(self.screen, PANEL, rect, border_radius=6)
        pg.draw.rect(self.screen, BORDER, rect, 1, border_radius=6)
        r = self.text(title, (rect.x + 12, rect.y + 9), TEXT, self.f_mono)
        if subtitle:
            self.text(subtitle, (r.right + 8, rect.y + 11), MUTED, self.f_small)

    def draw_side(self):
        pg, w = self.pg, self.world
        width = CANVAS[0] - SIDE_X - 20
        r = pg.Rect(SIDE_X, 72, width, 236)
        self.panel(r, "FLY BRAIN", "live · every neuron")
        origin = (r.centerx - self.map.w // 2, r.y + 38)
        self.screen.blit(self.map.render(self.glow.level, spot=False, gain=0.35), origin)
        for key, col in (("gf_L", GF), ("gf_R", GF), ("mn9", MN9)):
            for i in w.circ.out[key]:
                on = w.gf_flash.get(key[-1], 0) if key.startswith("gf") else min(w.mn9_hz / 60, 1)
                pg.draw.circle(self.screen, mix(FAINT, col, on), self.map.point(i, origin), 5 + int(3 * on), 1)
        self.text("rings: giant fibers (red), MN9 (amber)", (r.x + 12, r.bottom - 20), FAINT, self.f_small)

        r = pg.Rect(SIDE_X, 320, width, 196)
        self.panel(r, "SENSES → OUTPUTS", "Hz")
        state = {"walk": "walking", "feed": "drinking", "takeoff": "taking off",
                 "air": "flying", "dead": "dead"}[w.fly.state]
        rows = [("left eye  looming", w.loom["left"], 300, LOOM, "LC4 + LPLC2"),
                ("right eye looming", w.loom["right"], 300, LOOM, "LC4 + LPLC2"),
                ("taste", max(w.taste.values()), 150,
                 SUGAR if w.taste["sugar"] else BITTER, "sugar" if w.taste["sugar"] else
                 ("bitter" if w.taste["bitter"] else "-")),
                ("proboscis MN9", w.mn9_hz, 150, MN9, f"fly: {state}")]
        y = r.y + 38
        for label, val, full, col, note in rows:
            self.text(label, (r.x + 12, y), TEXT, self.f)
            self.text(note, (r.x + 12, y + 16), MUTED, self.f_small)
            bar = pg.Rect(r.x + 170, y + 6, width - 240, 10)
            pg.draw.rect(self.screen, OFF, bar, border_radius=3)
            if val > 0.5:
                pg.draw.rect(self.screen, col, (bar.x, bar.y, max(2, bar.w * min(val / full, 1)), bar.h),
                             border_radius=3)
            self.text(f"{val:4.0f}", (r.right - 12, y + 1), TEXT, self.f_mono, "topright")
            y += 38

        r = pg.Rect(SIDE_X, 528, width, 96)
        self.panel(r, "GIANT FIBERS", "spikes, last 0.6 s")
        for n, side in enumerate(("L", "R")):
            yy = r.y + 44 + n * 24
            lamp = w.gf_flash[side]
            pg.draw.circle(self.screen, mix(OFF, GF, lamp), (r.x + 22, yy + 6), 7)
            self.text(side, (r.x + 36, yy - 2), TEXT, self.f_mono)
            x0, x1 = r.x + 60, r.right - 14
            pg.draw.line(self.screen, OFF, (x0, yy + 6), (x1, yy + 6))
            for t, s in self.gf_raster:
                if s == side:
                    x = x1 - (w.t - t) / 600 * (x1 - x0)
                    pg.draw.line(self.screen, GF, (x, yy - 2), (x, yy + 14), 2)

        r = pg.Rect(SIDE_X, 636, width, 260)
        self.panel(r, "LAST SWING", "ms after it began · red curve = swatter height")
        self.draw_timeline(r)

    def draw_timeline(self, r):
        pg, w = self.pg, self.world
        rec = w.swings[-1] if w.swings else None
        if rec is None:
            self.text("click on the table to swing", (r.x + 12, r.y + 40), FAINT, self.f)
            self.text("tip: a fly drinking sugar reacts later", (r.x + 12, r.y + 60), FAINT, self.f)
            return
        span = rec.swing_ms + 40
        x0, x1 = r.x + 20, r.right - 20
        axis_y = r.y + 168

        def X(t):
            return x0 + min(max(t / span, 0), 1) * (x1 - x0)

        pts = []
        for i in range(41):
            t = rec.swing_ms * i / 40
            z = HOVER_Z * (1 - (t / rec.swing_ms) ** 2)
            pts.append((X(t), axis_y - z / HOVER_Z * 80))
        pg.draw.lines(self.screen, (200, 90, 90), False, pts, 2)
        pg.draw.line(self.screen, MUTED, (x0, axis_y), (x1, axis_y))
        for t in range(0, int(span) + 1, 20 if span < 200 else 50):
            pg.draw.line(self.screen, MUTED, (X(t), axis_y), (X(t), axis_y + 4))
            self.text(str(t), (X(t), axis_y + 6), FAINT, self.f_small, "midtop")

        marks = []
        if rec.loom_onset is not None:
            marks.append((rec.loom_onset, "eye", LOOM))
        for side, t in rec.gf.items():
            marks.append((t, f"GF {side}", GF))
        if rec.takeoff is not None:
            marks.append((rec.takeoff, "take-off", GOOD))
        if rec.impact is not None:
            marks.append((rec.impact, "impact", TEXT))
        for n, (t, label, col) in enumerate(sorted(marks)):
            x = X(t)
            pg.draw.line(self.screen, col, (x, axis_y - 84), (x, axis_y), 1)
            ly = axis_y - 86 - 15 * (n % 3)
            self.text(f"{label} {t:.0f}", (x, ly - 8), col, self.f_small, "midbottom")

        verdict = {"hit": ("HIT", GF), "escaped": ("ESCAPED", LOOM), "missed": ("MISSED", MUTED),
                   None: ("...", MUTED)}[rec.result]
        self.text(verdict[0], (r.x + 12, r.bottom - 66), verdict[1], self.f_num)
        why = []
        if rec.feeding:
            why.append("the fly was drinking")
        if rec.gf:
            first = min(rec.gf.values())
            why.append(f"giant fiber fired {first:.0f} ms into a {rec.swing_ms:.0f} ms swing")
        elif rec.result:
            why.append("no giant fiber spike" + (" (fibers cut)" if w.gf_ablated else ""))
        for n, line in enumerate(why[:2]):
            self.text(line, (r.x + 12, r.bottom - 36 + 16 * n), MUTED, self.f_small)


def benchmark(n):
    """Swing at the fly from straight above with a human-like aim error
    (15 mm s.d.) and report how often it is hit in each condition."""
    w = SwatWorld(seed=1)
    conditions = [
        ("walking fly, normal swing", dict()),
        ("drinking fly, normal swing", dict(feed=True)),
        ("walking fly, fast swing", dict(speed="fast")),
        ("walking fly, slow swing", dict(speed="slow")),
        ("giant fibers cut, normal swing", dict(ablate=True)),
    ]

    def trial(feed=False, speed="normal", ablate=False):
        while w.swatter.phase != "hover":
            w.step(1.0)
        w.new_fly()
        w.set_gf_ablation(ablate)
        w.drops = []
        f = w.fly
        if feed:
            w.add_drop(f.x, f.y, "sugar")
        aim = w.rng.normal(0, 15, 2)
        w.swatter.visible = False
        for _ in range(400):
            w.move_swatter(w.fly.x + aim[0], w.fly.y + aim[1])
            w.step(1.0)
        state = w.fly.state
        w.swing(speed)
        while w.swings[-1].result is None:
            w.step(1.0)
        rec = w.swings[-1]
        return state, rec.result, min(rec.gf.values()) if rec.gf else None

    print(f"{'condition':34s} {'hit':>8s} {'GF latency (median)':>21s}")
    t0 = time.time()
    for name, kw in conditions:
        res = [trial(**kw) for _ in range(n)]
        hits = sum(r[1] == "hit" for r in res)
        lat = [r[2] for r in res if r[2] is not None]
        lat_txt = f"{np.median(lat):.0f} ms" if lat else "-"
        print(f"{name:34s} {hits:3d}/{n:<4d} {lat_txt:>21s}")
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--headless", type=int, metavar="N",
                    help="measure hit rates with N swings per condition, no window")
    args = ap.parse_args()
    if args.headless:
        benchmark(args.headless)
    else:
        Game().run()
