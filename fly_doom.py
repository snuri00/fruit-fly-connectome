"""
FLY DOOM - the whole-brain fruit fly model plays ViZDoom "Defend the Center".

Small, distant enemies drive the fly's LC10 neurons and it turns towards them;
large, near ones drive LC4/LPLC2, it turns away and its giant fiber fires,
which pulls the trigger. Nothing is trained: this is the connectome's reflexes.

    python3 fly_doom.py                 watch the fly play
    python3 fly_doom.py --headless 10   compare with random and idle players
    python3 fly_doom.py --learn 20      20 episodes with dopamine learning (none /
                                        mushroom body / steering synapses)

Keys:  SPACE pause   L learning: off / mushroom body / steering synapses
       P switch player (fly brain / random)   G cut giant fibers   R restart   ESC quit
"""
import argparse
import math
import os
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from doom_world import BLIND, TIC_MS, DoomWorld

CANVAS = (1600, 960)
VIEW = (20, 72, 960, 720)
EYE = (20, 806, 960, 132)
SIDE_X = 1000

BG = (12, 12, 15)
PANEL = (20, 20, 25)
BORDER = (44, 44, 52)
TEXT = (230, 230, 234)
MUTED = (140, 140, 150)
FAINT = (80, 80, 90)
OFF = (38, 38, 46)
SMALL = (120, 230, 150)
LOOM = (255, 90, 80)
GF = (255, 90, 80)
STEER = (90, 190, 255)
GOLD = (240, 190, 80)
REWARD = (120, 230, 150)
LEARN_MODES = ["none", "mb", "steer"]
LEARN_NAMES = {"none": "off", "mb": "mushroom body", "steer": "steering synapses"}
LEARN_COLORS = {"none": (150, 150, 160), "mb": (200, 150, 255), "steer": (90, 190, 255)}


def mix(c1, c2, k):
    k = min(max(k, 0.0), 1.0)
    return tuple(int(a + (b - a) * k) for a, b in zip(c1, c2))


class App:
    def __init__(self):
        import pygame
        from brain_view import BrainMap, Glow
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("Fly DOOM - connectome edition")
        self.screen = pygame.display.set_mode(CANVAS)
        self.clock = pygame.time.Clock()
        f = pygame.font.SysFont
        self.f_title = f("dejavusans", 20, bold=True)
        self.f = f("dejavusans", 13)
        self.f_small = f("dejavusans", 11)
        self.f_mono = f("dejavusansmono", 13)
        self.f_num = f("dejavusansmono", 22)
        self.f_big = f("dejavusans", 40, bold=True)
        self.screen.fill(BG)
        msg = self.f_title.render("loading the fly's brain and DOOM ...", True, MUTED)
        self.screen.blit(msg, msg.get_rect(center=(CANVAS[0] // 2, CANVAS[1] // 2)))
        pygame.display.flip()

        self.world = DoomWorld(seed=int(time.time()))
        self.map = BrainMap(self.world.brain, (560, 220), background=PANEL)
        self.glow = Glow(self.world.brain.n, 200.0)
        self.paused = False
        self.history = []
        self.episode = 1
        self.gf_raster = []
        self.btn_flash = [0.0, 0.0, 0.0]
        self.end_time = None
        self.learn_mode = "none"
        self.learners = {}

    def set_learning(self, mode):
        from fly_learning import LEARNERS
        w = self.world
        if w.learner is not None:
            w.learner.reset_weights()
        if mode == "none":
            w.learner = None
        else:
            if mode not in self.learners:
                self.learners[mode] = LEARNERS[mode](w.brain)
            w.learner = self.learners[mode]
            w.learner.reset_weights()
        self.learn_mode = mode
        self.restart()

    def run(self):
        pg = self.pg
        running = True
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    running = self.on_key(ev.key)
            if not self.paused:
                self.advance()
            self.draw()
            self.clock.tick(60)
        self.world.game.close()
        pg.quit()

    def on_key(self, key):
        pg, w = self.pg, self.world
        if key == pg.K_ESCAPE:
            return False
        if key == pg.K_SPACE:
            self.paused = not self.paused
        elif key == pg.K_p:
            w.policy = "random" if w.policy == "connectome" else "connectome"
            self.restart()
        elif key == pg.K_g:
            w.set_gf_cut(not w.gf_cut)
        elif key == pg.K_l:
            nxt = LEARN_MODES[(LEARN_MODES.index(self.learn_mode) + 1) % len(LEARN_MODES)]
            self.set_learning(nxt)
        elif key == pg.K_r:
            self.restart()
        return True

    def restart(self):
        w = self.world
        if w.tic > 0:
            self.history.append({**w.summary(), "policy": w.policy, "gf_cut": w.gf_cut,
                                 "n": self.episode, "done": w.done, "learn": self.learn_mode})
        self.history = self.history[-16:]
        self.episode += 1
        w.reset()
        self.glow.clear()
        self.end_time = None

    def advance(self):
        w = self.world
        if w.done:
            if self.end_time is None:
                self.end_time = time.time()
            elif time.time() - self.end_time > 3.0:
                self.restart()
            return
        t0 = time.perf_counter()
        fired = []
        while time.perf_counter() - t0 < 0.035 and not w.done:
            w.step()
            if len(w.fired):
                fired.append(w.fired)
            if w.gf_spikes:
                self.gf_raster.append(w.tic)
            for i, a in enumerate(w.action):
                if a:
                    self.btn_flash[i] = 1.0
            if w.policy != "connectome":
                break
        self.gf_raster = [t for t in self.gf_raster if w.tic - t < 70]
        self.glow.update(np.concatenate(fired) if fired else np.empty(0, np.int64), TIC_MS)
        self.btn_flash = [f * 0.7 for f in self.btn_flash]

    def text(self, txt, pos, color=TEXT, f=None, anchor="topleft"):
        s = (f or self.f).render(str(txt), True, color)
        r = s.get_rect(**{anchor: pos})
        self.screen.blit(s, r)
        return r

    def panel(self, rect, title, subtitle=""):
        pg = self.pg
        pg.draw.rect(self.screen, PANEL, rect, border_radius=6)
        pg.draw.rect(self.screen, BORDER, rect, 1, border_radius=6)
        r = self.text(title, (rect.x + 12, rect.y + 9), TEXT, self.f_mono)
        if subtitle:
            self.text(subtitle, (r.right + 8, rect.y + 11), MUTED, self.f_small)

    def draw(self):
        self.screen.fill(BG)
        self.draw_header()
        self.draw_view()
        self.draw_eye()
        self.draw_side()
        self.text("SPACE pause   ·   L dopamine learning   ·   P switch player (fly brain / random)   ·   "
                  "G cut giant fibers   ·   R restart   ·   ESC quit",
                  (20, CANVAS[1] - 20), FAINT, self.f_small)
        self.pg.display.flip()

    def draw_header(self):
        w = self.world
        self.text("FLY DOOM", (20, 16), TEXT, self.f_title)
        self.text("a fruit-fly connectome (FlyWire, 138,639 neurons) plays Defend the Center  ·  "
                  "nothing is trained", (132, 22), MUTED, self.f_small)
        who = "FLY BRAIN" if w.policy == "connectome" else "RANDOM"
        stats = [("PLAYER", who), ("EPISODE", self.episode), ("KILLS", w.vars["kills"]),
                 ("HEALTH", max(w.vars["health"], 0)), ("AMMO", w.vars["ammo"]),
                 ("TIME", f"{w.tic / 35:5.1f}s")]
        x = CANVAS[0] - 20
        for label, val in reversed(stats):
            val = str(val)
            wd = max(self.f_num.size(val)[0], self.f_small.size(label)[0])
            self.text(label, (x - wd, 10), MUTED, self.f_small)
            self.text(val, (x - wd, 26), GOLD if label == "KILLS" else TEXT, self.f_num)
            x -= wd + 26
        if w.gf_cut:
            self.text("GIANT FIBERS CUT", (x - 160, 30), GF, self.f_mono)

    def draw_view(self):
        pg, w = self.pg, self.world
        x, y, vw, vh = VIEW
        if w.screen is not None:
            surf = pg.surfarray.make_surface(np.transpose(w.screen, (1, 0, 2)))
            sx, sy = vw / surf.get_width(), vh / surf.get_height()
            self.screen.blit(pg.transform.scale(surf, (vw, vh)), (x, y))
            by_id = {e.id: e for e in w.enemies}
            for lab in w.labels:
                e = by_id.get(lab.object_id)
                if e is None:
                    continue
                r = pg.Rect(x + lab.x * sx, y + lab.y * sy, lab.width * sx, lab.height * sy)
                threat = e.loom > e.small
                col = LOOM if threat else SMALL
                pg.draw.rect(self.screen, col, r, 2)
                tag = f"threat {e.loom:.0f} Hz" if threat else f"track {e.small:.0f} Hz"
                self.text(tag, (r.x, r.y - 16), col, self.f_small)
        pg.draw.rect(self.screen, BORDER, VIEW, 1)
        cx = x + vw // 2
        pg.draw.line(self.screen, (255, 255, 255), (cx - 8, y + vh // 2), (cx + 8, y + vh // 2), 1)
        pg.draw.line(self.screen, (255, 255, 255), (cx, y + vh // 2 - 8), (cx, y + vh // 2 + 8), 1)
        if w.done:
            dead = w.game.is_player_dead()
            msg = "THE FLY IS DEAD" if dead else "TIME IS UP"
            s = self.f_big.render(msg, True, GF if dead else GOLD)
            self.screen.blit(s, s.get_rect(center=(x + vw // 2, y + vh // 2 - 30)))
            self.text(f"{w.vars['kills']} kills in {w.tic / 35:.1f} s", (x + vw // 2, y + vh // 2 + 10),
                      TEXT, self.f_title, "midtop")

    def draw_eye(self):
        pg, w = self.pg, self.world
        rect = pg.Rect(EYE)
        self.panel(rect, "WHAT THE FLY SEES", "panoramic eye, 330°  ·  green: LC10 (turn towards)  ·  "
                   "red: LC4 + LPLC2 (turn away, giant fiber)")
        strip = pg.Rect(rect.x + 14, rect.y + 34, rect.w - 28, rect.h - 52)
        pg.draw.rect(self.screen, (16, 16, 20), strip)
        cx = strip.centerx
        scale = strip.w / (2 * BLIND)
        for deg in range(-150, 151, 30):
            xx = cx - deg * scale
            pg.draw.line(self.screen, OFF, (xx, strip.y), (xx, strip.bottom))
            self.text(f"{deg:+d}°" if deg else "ahead", (xx, strip.bottom + 2), FAINT, self.f_small, "midtop")
        vx = 45 * scale
        pg.draw.rect(self.screen, (30, 30, 40), (cx - vx, strip.y, 2 * vx, strip.h))
        self.text("screen", (cx, strip.y + 2), FAINT, self.f_small, "midtop")
        self.text("LEFT EYE", (strip.x + 6, strip.y + 4), MUTED, self.f_small)
        self.text("RIGHT EYE", (strip.right - 6, strip.y + 4), MUTED, self.f_small, "topright")
        for e in sorted(w.enemies, key=lambda e: -e.dist):
            if not e.seen:
                continue
            xx = cx - e.bearing * scale
            rad = max(3, min(strip.h / 2 - 4, math.degrees(e.theta) * scale / 2))
            k = e.loom / (e.loom + e.small + 1e-6)
            col = mix(SMALL, LOOM, k)
            pg.draw.circle(self.screen, col, (xx, strip.centery), rad)
            pg.draw.circle(self.screen, (255, 255, 255), (xx, strip.centery), rad, 1)

    def draw_side(self):
        pg, w = self.pg, self.world
        width = CANVAS[0] - SIDE_X - 20

        r = pg.Rect(SIDE_X, 72, width, 262)
        self.panel(r, "FLY BRAIN", "live · every neuron")
        origin = (r.centerx - self.map.w // 2, r.y + 34)
        self.screen.blit(self.map.render(self.glow.level, spot=False, gain=0.35), origin)

        r = pg.Rect(SIDE_X, 346, width, 170)
        self.panel(r, "EYE → BRAIN", "Hz of input")
        rows = [("LC10  left", w.drive["small_left"], SMALL), ("LC10  right", w.drive["small_right"], SMALL),
                ("LC4+LPLC2  left", w.drive["loom_left"], LOOM), ("LC4+LPLC2  right", w.drive["loom_right"], LOOM)]
        y = r.y + 38
        for label, val, col in rows:
            self.text(label, (r.x + 12, y), TEXT, self.f)
            bar = pg.Rect(r.x + 170, y + 4, width - 250, 10)
            pg.draw.rect(self.screen, OFF, bar, border_radius=3)
            if val > 0.5:
                pg.draw.rect(self.screen, col, (bar.x, bar.y, max(2, bar.w * min(val / 300, 1)), bar.h),
                             border_radius=3)
            self.text(f"{val:4.0f}", (r.right - 12, y), TEXT, self.f_mono, "topright")
            y += 30

        r = pg.Rect(SIDE_X, 528, width, 206)
        self.panel(r, "BRAIN → BUTTONS", "DNa01+DNa02 steering, giant fiber trigger")
        L, R = w.steer["left"], w.steer["right"]
        mid = r.centerx
        bar_y = r.y + 50
        span = (width - 60) / 2
        pg.draw.line(self.screen, OFF, (mid - span, bar_y), (mid + span, bar_y), 8)
        diff = L - R
        end = mid - max(-1, min(1, diff / 60)) * span
        pg.draw.line(self.screen, STEER, (mid, bar_y), (end, bar_y), 8)
        pg.draw.line(self.screen, MUTED, (mid, bar_y - 10), (mid, bar_y + 10), 1)
        self.text(f"DNa left {L:3.0f} Hz", (mid - span, bar_y + 12), MUTED, self.f_small)
        self.text(f"DNa right {R:3.0f} Hz", (mid + span, bar_y + 12), MUTED, self.f_small, "topright")
        raster = pg.Rect(r.x + 110, r.y + 94, width - 130, 16)
        self.text("giant fiber", (r.x + 12, r.y + 94), TEXT, self.f)
        pg.draw.rect(self.screen, OFF, raster)
        for t in self.gf_raster:
            xx = raster.right - (w.tic - t) / 70 * raster.w
            pg.draw.line(self.screen, GF, (xx, raster.y), (xx, raster.bottom), 2)
        labels = ["TURN LEFT", "TURN RIGHT", "FIRE"]
        cols = [STEER, STEER, GF]
        bw = (width - 24 - 20) / 3
        for i in range(3):
            b = pg.Rect(r.x + 12 + i * (bw + 10), r.y + 130, bw, 58)
            on = self.btn_flash[i]
            pg.draw.rect(self.screen, mix(OFF, cols[i], on), b, border_radius=6)
            self.text(labels[i], b.center, BG if on > 0.5 else MUTED, self.f_mono, "center")

        r = pg.Rect(SIDE_X, 746, width, 192)
        mode = self.learn_mode
        self.panel(r, "DOPAMINE LEARNING", f"L: {LEARN_NAMES[mode]}")
        y = r.y + 36
        for label, key, col in (("PAM  reward (kill)", "reward", REWARD), ("PPL1  punishment (hit)", "punish", GF)):
            on = w.dan_timer[key] > 0
            pg.draw.circle(self.screen, col if on else OFF, (r.x + 20, y + 8), 6)
            self.text(label, (r.x + 32, y), TEXT if on else MUTED, self.f)
            y += 20
        stats = w.learner.stats() if w.learner is not None else None
        if stats is None:
            line = "plasticity off: dopamine neurons fire, no synapse changes"
        elif mode == "mb":
            line = (f"KC→MBON synapses depressed: {stats['depressed']:,} of {stats['synapses']:,}"
                    f"  ·  mean weight {100 * stats['mean_w']:.1f}%")
        else:
            line = (f"synapses onto DNa changed: {stats['changed']} of {stats['synapses']}"
                    f"  ·  mean weight {100 * stats['mean_w']:.1f}%")
        self.text(line, (r.x + 12, y + 2), MUTED, self.f_small)
        chart = pg.Rect(r.x + 12, y + 24, width - 24, r.bottom - y - 34)
        pg.draw.line(self.screen, OFF, chart.bottomleft, chart.bottomright)
        rows = [h for h in self.history if h["policy"] == "connectome"][-14:]
        top = max([h["kills"] for h in rows] + [5])
        bw = chart.w / 14
        for i, h in enumerate(rows):
            hgt = (chart.h - 14) * h["kills"] / top
            bar = pg.Rect(chart.x + i * bw + 2, chart.bottom - hgt, bw - 4, hgt)
            pg.draw.rect(self.screen, LEARN_COLORS[h["learn"]], bar, border_radius=2)
            self.text(h["kills"], (bar.centerx, bar.top - 13), MUTED, self.f_small, "midtop")
        if not rows:
            self.text("kills per episode will appear here", (chart.x, chart.y + 10), FAINT, self.f)
        self.text("kills / episode", (chart.right, chart.y - 2), FAINT, self.f_small, "topright")


def benchmark(n):
    from fly_brain import FlyBrain
    brain = FlyBrain(seed=0)
    lc10 = None
    conditions = [("fly brain", "connectome", None), ("fly brain, LC10 silenced", "connectome", "lc10"),
                  ("fly brain, giant fibers cut", "connectome", "gf"), ("random buttons", "random", None),
                  ("idle", "noop", None)]
    print(f"{'player':30s} {'kills (mean)':>13s} {'survival (mean)':>16s}")
    t0 = time.time()
    for name, policy, lesion in conditions:
        kills, surv = [], []
        for ep in range(n):
            w = DoomWorld(brain=brain, seed=100 + ep, resolution="RES_160X120", policy=policy)
            if lesion == "lc10":
                lc10 = np.concatenate(list(w.small_idx.values()))
                brain.set_silenced(lc10)
            w.set_gf_cut(lesion == "gf")
            while not w.done:
                w.step()
            s = w.summary()
            kills.append(s["kills"])
            surv.append(s["tics"] / 35)
            w.game.close()
            brain.set_silenced([])
        print(f"{name:30s} {np.mean(kills):13.1f} {np.mean(surv):14.1f} s")
    print(f"({time.time() - t0:.0f} s)")


def learn_experiment(n):
    """Each condition starts from a fresh brain and plays the same n games."""
    from fly_brain import FlyBrain
    from fly_learning import LEARNERS
    results = {}
    t0 = time.time()
    for cond in ("none", "mb", "steer"):
        brain = FlyBrain(seed=0)
        learner = LEARNERS[cond](brain) if cond != "none" else None
        rows = []
        for ep in range(n):
            w = DoomWorld(brain=brain, seed=400 + ep, resolution="RES_160X120", learner=learner)
            on = tot = left = right = 0
            while not w.done:
                w.step()
                tot += 1
                on += any(abs(e.bearing) < 4 for e in w.enemies)
                left += w.action[0]
                right += w.action[1]
            rows.append((w.vars["kills"], w.tic / 35, on / max(tot, 1), left, right))
            w.game.close()
        results[cond] = np.array(rows)
        k = results[cond][:, 0]
        extra = learner.stats() if learner else {}
        print(f"{cond:6s} kills per episode {k.astype(int).tolist()}  "
              f"first half {k[:n // 2].mean():.2f}  second half {k[n // 2:].mean():.2f}  {extra}", flush=True)
    print(f"\n{'condition':20s} {'kills 1st half':>15s} {'kills 2nd half':>15s} {'on target':>10s} {'turns L:R':>10s}")
    names = {"none": "no learning", "mb": "mushroom body", "steer": "steering synapses"}
    for cond, r in results.items():
        h = n // 2
        print(f"{names[cond]:20s} {r[:h, 0].mean():15.2f} {r[h:, 0].mean():15.2f} "
              f"{100 * r[:, 2].mean():9.1f}% {r[:, 3].sum() / max(r[:, 4].sum(), 1):9.2f}")
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--headless", type=int, metavar="N",
                    help="play N episodes per condition without a window and print the scores")
    ap.add_argument("--learn", type=int, metavar="N",
                    help="play N episodes per learning condition without a window")
    args = ap.parse_args()
    if args.learn:
        learn_experiment(args.learn)
    elif args.headless:
        benchmark(args.headless)
    else:
        App().run()
