"""
The fly-swatting world: a fly on a table, a swatter, and drops of sugar or
bitter water. Everything runs on the brain's clock (model milliseconds).

What the connectome decides:
  - whether and when the fly escapes: the swatter's image expanding on the eye
    drives the looming detectors LC4 and LPLC2 of that eye; the fly takes off
    when a giant fiber (DNp01) spikes
  - whether it feeds: standing in a drop drives the sugar or bitter taste
    neurons; the fly stops and drinks while the proboscis motor neuron MN9 fires

What is scripted:
  - the walking path (random walk, drawn to sugar drops within smelling range)
  - the direction of the jump (away from the threat) and the flight path
"""
import math

import numpy as np

from fly_brain import FlyBrain
from fly_circuits import Circuits

TABLE_W, TABLE_H = 300.0, 200.0
SWATTER = 90.0
HOVER_Z = 140.0
SWING_MS = {"slow": 220.0, "normal": 130.0, "fast": 80.0}
IMPACT_HOLD = 160.0
LIFT_MS = 260.0
HAND_SPEED = 1500.0

WALK_SPEED = 18.0
TURN_NOISE = 2.2
SMELL_RANGE = 70.0
DROP_RADIUS = 7.0
TAKEOFF_DELAY = 6.0
JUMP_SPEED = 550.0
FLIGHT_MS = (450.0, 800.0)
FEED_ON_HZ = 20.0
TASTE_RATE = 150.0
DRINK_RATE = 0.6

LOOM_GAIN = 12.0
LOOM_MIN = 3.0
LOOM_MAX = 300.0


class Drop:
    def __init__(self, x, y, kind):
        self.x, self.y, self.kind = x, y, kind
        self.volume = 1.0


class Fly:
    def __init__(self, rng):
        self.rng = rng
        self.x = rng.uniform(60, TABLE_W - 60)
        self.y = rng.uniform(50, TABLE_H - 50)
        self.heading = rng.uniform(-math.pi, math.pi)
        self.state = "walk"
        self.z = 0.0
        self.vx = self.vy = 0.0
        self.t_state = 0.0
        self.flight_ms = 0.0
        self.path = None
        self.proboscis = 0.0


class Swatter:
    def __init__(self):
        self.x, self.y = TABLE_W / 2, TABLE_H / 2
        self.target = (self.x, self.y)
        self.z = HOVER_Z
        self.phase = "hover"
        self.t = 0.0
        self.swing_ms = SWING_MS["normal"]
        self.visible = False


class SwingRecord:
    """Timeline of one swing, in ms from the start of the swing."""

    def __init__(self, t0, swing_ms, feeding):
        self.t0 = t0
        self.swing_ms = swing_ms
        self.feeding = feeding
        self.gf = {}
        self.takeoff = None
        self.impact = None
        self.result = None
        self.loom_onset = None


class SwatWorld:
    def __init__(self, brain=None, seed=None):
        self.rng = np.random.default_rng(seed)
        self.brain = brain or FlyBrain(seed=seed)
        self.brain.run(0.2)
        self.circ = Circuits(self.brain)
        self.gf_idx = {"L": self.circ.out["gf_L"], "R": self.circ.out["gf_R"]}
        self.mn9_idx = self.circ.out["mn9"]
        self.gf_ablated = False
        self.reset()

    def reset(self):
        self.brain.reset()
        self.t = 0.0
        self.fly = Fly(self.rng)
        self.swatter = Swatter()
        self.drops = []
        self.swings = []
        self.score = {"swings": 0, "hits": 0, "escapes": 0}
        self.mn9_hz = 0.0
        self.gf_flash = {"L": 0.0, "R": 0.0}
        self.loom = {"left": 0.0, "right": 0.0}
        self.taste = {"sugar": 0.0, "bitter": 0.0}
        self.smell = 0.0
        self._theta = None
        self.events = []
        self.fired = np.empty(0, dtype=np.int64)

    def new_fly(self):
        self.fly = Fly(self.rng)
        self._theta = None
        self.brain.reset()
        self.event("a new fly lands on the table")

    def set_gf_ablation(self, on):
        """Cut the giant fibers: they still spike, but the fly cannot jump."""
        self.gf_ablated = on
        self.event("giant fibers cut" if on else "giant fibers restored")

    def event(self, text):
        self.events.append((self.t, text))
        del self.events[:-8]

    def add_drop(self, x, y, kind):
        self.drops.append(Drop(x, y, kind))
        del self.drops[:-6]

    def move_swatter(self, x, y):
        """Aim the swatter; it follows at hand speed (and not while down)."""
        s = self.swatter
        s.target = (x, y)
        if not s.visible:
            s.visible = True
            s.x, s.y = x, y
            self._theta = None

    def swing(self, speed="normal"):
        s = self.swatter
        if s.phase != "hover":
            return False
        s.phase, s.t, s.swing_ms = "down", 0.0, SWING_MS[speed]
        self.score["swings"] += 1
        self.swings.append(SwingRecord(self.t, s.swing_ms, self.fly.state == "feed"))
        del self.swings[:-20]
        return True

    def _update_swatter(self, ms):
        s = self.swatter
        s.t += ms
        if s.phase in ("hover", "up"):
            dx, dy = s.target[0] - s.x, s.target[1] - s.y
            d = math.hypot(dx, dy)
            step = HAND_SPEED * ms * 1e-3
            if d <= step:
                s.x, s.y = s.target
            else:
                s.x += dx / d * step
                s.y += dy / d * step
        if s.phase == "down":
            k = min(s.t / s.swing_ms, 1.0)
            s.z = HOVER_Z * (1.0 - k * k)
            f = self.fly
            if f.state == "air" and f.z >= s.z and self.under_swatter(f.x, f.y):
                self._impact(mid_air=True)
            if k >= 1.0:
                s.phase, s.t = "impact", 0.0
                if self.fly.state != "dead":
                    self._impact()
        elif s.phase == "impact" and s.t >= IMPACT_HOLD:
            s.phase, s.t = "up", 0.0
        elif s.phase == "up":
            k = min(s.t / LIFT_MS, 1.0)
            s.z = HOVER_Z * (1 - (1 - k) ** 2)
            if k >= 1.0:
                s.phase, s.z = "hover", HOVER_Z

    def under_swatter(self, x, y, margin=0.0):
        s = self.swatter
        h = SWATTER / 2 + margin
        return abs(x - s.x) <= h and abs(y - s.y) <= h

    def _impact(self, mid_air=False):
        f = self.fly
        rec = self.swings[-1] if self.swings else None
        on_table = f.state in ("walk", "feed", "takeoff")
        hit = mid_air or (on_table and self.under_swatter(f.x, f.y))
        if rec:
            rec.impact = self.t - rec.t0
        if hit:
            f.state, f.t_state = "dead", 0.0
            self.score["hits"] += 1
            if rec:
                rec.result = "hit"
            self.event("SPLAT - caught in mid-air" if mid_air else "SPLAT - the fly was hit")
        else:
            escaped = rec is not None and rec.takeoff is not None
            if rec:
                rec.result = "escaped" if escaped else "missed"
            if escaped:
                self.score["escapes"] += 1
            self.event("the fly escaped" if escaped else "missed")

    def _looming(self, ms):
        """Angular expansion of the swatter on each eye -> Hz on LC4/LPLC2."""
        f, s = self.fly, self.swatter
        self.loom = {"left": 0.0, "right": 0.0}
        if not s.visible or f.state in ("air", "dead"):
            self._theta = None
            return
        dx, dy, dz = s.x - f.x, s.y - f.y, s.z - f.z
        dist = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1.0)
        theta = 2 * math.atan(SWATTER / 2 / dist)
        if self._theta is None:
            self._theta = theta
            return
        dtheta = (theta - self._theta) / (ms * 1e-3)
        self._theta = theta
        if dtheta < LOOM_MIN:
            return
        rate = min(LOOM_GAIN * dtheta, LOOM_MAX)
        az = math.atan2(dy, dx) - f.heading
        right = math.sin(az) * math.hypot(dx, dy) / dist
        w_left = min(max(0.5 - 0.75 * right, 0.0), 1.0)
        self.loom = {"left": rate * min(1.0, 2 * w_left),
                     "right": rate * min(1.0, 2 * (1 - w_left))}
        rec = self.swings[-1] if self.swings else None
        if rec and rec.loom_onset is None and s.phase == "down":
            rec.loom_onset = self.t - rec.t0

    def _drop_under_fly(self):
        f = self.fly
        for d in self.drops:
            if d.volume > 0 and math.hypot(d.x - f.x, d.y - f.y) < DROP_RADIUS * math.sqrt(d.volume) + 2:
                return d
        return None

    def _senses(self):
        f = self.fly
        on_table = f.state in ("walk", "feed")
        d = self._drop_under_fly() if on_table else None
        self.taste = {"sugar": 0.0, "bitter": 0.0}
        if d is not None:
            self.taste[d.kind] = TASTE_RATE
        near = [math.hypot(d.x - f.x, d.y - f.y) for d in self.drops
                if d.kind == "sugar" and d.volume > 0]
        self.smell = max(0.0, 1 - min(near) / SMELL_RANGE) * 120.0 if near and on_table else 0.0
        drive = {}
        for side in ("left", "right"):
            drive["loom", side] = self.loom[side]
            drive["sugar", side] = self.taste["sugar"]
            drive["bitter", side] = self.taste["bitter"]
            drive["vinegar", side] = self.smell
        self.brain.set_input(self.circ.drive(drive))
        return d

    def _takeoff(self, side):
        f = self.fly
        s = self.swatter
        dx, dy = f.x - s.x, f.y - s.y
        if math.hypot(dx, dy) < 5:
            ang = f.heading
        else:
            ang = math.atan2(dy, dx) + self.rng.normal(0, 0.35)
        f.state, f.t_state = "takeoff", 0.0
        f.heading = ang
        f.flight_ms = self.rng.uniform(*FLIGHT_MS)
        reach = JUMP_SPEED * f.flight_ms * 1e-3 / 2
        p1 = (f.x + reach * math.cos(ang), f.y + reach * math.sin(ang))
        for _ in range(20):
            land = (self.rng.uniform(40, TABLE_W - 40), self.rng.uniform(30, TABLE_H - 30))
            if not self.under_swatter(*land, margin=30):
                break
        f.path = ((f.x, f.y), p1, land)
        self.event(f"giant fiber {side} spiked - take-off")

    def _move_fly(self, ms, drop):
        f = self.fly
        f.t_state += ms
        dt = ms * 1e-3
        if f.state == "walk":
            target = self._nearest_sugar()
            f.heading += self.rng.normal(0, TURN_NOISE * math.sqrt(dt))
            if target is not None:
                want = math.atan2(target.y - f.y, target.x - f.x)
                err = (want - f.heading + math.pi) % (2 * math.pi) - math.pi
                f.heading += 3.0 * err * dt
            f.x += WALK_SPEED * math.cos(f.heading) * dt
            f.y += WALK_SPEED * math.sin(f.heading) * dt
            self._keep_on_table(f)
            if drop is not None and self.mn9_hz > FEED_ON_HZ:
                f.state, f.t_state = "feed", 0.0
                self.event("MN9 fires - the fly stops to drink")
        elif f.state == "feed":
            if drop is not None:
                drop.volume = max(0.0, drop.volume - DRINK_RATE * dt * (self.mn9_hz > 5))
            if drop is None or drop.volume <= 0 or (self.mn9_hz < 5 and f.t_state > 250):
                f.state, f.t_state = "walk", 0.0
                f.heading += math.pi * self.rng.uniform(0.6, 1.4)
                self.event("MN9 quiet - the fly walks on")
        elif f.state == "takeoff":
            if f.t_state >= TAKEOFF_DELAY:
                f.state, f.t_state = "air", 0.0
                rec = self.swings[-1] if self.swings else None
                if rec and rec.result is None:
                    rec.takeoff = self.t - rec.t0
        elif f.state == "air":
            k = min(f.t_state / f.flight_ms, 1.0)
            (x0, y0), (x1, y1), (x2, y2) = f.path
            f.x = (1 - k) ** 2 * x0 + 2 * (1 - k) * k * x1 + k * k * x2
            f.y = (1 - k) ** 2 * y0 + 2 * (1 - k) * k * y1 + k * k * y2
            f.vx = 2 * (1 - k) * (x1 - x0) + 2 * k * (x2 - x1)
            f.vy = 2 * (1 - k) * (y1 - y0) + 2 * k * (y2 - y1)
            f.heading = math.atan2(f.vy, f.vx)
            f.z = 60.0 * math.sin(math.pi * k)
            if k >= 1.0:
                f.state, f.t_state, f.z = "walk", 0.0, 0.0
                self._theta = None
        f.proboscis += (float(self.mn9_hz > 5 and f.state in ("walk", "feed")) - f.proboscis) * min(1.0, dt * 12)

    def _keep_on_table(self, f):
        margin = 8.0
        if not margin <= f.x <= TABLE_W - margin:
            f.heading = math.pi - f.heading
            f.x = min(max(f.x, margin), TABLE_W - margin)
        if not margin <= f.y <= TABLE_H - margin:
            f.heading = -f.heading
            f.y = min(max(f.y, margin), TABLE_H - margin)

    def _nearest_sugar(self):
        f = self.fly
        best, bd = None, SMELL_RANGE
        for d in self.drops:
            if d.kind == "sugar" and d.volume > 0:
                dist = math.hypot(d.x - f.x, d.y - f.y)
                if dist < bd:
                    best, bd = d, dist
        return best

    def step(self, ms=1.0):
        """Advance the world and the brain by `ms` model milliseconds."""
        self._update_swatter(ms)
        self._looming(ms)
        drop = self._senses()
        spikes = self.brain.run(ms, record=True)
        fired = spikes[:, 1] if len(spikes) else np.empty(0, np.int64)
        self.fired = fired
        mn9 = np.isin(self.mn9_idx, fired).sum() / len(self.mn9_idx) / (ms * 1e-3)
        self.mn9_hz += (1 - math.exp(-ms / 60.0)) * (mn9 - self.mn9_hz)
        for side, idx in self.gf_idx.items():
            self.gf_flash[side] *= math.exp(-ms / 120.0)
            if np.isin(idx, fired).any():
                self.gf_flash[side] = 1.0
                rec = self.swings[-1] if self.swings else None
                if rec and rec.result is None and side not in rec.gf:
                    rec.gf[side] = self.t + ms - rec.t0
                if self.fly.state in ("walk", "feed") and not self.gf_ablated:
                    self._takeoff(side)
        self._move_fly(ms, drop)
        if self.fly.state == "dead" and self.fly.t_state > 1500:
            self.new_fly()
        self.t += ms
        return fired
