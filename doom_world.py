"""
The fly brain plays DOOM (ViZDoom, "Defend the Center"): the player stands in
the middle of a round room while demons and chainsaw marines walk in from all
sides. The only buttons are turn left, turn right and fire.

Eye (hand-made, like the retina the model lacks): every enemy has a bearing
and an angular size on the fly's panoramic eye (330 degrees, blind straight
behind). Two channels of visual projection neurons, both found in the
connectome to steer in opposite directions:

  small, distant enemies  -> LC10a/c/d/e  -> DNa01/DNa02 on the same side:
                                             the fly turns TOWARDS them
  large, near enemies     -> LC4 + LPLC2  -> DNa01/DNa02 on the other side
                                             and the giant fibers: it turns
                                             AWAY and "jumps"

Motor mapping: turn left/right when the left and right DNa01+DNa02 rates
differ; fire in every game tic in which a giant fiber spikes. The connectome
steers more strongly to the left (with the same input on both sides the left
DNa fire about twice as fast); this bias is left in, and the fly turns left
about three times as often as right.

Learning context (used with fly_learning): the same enemies also drive the
visual projection neurons that feed the visual Kenyon cells of the mushroom
body (distant enemies aMe12/aMe20, near ones MTe30/32/40 and LTe25, per side),
MBONs get a tonic 6.5 mV depolarisation, and every kill activates the reward
dopamine neurons (PAM) and every hit taken the punishment ones (PPL1) for
300 ms. These are on in every condition, with or without plasticity.

Bearings and distances are read from the game engine (object positions), not
from the pixels. One game tic (1/35 s) is 28.6 ms of brain time.
"""
import math
import os

import numpy as np
import vizdoom as vzd

from fly_brain import FlyBrain
from fly_circuits import Circuits
from fly_learning import LEARNERS, dan_groups, make_dopamine_modulatory

TIC_MS = 1000.0 / 35.0
ENEMIES = {"Demon": 30.0, "MarineChainsawVzd": 16.0}
BLIND = 165.0
BINOCULAR = 15.0

SMALL_TYPES = ["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"]
SMALL_RATE = 110.0
SMALL_FADE = (0.15, 0.35)
LOOM_SIZE = (0.12, 0.45)
LOOM_SIZE_RATE = 220.0
LOOM_VEL_GAIN = 40.0
LOOM_MAX = 300.0

VPN_FAR = ["aMe12", "aMe20"]
VPN_NEAR = ["MTe30", "MTe32", "MTe40", "LTe25"]
VPN_RATE = 120.0
MBON_BIAS = 6.5
DAN_RATE = 150.0
DAN_MS = 300.0

STEER_TAU = 100.0
STEER_DEADBAND = 8.0


def smooth(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Enemy:
    def __init__(self, oid, name, bearing, dist, theta, dtheta):
        self.id, self.name = oid, name
        self.bearing, self.dist = bearing, dist
        self.theta, self.dtheta = theta, dtheta
        self.small = 0.0
        self.loom = 0.0
        self.seen = abs(bearing) < BLIND


class DoomWorld:
    def __init__(self, brain=None, seed=None, resolution="RES_640X480",
                 policy="connectome", learner=None):
        self.rng = np.random.default_rng(seed)
        self.brain = brain or FlyBrain(seed=seed)
        self.brain.run(0.2)
        self.circ = Circuits(self.brain)
        b = self.brain
        self.small_idx = {s: b.neurons(cell_type=SMALL_TYPES, side=s) for s in ("left", "right")}
        self.loom_idx = self.circ.sense
        self.steer_idx = {s: np.concatenate([self.circ.out[f"dna01_{s[0].upper()}"],
                                             self.circ.out[f"dna02_{s[0].upper()}"]])
                          for s in ("left", "right")}
        self.gf_idx = np.concatenate([self.circ.out["gf_L"], self.circ.out["gf_R"]])
        self.vpn = {(kind, s): b.neurons(cell_type=types, side=s)
                    for kind, types in (("far", VPN_FAR), ("near", VPN_NEAR))
                    for s in ("left", "right")}
        self.pam, self.ppl1 = dan_groups(b)
        make_dopamine_modulatory(b)
        b.bias[np.flatnonzero(b.annot["cell_class"].to_numpy() == "MBON")] = MBON_BIAS
        if isinstance(learner, str):
            learner = LEARNERS[learner](b)
        self.learner = learner
        self.policy = policy
        self.gf_cut = False

        g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
        g.set_window_visible(False)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_screen_resolution(getattr(vzd.ScreenResolution, resolution))
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_render_hud(True)
        g.set_objects_info_enabled(True)
        g.set_labels_buffer_enabled(True)
        g.set_available_game_variables([
            vzd.GameVariable.AMMO2, vzd.GameVariable.HEALTH, vzd.GameVariable.KILLCOUNT,
            vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE])
        if seed is not None:
            g.set_seed(int(seed) % (2 ** 31))
        g.init()
        self.game = g
        self.reset()

    def reset(self):
        self.game.new_episode()
        self.brain.reset()
        self.tic = 0
        self.prev_theta = {}
        self.enemies = []
        self.steer = {"left": 0.0, "right": 0.0}
        self.drive = {"small_left": 0.0, "small_right": 0.0, "loom_left": 0.0, "loom_right": 0.0}
        self.action = [0, 0, 0]
        self.dan_timer = {"reward": 0.0, "punish": 0.0}
        self.rewards = 0
        self.punishments = 0
        self.gf_spikes = 0
        self.shots = 0
        self.events = []
        self.fired = np.empty(0, dtype=np.int64)
        self.screen = None
        self.labels = []
        self.vars = {"ammo": 26, "health": 100, "kills": 0}
        self._observe()

    def set_gf_cut(self, on):
        self.gf_cut = on

    def event(self, text):
        self.events.append((self.tic, text))
        del self.events[:-8]

    @property
    def done(self):
        return self.game.is_episode_finished()

    def _observe(self):
        s = self.game.get_state()
        if s is None:
            return
        self.screen = s.screen_buffer
        self.labels = [l for l in s.labels if l.object_name in ENEMIES]
        ammo, health, kills, px, py, ang = s.game_variables
        self.vars = {"ammo": int(ammo), "health": int(health), "kills": int(kills)}
        enemies, thetas = [], {}
        for o in s.objects:
            if o.name not in ENEMIES:
                continue
            dx, dy = o.position_x - px, o.position_y - py
            dist = max(math.hypot(dx, dy), 1.0)
            bearing = (math.degrees(math.atan2(dy, dx)) - ang + 180) % 360 - 180
            theta = 2 * math.atan(ENEMIES[o.name] / dist)
            prev = self.prev_theta.get(o.id)
            dtheta = (theta - prev) / (TIC_MS * 1e-3) if prev is not None else 0.0
            thetas[o.id] = theta
            enemies.append(Enemy(o.id, o.name, bearing, dist, theta, dtheta))
        self.prev_theta = thetas
        self.enemies = enemies
        self._eye()

    def _eye(self):
        d = {"small_left": 0.0, "small_right": 0.0, "loom_left": 0.0, "loom_right": 0.0}
        for e in self.enemies:
            if not e.seen:
                continue
            w_left = min(max(0.5 + e.bearing / (2 * BINOCULAR), 0.0), 1.0)
            e.small = SMALL_RATE * (1 - smooth(*SMALL_FADE, e.theta))
            e.loom = min(LOOM_MAX, LOOM_SIZE_RATE * smooth(*LOOM_SIZE, e.theta)
                         + LOOM_VEL_GAIN * max(e.dtheta, 0.0))
            d["small_left"] += e.small * w_left
            d["small_right"] += e.small * (1 - w_left)
            d["loom_left"] += e.loom * w_left
            d["loom_right"] += e.loom * (1 - w_left)
        self.drive = {k: min(v, LOOM_MAX) for k, v in d.items()}
        inp = []
        for s in ("left", "right"):
            inp.append((self.small_idx[s], self.drive[f"small_{s}"]))
            inp.append((self.loom_idx["loom", s], self.drive[f"loom_{s}"]))
            inp.append((self.vpn["far", s], VPN_RATE * min(self.drive[f"small_{s}"] / SMALL_RATE, 1.0)))
            inp.append((self.vpn["near", s], VPN_RATE * min(self.drive[f"loom_{s}"] / LOOM_SIZE_RATE, 1.0)))
        if self.dan_timer["reward"] > 0:
            inp.append((self.pam, DAN_RATE))
        if self.dan_timer["punish"] > 0:
            inp.append((self.ppl1, DAN_RATE))
        self.brain.set_input([(i, hz) for i, hz in inp if hz > 0])

    def _brain_tic(self):
        spikes = self.brain.run(TIC_MS, record=True)
        fired = spikes[:, 1] if len(spikes) else np.empty(0, np.int64)
        self.fired = fired
        if self.learner is not None:
            self.learner.update(np.bincount(fired, minlength=self.brain.n), TIC_MS)
        for key in self.dan_timer:
            self.dan_timer[key] = max(0.0, self.dan_timer[key] - TIC_MS)
        k = 1 - math.exp(-TIC_MS / STEER_TAU)
        for s in ("left", "right"):
            n = np.isin(fired, self.steer_idx[s]).sum()
            hz = n / len(self.steer_idx[s]) / (TIC_MS * 1e-3)
            self.steer[s] += k * (hz - self.steer[s])
        gf = int(np.isin(fired, self.gf_idx).sum())
        self.gf_spikes = gf
        diff = self.steer["left"] - self.steer["right"]
        turn_left = diff > STEER_DEADBAND
        turn_right = diff < -STEER_DEADBAND
        fire = gf > 0 and not self.gf_cut
        return [int(turn_left), int(turn_right), int(fire)]

    def _random_action(self):
        return [int(self.rng.random() < 0.5), int(self.rng.random() < 0.5),
                int(self.rng.random() < 0.1)]

    def step(self):
        """One game tic: brain (or baseline policy) decides, the game advances."""
        if self.done:
            return
        if self.policy == "connectome":
            action = self._brain_tic()
        elif self.policy == "random":
            action = self._random_action()
        else:
            action = [0, 0, 0]
        self.action = action
        kills = self.vars["kills"]
        health = self.vars["health"]
        if action[2]:
            self.shots += 1
        self.game.make_action(action)
        self.tic += 1
        if self.done:
            self.event("the fly is dead" if self.game.is_player_dead() else "time is up")
            return
        self._observe()
        if self.vars["kills"] > kills:
            self.event(f"kill #{self.vars['kills']}  ->  reward (PAM)")
            self.dan_timer["reward"] = DAN_MS
            self.rewards += 1
        if self.vars["health"] < health:
            self.event(f"hit, health {self.vars['health']}  ->  punishment (PPL1)")
            self.dan_timer["punish"] = DAN_MS
            self.punishments += 1

    def summary(self):
        return {"kills": self.vars["kills"], "tics": self.tic, "shots": self.shots,
                "health": self.vars["health"]}
