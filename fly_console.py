"""
Fly Brain Rig: a live console for the whole-brain Drosophila model.

The centre shows every one of the 138,639 neurons at its real position
(frontal view of the FlyWire brain). Neurons light up when they spike and fade
over ~150 ms, like a calcium-imaging movie. Around it: switchable senses,
the motor/descending outputs, the most active cell types, and a table of
stimulus -> first-output-spike latencies.

Mouse:
  click a sense's L / R chip    switch that side on/off
  click a sense's name          switch both sides
  wheel over a sense            change its rate (Hz)
  hover over the brain          name of the nearest active neuron
Keys:
  /        type a cell type (Enter: stimulate, Tab: switch to silence)
  SPACE    pause          1-4   speed 0.1x / 0.25x / 0.5x / 1x
  C        all senses off R     reset brain state
  E        export CSV     ESC   quit
"""
import csv
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import pygame

from brain_view import CLASS_COLORS, LEGEND, UNKNOWN_CLASS, BrainMap, Glow
from fly_brain import FlyBrain
from fly_circuits import MODALITIES, OUTPUTS, SENSES, SIDES, Circuits

HERE = os.path.dirname(os.path.abspath(__file__))

CANVAS = (1600, 960)
HEADER = pygame.Rect(0, 0, 1600, 60)
STIM = pygame.Rect(16, 76, 312, 556)
CUSTOM = pygame.Rect(16, 648, 312, 296)
BRAIN = pygame.Rect(344, 76, 848, 556)
TRACES = pygame.Rect(344, 648, 848, 296)
OUTS = pygame.Rect(1208, 76, 376, 290)
TYPES = pygame.Rect(1208, 382, 376, 250)
TRIALS = pygame.Rect(1208, 648, 376, 296)

BG = (9, 11, 15)
PANEL = (16, 19, 25)
BORDER = (38, 44, 54)
TEXT = (226, 230, 236)
MUTED = (132, 140, 152)
FAINT = (72, 79, 90)
OFF = (34, 39, 48)

MODALITY_COLORS = {
    "taste": (246, 180, 72),
    "vision": (110, 196, 255),
    "antenna": (176, 146, 255),
    "smell": (104, 222, 160),
    "custom": (230, 230, 230),
}
OUT_COLORS = {
    "mn9": (246, 180, 72),
    "gf_L": (255, 92, 92), "gf_R": (255, 160, 110),
    "dna01_L": (80, 205, 255), "dna01_R": (150, 230, 255),
    "dna02_L": (90, 130, 255), "dna02_R": (160, 180, 255),
    "mdn": (200, 150, 255),
}
SILENCE = (255, 80, 80)

SPEEDS = {pygame.K_1: 0.1, pygame.K_2: 0.25, pygame.K_3: 0.5, pygame.K_4: 1.0}
DEFAULT_RATE = 150
GLOW_TAU = 150.0
RATE_TAU = 60.0
TYPE_TAU = 250.0
TRACE_MS = 4000.0
TRIAL_MS = 1000.0
FRAME_BUDGET = 0.030


def font(size, mono=False, bold=False):
    name = "dejavusansmono" if mono else "dejavusans"
    return pygame.font.SysFont(name, size, bold=bold)


def mix(c1, c2, k):
    return tuple(int(a + (b - a) * k) for a, b in zip(c1, c2))


class Sense:
    """One switchable input row: a named group with a left and right side."""

    def __init__(self, key, modality, label, idx_by_side, custom=False):
        self.key, self.modality, self.label = key, modality, label
        self.idx = idx_by_side
        self.on = {s: False for s in self.idx}
        self.rate = DEFAULT_RATE
        self.custom = custom
        self.chips = {}
        self.row = None

    def active_sides(self):
        return [s for s, v in self.on.items() if v]


class Console:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Fly Brain Rig")
        self.screen = pygame.display.set_mode(CANVAS)
        self.clock = pygame.time.Clock()
        self.f_title = font(19, bold=True)
        self.f = font(13)
        self.f_small = font(11)
        self.f_mono = font(13, mono=True)
        self.f_mono_small = font(11, mono=True)
        self.f_big = font(20, mono=True)

        self._splash("loading the FlyWire connectome ...")
        t0 = time.time()
        self.brain = FlyBrain(seed=int(time.time()))
        self.brain.run(0.2)
        self.load_s = time.time() - t0
        self.circ = Circuits(self.brain)
        b = self.brain

        a = b.annot
        self.type_codes, self.type_names = pd.factorize(a["cell_type"].fillna("?"))
        self.type_size = np.bincount(self.type_codes)
        self.type_class = (pd.Series(a["super_class"].fillna("?").to_numpy())
                           .groupby(self.type_codes).agg(lambda c: c.mode()[0]).to_numpy())
        self.cls_names = a["super_class"].fillna("?").to_numpy()
        self.cell_type = a["cell_type"].fillna("?").to_numpy()
        self.side_short = a["side"].fillna("").str[:1].str.upper().to_numpy()
        self.known_types = sorted(set(self.type_names) - {"?"})

        self.senses = []
        for key, modality, label, _ in SENSES:
            idx = {s: self.circ.sense[key, s] for s in SIDES}
            self.senses.append(Sense(key, modality, label, idx))
        self.custom = []
        self.silenced_types = []
        self.entry = None
        self.entry_mode = "stimulate"

        self._build_brain_map()
        self.reset_state()

    def _splash(self, msg):
        self.screen.fill(BG)
        s = self.f_title.render(msg, True, MUTED)
        self.screen.blit(s, s.get_rect(center=(CANVAS[0] // 2, CANVAS[1] // 2)))
        pygame.display.flip()

    def _build_brain_map(self):
        inner = BRAIN.inflate(-32, -80)
        self.map = BrainMap(self.brain, inner.size, background=PANEL)
        self.map_origin = (inner.centerx - self.map.w // 2,
                           inner.top + 16 + (inner.h - self.map.h) // 2)
        self.out_pos = {k: [self.map.point(i, self.map_origin) for i in idx if self.map.ok[i]]
                        for k, idx in self.circ.out.items()}

    def reset_state(self):
        b = self.brain
        b.reset()
        self.apply_input()
        self.glow = Glow(b.n, GLOW_TAU)
        self.recent = np.zeros(b.n, dtype=np.float32)
        self.out_rate = {k: 0.0 for k in self.circ.out}
        self.history = deque()
        self.trials = []
        self.flash = {k: 0.0 for k in self.circ.out}
        self.paused = False
        self.speed = 0.5
        self.real_factor = 0.0
        self.active_count = 0
        self.spike_rate = 0.0
        self.message = "ready - switch on a sense on the left"
        self.hover = None

    def all_rows(self):
        return self.senses + self.custom

    def apply_input(self):
        drive = []
        for s in self.all_rows():
            for side in s.active_sides():
                drive.append((s.idx[side], s.rate))
        self.brain.set_input(drive)

    def toggle(self, sense, sides):
        turning_on = not all(sense.on[s] for s in sides)
        for s in sides:
            sense.on[s] = turning_on
        self.apply_input()
        if turning_on:
            side_txt = "" if list(sense.on) == ["all"] else (
                " L+R" if len(sides) == 2 else f" {sides[0][0].upper()}")
            self.start_trial(f"{sense.label.split('  ')[0]}{side_txt}", sense.modality)
        self.message = (f"{sense.label.split('  ')[0]} {'on' if turning_on else 'off'}"
                        f" at {sense.rate} Hz")

    def all_off(self):
        for s in self.all_rows():
            for k in s.on:
                s.on[k] = False
        self.apply_input()
        self.message = "all senses off"

    def add_custom(self, name, mode):
        idx = self.brain.neurons(cell_type=name)
        if not len(idx):
            self.message = f"no neurons of type '{name}'"
            return
        if mode == "silence":
            if name not in self.silenced_types:
                self.silenced_types.append(name)
                self.update_silenced()
            self.message = f"silenced {name} ({len(idx)} neurons)"
        else:
            row = Sense(f"custom:{name}", "custom", name, {"all": idx}, custom=True)
            self.custom.append(row)
            self.toggle(row, ["all"])

    def update_silenced(self):
        idx = [self.brain.neurons(cell_type=t) for t in self.silenced_types]
        self.brain.set_silenced(np.concatenate(idx) if idx else [])

    def start_trial(self, label, modality):
        """A trial collects the first spike of each output after the onset.
        Only the latest trial collects, and outputs that were already firing
        at the onset are not counted as responses."""
        busy = [k for k, r in self.out_rate.items() if r > 2.0]
        self.trials.append({"n": len(self.trials) + 1, "label": label,
                            "modality": modality, "t0": self.brain.t * self.brain.dt,
                            "first": {}, "busy": busy})

    def update_trials(self, spikes):
        if not self.trials or not len(spikes):
            return
        tr = self.trials[-1]
        if self.brain.t * self.brain.dt - tr["t0"] > TRIAL_MS + 50:
            return
        for key, idx in self.circ.out.items():
            if key in tr["first"] or key in tr["busy"]:
                continue
            hit = spikes[np.isin(spikes[:, 1], idx)]
            if len(hit):
                t = hit[0, 0] * self.brain.dt - tr["t0"]
                if 0 <= t <= TRIAL_MS:
                    tr["first"][key] = t

    def advance(self, wall_dt):
        """Run the brain for speed x wall_dt ms of model time, within budget."""
        b = self.brain
        target = min(self.speed * wall_dt * 1000.0, 60.0)
        t_start = time.perf_counter()
        before = b.spike_count.copy()
        chunks, sim = [], 0.0
        while sim < target and time.perf_counter() - t_start < FRAME_BUDGET:
            chunks.append(b.run(1.0, record=True))
            sim += 1.0
        spikes = np.concatenate(chunks) if chunks else np.empty((0, 2), np.int64)
        self.real_factor = 0.9 * self.real_factor + 0.1 * (sim / 1000.0 / max(wall_dt, 1e-3))
        if sim == 0:
            return
        delta = b.spike_count - before
        fired = np.flatnonzero(delta)

        self.glow.update(fired, sim)
        k = np.exp(-sim / TYPE_TAU)
        self.recent *= k
        self.recent[fired] += (1 - k) * delta[fired] / sim
        self.active_count = int((self.glow.level > 0.35).sum())
        self.spike_rate = 0.8 * self.spike_rate + 0.2 * delta.sum() / sim * 1000.0

        rates = self.circ.rates(delta, sim)
        a = 1 - np.exp(-sim / RATE_TAU)
        for key, r in rates.items():
            self.out_rate[key] += a * (r - self.out_rate[key])
            if r > 0:
                self.flash[key] = 1.0
        for key in self.flash:
            self.flash[key] *= np.exp(-sim / 80.0)

        now = b.t * b.dt
        active = tuple((s.modality, s.key) for s in self.all_rows() if s.active_sides())
        self.history.append((now, dict(self.out_rate), active))
        while self.history and now - self.history[0][0] > TRACE_MS:
            self.history.popleft()
        self.update_trials(spikes)

    def handle_event(self, ev):
        if ev.type == pygame.KEYDOWN and self.entry is not None:
            return self.handle_entry_key(ev)
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                return False
            if ev.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif ev.key in SPEEDS:
                self.speed = SPEEDS[ev.key]
                self.message = f"speed {self.speed}x model time"
            elif ev.key == pygame.K_c:
                self.all_off()
            elif ev.key == pygame.K_r:
                self.reset_state()
                self.message = "brain state reset"
            elif ev.key == pygame.K_e:
                self.export()
            elif ev.key == pygame.K_SLASH or ev.unicode == "/":
                self.entry, self.entry_mode = "", "stimulate"
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.on_click(ev.pos)
        elif ev.type == pygame.MOUSEWHEEL:
            pos = pygame.mouse.get_pos()
            for s in self.all_rows():
                if s.row and s.row.collidepoint(pos):
                    s.rate = int(np.clip(s.rate + 25 * ev.y, 25, 400))
                    self.apply_input()
                    self.message = f"{s.label.split('  ')[0]}: {s.rate} Hz"
        elif ev.type == pygame.MOUSEMOTION:
            self.hover = ev.pos if BRAIN.collidepoint(ev.pos) else None
        return True

    def handle_entry_key(self, ev):
        if ev.key == pygame.K_ESCAPE:
            self.entry = None
        elif ev.key == pygame.K_TAB:
            self.entry_mode = "silence" if self.entry_mode == "stimulate" else "stimulate"
        elif ev.key == pygame.K_RETURN:
            name = self.entry.strip()
            match = self.completion(name)
            if name:
                self.add_custom(name if name in self.known_types else (match or name),
                                self.entry_mode)
            self.entry = None
        elif ev.key == pygame.K_BACKSPACE:
            self.entry = self.entry[:-1]
        elif ev.unicode and ev.unicode.isprintable():
            self.entry += ev.unicode
        return True

    def completion(self, prefix):
        if not prefix:
            return None
        for t in self.known_types:
            if t.startswith(prefix):
                return t
        low = prefix.lower()
        for t in self.known_types:
            if t.lower().startswith(low):
                return t
        return None

    def on_click(self, pos):
        for s in self.all_rows():
            for side, rect in s.chips.items():
                if rect.collidepoint(pos):
                    self.toggle(s, [side])
                    return
            if s.row and s.row.collidepoint(pos):
                self.toggle(s, list(s.on))
                return
        for i, (rect, kind) in enumerate(getattr(self, "remove_buttons", [])):
            if rect.collidepoint(pos):
                if kind[0] == "custom":
                    row = kind[1]
                    self.custom.remove(row)
                    self.apply_input()
                else:
                    self.silenced_types.remove(kind[1])
                    self.update_silenced()
                return
        if getattr(self, "entry_rect", None) and self.entry_rect.collidepoint(pos):
            self.entry, self.entry_mode = "", "stimulate"

    def export(self):
        out = os.path.join(HERE, "recordings")
        os.makedirs(out, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        keys = list(self.circ.out)
        with open(os.path.join(out, f"rig_{stamp}_timeseries.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_ms"] + [f"{k}_hz" for k in keys] + ["active_senses"])
            for t, r, act in self.history:
                w.writerow([f"{t:.1f}"] + [f"{r[k]:.2f}" for k in keys]
                           + [" ".join(k for _, k in act)])
        with open(os.path.join(out, f"rig_{stamp}_trials.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "stimulus", "onset_ms"] + [f"{k}_latency_ms" for k in keys])
            for tr in self.trials:
                w.writerow([tr["n"], tr["label"], f"{tr['t0']:.1f}"]
                           + [f"{tr['first'][k]:.1f}" if k in tr["first"] else "" for k in keys])
        self.message = f"exported recordings/rig_{stamp}_*.csv"

    def text(self, txt, pos, color=TEXT, f=None, anchor="topleft"):
        s = (f or self.f).render(str(txt), True, color)
        r = s.get_rect(**{anchor: pos})
        self.screen.blit(s, r)
        return r

    def panel(self, rect, title, subtitle=""):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=6)
        pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=6)
        r = self.text(title, (rect.x + 14, rect.y + 10), TEXT, self.f_mono)
        if subtitle:
            self.text(subtitle, (r.right + 10, rect.y + 11), MUTED, self.f_small)

    def draw(self):
        self.screen.fill(BG)
        self.draw_header()
        self.draw_senses()
        self.draw_custom()
        self.draw_brain()
        self.draw_traces()
        self.draw_outputs()
        self.draw_types()
        self.draw_trials()
        pygame.display.flip()

    def draw_header(self):
        b = self.brain
        self.text("FLY BRAIN RIG", (20, 10), TEXT, self.f_title)
        self.text(f"FlyWire v783 whole-brain connectome  ·  {b.n:,} neurons  ·  "
                  f"{b.n_synapses / 1e6:.1f} M synapses  ·  leaky integrate-and-fire "
                  f"(Shiu et al. 2024)", (20, 36), MUTED, self.f_small)
        stats = [
            ("MODEL TIME", f"{b.t * b.dt / 1000:7.2f} s"),
            ("SPEED", f"{self.speed:g}x  ({self.real_factor:.2f}x real)"),
            ("ACTIVE", f"{self.active_count:5d}"),
            ("SPIKES / s", f"{self.spike_rate:8.0f}"),
        ]
        x = CANVAS[0] - 20
        for label, val in reversed(stats):
            w = max(self.f_big.size(val)[0], self.f_small.size(label)[0])
            self.text(label, (x - w, 10), MUTED, self.f_small)
            self.text(val, (x - w, 26), TEXT, self.f_big)
            x -= w + 34
        state = "PAUSED" if self.paused else "RUNNING"
        col = (255, 190, 80) if self.paused else (110, 230, 150)
        pygame.draw.circle(self.screen, col, (x - 100, 32), 5)
        self.text(state, (x - 88, 24), col, self.f_mono)

    def draw_senses(self):
        self.panel(STIM, "SENSES", "click L / R · wheel = rate")
        y = STIM.y + 40
        for modality in MODALITIES:
            color = MODALITY_COLORS[modality]
            self.text(modality.upper(), (STIM.x + 14, y), color, self.f_small)
            y += 18
            for s in [s for s in self.senses if s.modality == modality]:
                self.draw_sense_row(s, y, color)
                y += 40
            y += 8

    def draw_sense_row(self, s, y, color, x=None, width=None):
        x = STIM.x + 10 if x is None else x
        width = STIM.w - 20 if width is None else width
        s.row = pygame.Rect(x, y, width, 34)
        any_on = bool(s.active_sides())
        pygame.draw.rect(self.screen, mix(PANEL, color, 0.12 if any_on else 0.0),
                         s.row, border_radius=4)
        name, _, detail = s.label.partition("  ")
        self.text(name, (x + 8, y + 3), TEXT if any_on else mix(TEXT, PANEL, 0.25), self.f)
        sub = f"{detail}  ·  {s.rate} Hz" if detail else f"{s.rate} Hz"
        n = sum(len(v) for v in s.idx.values())
        self.text(f"{sub}  ·  {n} n", (x + 8, y + 19), MUTED, self.f_small)
        s.chips = {}
        cx = x + width - 8
        for side in reversed(list(s.idx)):
            label = "ON" if side == "all" else side[0].upper()
            r = pygame.Rect(0, 0, 30 if side == "all" else 24, 22)
            r.midright = (cx, y + 17)
            on = s.on[side]
            pygame.draw.rect(self.screen, color if on else OFF, r, border_radius=4)
            self.text(label, r.center, BG if on else MUTED, self.f_mono_small, "center")
            s.chips[side] = r
            cx = r.left - 5

    def draw_custom(self):
        self.panel(CUSTOM, "CELL TYPES", "/ to type · Tab = silence")
        y = CUSTOM.y + 38
        box = pygame.Rect(CUSTOM.x + 10, y, CUSTOM.w - 20, 28)
        self.entry_rect = box
        editing = self.entry is not None
        mode_col = SILENCE if self.entry_mode == "silence" else MODALITY_COLORS["custom"]
        pygame.draw.rect(self.screen, OFF, box, border_radius=4)
        pygame.draw.rect(self.screen, mode_col if editing else BORDER, box, 1, border_radius=4)
        if editing:
            comp = self.completion(self.entry)
            shown = self.entry + ("_" if int(time.time() * 2) % 2 else " ")
            r = self.text(shown, (box.x + 8, box.y + 6), TEXT, self.f_mono)
            if comp and comp != self.entry and comp.startswith(self.entry):
                self.text(comp[len(self.entry):], (r.right - 8, box.y + 6), FAINT, self.f_mono)
            self.text("SILENCE" if self.entry_mode == "silence" else "STIMULATE",
                      (box.right - 8, box.y + 8), mode_col, self.f_small, "topright")
        else:
            self.text("e.g. DNp01, LC6, MBON01, Kenyon_Cell ...", (box.x + 8, box.y + 7),
                      FAINT, self.f_small)
        y = box.bottom + 10
        self.remove_buttons = []
        for row in self.custom[-4:]:
            self.draw_sense_row(row, y, MODALITY_COLORS["custom"],
                                x=CUSTOM.x + 10, width=CUSTOM.w - 44)
            xr = pygame.Rect(CUSTOM.right - 30, y + 7, 20, 20)
            self.text("×", xr.center, MUTED, self.f, "center")
            self.remove_buttons.append((xr, ("custom", row)))
            y += 40
        for name in self.silenced_types[-3:]:
            r = pygame.Rect(CUSTOM.x + 10, y, CUSTOM.w - 44, 26)
            pygame.draw.rect(self.screen, mix(PANEL, SILENCE, 0.15), r, border_radius=4)
            n = len(self.brain.neurons(cell_type=name))
            self.text(f"silenced  {name}  ·  {n} n", (r.x + 8, r.y + 5), SILENCE, self.f)
            xr = pygame.Rect(CUSTOM.right - 30, y + 3, 20, 20)
            self.text("×", xr.center, MUTED, self.f, "center")
            self.remove_buttons.append((xr, ("silence", name)))
            y += 32

    def draw_brain(self):
        self.panel(BRAIN, "BRAIN", "frontal view · every neuron at its soma · "
                   "light = spiked in the last ~150 ms")
        ox, oy = self.map_origin
        self.screen.blit(self.map.render(self.glow.level), (ox, oy))

        lx, rx = (ox + 10, ox + self.map.w - 10)
        left_txt, right_txt = ("fly's LEFT", "fly's RIGHT") if self.map.left_on_screen_left \
            else ("fly's RIGHT", "fly's LEFT")
        self.text(left_txt, (lx, oy - 18), FAINT, self.f_small)
        self.text(right_txt, (rx, oy - 18), FAINT, self.f_small, "topright")
        bar = 100 / self.map.um_per_px
        by = oy + self.map.h + 14
        pygame.draw.line(self.screen, MUTED, (ox, by), (ox + bar, by), 2)
        self.text("100 µm", (ox + bar + 8, by - 7), MUTED, self.f_small)

        for key, pts in self.out_pos.items():
            col = OUT_COLORS[key]
            k = self.flash[key]
            for (x, y) in pts:
                pygame.draw.circle(self.screen, mix(FAINT, col, 0.4 + 0.6 * k), (x, y),
                                   5 + int(4 * k), 1)
        lx = BRAIN.x + 14
        ly = BRAIN.bottom - 22
        for cls in LEGEND:
            pygame.draw.circle(self.screen, CLASS_COLORS[cls], (lx + 4, ly + 7), 4)
            r = self.text(cls.replace("_", " "), (lx + 12, ly), MUTED, self.f_small)
            lx = r.right + 16
        self.text("rings = output neurons", (BRAIN.right - 14, ly), MUTED, self.f_small, "topright")

        if self.hover:
            self.draw_hover()

    def draw_hover(self):
        mx, my = self.hover
        cand = np.flatnonzero(self.glow.level > 0.05)
        i = self.map.nearest(self.hover, self.map_origin, cand)
        if i is None:
            return
        pygame.draw.circle(self.screen, TEXT, self.map.point(i, self.map_origin), 7, 1)
        label = (f"{self.cell_type[i]} {self.side_short[i]}  ·  "
                 f"{self.cls_names[i].replace('_', ' ')}  ·  {self.recent[i] * 1000:.0f} Hz")
        s = self.f.render(label, True, TEXT)
        r = s.get_rect(bottomleft=(mx + 14, my - 8))
        r.clamp_ip(BRAIN)
        pygame.draw.rect(self.screen, BG, r.inflate(12, 8), border_radius=4)
        self.screen.blit(s, r)

    def draw_traces(self):
        self.panel(TRACES, "OUTPUTS OVER TIME", f"last {TRACE_MS / 1000:.0f} s model time · Hz")
        strips = [("proboscis  MN9", ["mn9"], 200),
                  ("escape  giant fiber L / R", ["gf_L", "gf_R"], 250),
                  ("steering  DNa01 / DNa02  L / R",
                   ["dna01_L", "dna01_R", "dna02_L", "dna02_R"], 120),
                  ("backward  MDN", ["mdn"], 100)]
        plot = pygame.Rect(TRACES.x + 56, TRACES.y + 40, TRACES.w - 76, TRACES.h - 76)
        h = plot.h // len(strips)
        now = self.brain.t * self.brain.dt
        hist = list(self.history)
        for n, (title, keys, full) in enumerate(strips):
            r = pygame.Rect(plot.x, plot.y + n * h, plot.w, h - 8)
            pygame.draw.line(self.screen, BORDER, r.bottomleft, r.bottomright)
            self.text(title, (r.x + 6, r.y), MUTED, self.f_small)
            self.text(str(full), (r.x - 6, r.y), FAINT, self.f_small, "topright")
            self.text("0", (r.x - 6, r.bottom - 12), FAINT, self.f_small, "topright")
            for key in keys:
                pts = [(r.right - (now - t) / TRACE_MS * r.w,
                        r.bottom - min(rates[key] / full, 1.0) * (r.h - 14))
                       for t, rates, _ in hist]
                if len(pts) > 1:
                    pygame.draw.lines(self.screen, OUT_COLORS[key], False, pts, 2)
        by = plot.bottom + 4
        for t, _, act in hist:
            x = plot.right - (now - t) / TRACE_MS * plot.w
            for j, (modality, _) in enumerate(act[:4]):
                pygame.draw.line(self.screen, MODALITY_COLORS[modality],
                                 (x, by + j * 4), (x + 2, by + j * 4), 3)
        for tr in self.trials:
            x = plot.right - (now - tr["t0"]) / TRACE_MS * plot.w
            if x >= plot.x:
                pygame.draw.line(self.screen, MODALITY_COLORS[tr["modality"]],
                                 (x, plot.y), (x, plot.bottom), 1)
        for s in range(5):
            x = plot.right - s * 1000 / TRACE_MS * plot.w
            self.text(f"-{s}s" if s else "now", (x, TRACES.bottom - 18), FAINT,
                      self.f_small, "midtop")

    def draw_outputs(self):
        self.panel(OUTS, "OUTPUT NEURONS", "motor & descending · Hz")
        y = OUTS.y + 40
        for key, label, _, _, role in OUTPUTS:
            col = OUT_COLORS[key]
            k = self.flash[key]
            pygame.draw.circle(self.screen, mix(OFF, col, k), (OUTS.x + 20, y + 10), 5)
            self.text(label, (OUTS.x + 32, y + 1), TEXT, self.f)
            self.text(role, (OUTS.x + 32, y + 17), MUTED, self.f_small)
            bar = pygame.Rect(OUTS.x + 176, y + 6, 136, 10)
            pygame.draw.rect(self.screen, OFF, bar, border_radius=3)
            frac = min(self.out_rate[key] / 250.0, 1.0)
            if frac > 0.002:
                pygame.draw.rect(self.screen, col, (bar.x, bar.y, max(2, bar.w * frac), bar.h),
                                 border_radius=3)
            self.text(f"{self.out_rate[key]:4.0f}", (OUTS.right - 14, y + 2), TEXT,
                      self.f_mono, "topright")
            y += 30

    def draw_types(self):
        self.panel(TYPES, "MOST ACTIVE CELL TYPES", f"last ~{TYPE_TAU:.0f} ms · mean Hz")
        rate = np.bincount(self.type_codes, weights=self.recent,
                           minlength=len(self.type_names)) / self.type_size * 1000.0
        driven = set()
        for s in self.all_rows():
            for side in s.active_sides():
                driven.update(np.unique(self.type_codes[s.idx[side]]).tolist())
        order = [i for i in np.argsort(rate)[::-1] if rate[i] > 0.5][:9]
        y = TYPES.y + 38
        if not order:
            self.text("silent - no neuron is spiking", (TYPES.x + 14, y), FAINT, self.f)
        for i in order:
            cls = self.type_class[i]
            col = CLASS_COLORS.get(cls, UNKNOWN_CLASS)
            pygame.draw.rect(self.screen, col, (TYPES.x + 14, y + 4, 8, 8), border_radius=2)
            name = self.type_names[i]
            self.text(name[:22], (TYPES.x + 30, y), TEXT, self.f)
            tag = "input" if i in driven else cls.replace("_", " ")
            self.text(f"{tag} · {self.type_size[i]} n", (TYPES.x + 196, y + 2), MUTED, self.f_small)
            self.text(f"{rate[i]:4.0f}", (TYPES.right - 14, y), TEXT, self.f_mono, "topright")
            y += 22

    def draw_trials(self):
        self.panel(TRIALS, "STIMULUS → FIRST OUTPUT SPIKE", "latency in ms")
        y = TRIALS.y + 38
        if not self.trials:
            self.text("switch on a sense to start a trial", (TRIALS.x + 14, y), FAINT, self.f)
        for tr in reversed(self.trials[-6:]):
            col = MODALITY_COLORS[tr["modality"]]
            self.text(f"#{tr['n']}", (TRIALS.x + 14, y), FAINT, self.f_mono_small)
            self.text(tr["label"], (TRIALS.x + 44, y - 1), col, self.f)
            self.text(f"t = {tr['t0'] / 1000:.2f} s", (TRIALS.right - 14, y), FAINT,
                      self.f_small, "topright")
            firsts = sorted(tr["first"].items(), key=lambda kv: kv[1])
            txt = "   ".join(f"{self.circ.out_label[k]} {v:.0f}" for k, v in firsts[:4])
            now = self.brain.t * self.brain.dt
            if tr["busy"]:
                busy = ", ".join(self.circ.out_label[k] for k in tr["busy"])
                txt = (txt + "   " if txt else "") + f"(already firing: {busy})"
            if not txt:
                txt = "waiting ..." if now - tr["t0"] < TRIAL_MS else "no output spike"
            self.text(txt, (TRIALS.x + 44, y + 17), TEXT if firsts else MUTED, self.f_small)
            y += 42
        pygame.draw.line(self.screen, BORDER, (TRIALS.x + 10, TRIALS.bottom - 30),
                         (TRIALS.right - 10, TRIALS.bottom - 30))
        self.text(self.message, (TRIALS.x + 14, TRIALS.bottom - 23), MUTED, self.f_small)

    def run(self):
        running = True
        wall_dt = 1 / 30
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif not self.handle_event(ev):
                    running = False
            if not self.paused:
                self.advance(wall_dt)
            self.draw()
            wall_dt = self.clock.tick(60) / 1000.0
        pygame.quit()


if __name__ == "__main__":
    Console().run()
