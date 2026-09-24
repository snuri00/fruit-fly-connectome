"""
Top-down fruit fly and fly swatter sprites, rendered once at start-up with
numpy. Every part is a signed-distance shape, drawn 2x supersampled, shaded as
a lit ellipsoid and composited with premultiplied alpha.

The fly faces +x. Coordinates are pixels of a 320 x 320 canvas centred on the
thorax; the fly is about 250 px long with its wings folded. Layers (all the same size, so they can be
rotated and stacked at run time):

    legs[i]        8 frames of a tripod gait
    legs_air       legs tucked for flight
    body           abdomen, thorax, head, eyes, antennae
    proboscis      extended mouthparts
    wings_rest     folded wings
    wings_flight   2 frames of beating, motion-blurred wings
    dead, splat    a swatted fly
    shadow         soft shadow of the standing fly
"""
import math

import numpy as np
import pygame

SIZE = 320
SS = 2
GAIT_FRAMES = 8
_L = np.array([-0.45, -0.55, 0.70])
LIGHT = _L / np.linalg.norm(_L)


class Canvas:
    def __init__(self, size=SIZE, ss=SS):
        self.size, self.ss = size, ss
        self.n = size * ss
        c = (np.arange(self.n) + 0.5) / ss - size / 2
        self.x, self.y = np.meshgrid(c, c, indexing="ij")
        self.p = np.zeros((self.n, self.n, 3), np.float32)
        self.a = np.zeros((self.n, self.n), np.float32)

    def region(self, x0, x1, y0, y1):
        def idx(v):
            return int(np.clip((v + self.size / 2) * self.ss, 0, self.n))
        return (slice(idx(x0), idx(x1) + 1), slice(idx(y0), idx(y1) + 1))

    def cover(self, d):
        return np.clip(0.5 - d * self.ss, 0.0, 1.0)

    def paint(self, sl, rgb, alpha):
        rgb = np.asarray(rgb, np.float32)
        if rgb.ndim == 1:
            rgb = np.broadcast_to(rgb, alpha.shape + (3,))
        al = alpha[..., None]
        self.p[sl] = rgb * al + self.p[sl] * (1 - al)
        self.a[sl] = alpha + self.a[sl] * (1 - alpha)

    def arrays(self):
        s, m = self.ss, self.size
        p = self.p.reshape(m, s, m, s, 3).mean((1, 3))
        a = self.a.reshape(m, s, m, s).mean((1, 3))
        rgb = np.where(a[..., None] > 1e-4, p / np.maximum(a[..., None], 1e-4), 0.0)
        return rgb, a

    def surface(self, out=None):
        rgb, a = self.arrays()
        return to_surface(rgb, a, out)


def to_surface(rgb, a, out=None):
    surf = pygame.Surface(a.shape, pygame.SRCALPHA)
    px = pygame.surfarray.pixels3d(surf)
    px[:] = np.clip(rgb, 0, 255).astype(np.uint8)
    del px
    pa = pygame.surfarray.pixels_alpha(surf)
    pa[:] = np.clip(a * 255, 0, 255).astype(np.uint8)
    del pa
    if out:
        surf = pygame.transform.smoothscale(surf, (out, out))
    return surf


def ellipse(cv, cx, cy, rx, ry, rot=0.0, pad=2.0):
    """Returns (slice, distance, u, v, r) for an ellipse; u, v are the
    normalised coordinates in the ellipse frame, r = |(u, v)|."""
    ext = max(rx, ry) + pad
    sl = cv.region(cx - ext, cx + ext, cy - ext, cy + ext)
    c, s = math.cos(rot), math.sin(rot)
    dx, dy = cv.x[sl] - cx, cv.y[sl] - cy
    u = (dx * c + dy * s) / rx
    v = (-dx * s + dy * c) / ry
    r = np.sqrt(u * u + v * v)
    return sl, (r - 1) * min(rx, ry), u, v, r


def capsule(cv, p0, p1, r0, r1=None, pad=2.0):
    """Returns (slice, distance, h) for a tapered segment; h runs 0..1 along it."""
    r1 = r0 if r1 is None else r1
    (ax, ay), (bx, by) = p0, p1
    ext = max(r0, r1) + pad
    sl = cv.region(min(ax, bx) - ext, max(ax, bx) + ext, min(ay, by) - ext, max(ay, by) + ext)
    pax, pay = cv.x[sl] - ax, cv.y[sl] - ay
    bax, bay = bx - ax, by - ay
    h = np.clip((pax * bax + pay * bay) / max(bax * bax + bay * bay, 1e-6), 0, 1)
    dx, dy = pax - bax * h, pay - bay * h
    return sl, np.sqrt(dx * dx + dy * dy) - (r0 + (r1 - r0) * h), h


def lit(base, u, v, r, amb=0.38, spec=0.35, shine=20.0):
    """Colour of a lit ellipsoid with base colour `base` (array or tuple)."""
    nz = np.sqrt(np.clip(1 - r * r, 0, 1))
    n = np.stack([u * 0.9, v * 0.9, nz], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    lam = np.clip(n @ LIGHT, 0, 1)
    refl = 2 * lam[..., None] * n - LIGHT
    hl = np.clip(refl[..., 2], 0, 1) ** shine
    base = np.asarray(base, np.float32)
    col = base * (amb + (1 - amb) * lam)[..., None] + 255 * spec * hl[..., None]
    return col


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def mixc(a, b, t):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return a + (b - a) * np.asarray(t, np.float32)[..., None]


def box_blur(a, r):
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(r + 1, r) if i == axis else (0, 0) for i in range(2)],
                             mode="edge"), axis=axis)
        hi = np.take(c, np.arange(2 * r + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - 2 * r - 1), axis=axis)
        a = (hi - lo) / (2 * r + 1)
    return a


LEG_ROOT = {"front": (36, 12), "mid": (22, 15), "hind": (8, 13)}
LEG_LEN = {"front": (26, 25, 20), "mid": (27, 30, 24), "hind": (31, 34, 27)}
LEG_ANG = {"front": (60, 30, 18), "mid": (98, 112, 126), "hind": (132, 152, 166)}
TRIPOD = {("front", 1): 0, ("mid", -1): 0, ("hind", 1): 0,
          ("front", -1): 1, ("mid", 1): 1, ("hind", -1): 1}
LEG_COL = (128, 90, 50)
TARSUS_COL = (78, 56, 36)


def draw_leg(cv, name, side, swing=0.0, fold=0.0):
    """side +1 = fly's right (+y). swing in degrees moves the whole leg
    forward/back; fold 0..1 bends it towards the body."""
    rx, ry = LEG_ROOT[name]
    p = (rx, ry * side)
    lens = LEG_LEN[name]
    angs = LEG_ANG[name]
    for k, (L, a) in enumerate(zip(lens, angs)):
        a = a - swing
        if fold:
            a = a + fold * (40 + 50 * k) * (1 if name != "front" else -0.6)
            L = L * (1 - 0.35 * fold)
        rad = math.radians(a)
        q = (p[0] + L * math.cos(rad), p[1] + L * math.sin(rad) * side)
        width = (3.3, 2.7, 1.7)[k]
        sl, d, h = capsule(cv, p, q, width, width * (0.85 if k < 2 else 0.6))
        col = LEG_COL if k < 2 else TARSUS_COL
        shade = 0.75 + 0.35 * np.clip(-d / width, 0, 1)
        cv.paint(sl, np.asarray(col, np.float32) * shade[..., None], cv.cover(d))
        if k == 1:
            for j in range(2):
                t = 0.35 + 0.35 * j
                bx, by = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
                nx, ny = -(q[1] - p[1]) / L, (q[0] - p[0]) / L
                sl, d, _ = capsule(cv, (bx, by), (bx + 4 * nx * side + 2, by + 4 * ny * side), 0.45, 0.2)
                cv.paint(sl, (40, 30, 24), cv.cover(d) * 0.5)
        p = q


def draw_legs(cv, phase=None, air=False, dead=False):
    for name in ("hind", "mid", "front"):
        for side in (-1, 1):
            if dead:
                draw_leg(cv, name, side, swing=0, fold=1.0)
            elif air:
                draw_leg(cv, name, side, swing=-35 if name != "front" else -10, fold=0.35)
            else:
                group = TRIPOD[name, side]
                s = math.sin(2 * math.pi * (phase + 0.5 * group))
                draw_leg(cv, name, side, swing=16 * s)


def draw_body(cv, dark=1.0, flat=1.0):
    rng = np.random.default_rng(7)

    sl, d, u, v, r = ellipse(cv, -52, 0, 56, 32 * flat)
    x = cv.x[sl]
    seg = ((-x - 10) % 17.5) / 17.5
    band = smooth(0.62, 0.84, seg) * smooth(-10, -22, x)
    side_fade = 1 - 0.55 * smooth(0.55, 0.95, np.abs(v))
    base = mixc((200, 158, 98), (62, 44, 32), band * side_fade)
    base = mixc(base, (70, 50, 36), smooth(-88, -106, x) * 0.7)
    cv.paint(sl, lit(base, u, v, r, spec=0.25) * dark, cv.cover(d))

    sl, d, u, v, r = ellipse(cv, 22, 0, 37, 32 * flat)
    x, y = cv.x[sl], cv.y[sl]
    stripe = np.exp(-((np.abs(y) - 9) / 3.5) ** 2) * 0.35
    speck = (np.sin(x * 2.3 + np.sin(y * 1.7) * 3) * np.sin(y * 2.9 + x * 0.7) > 0.93) * 0.35
    base = mixc((168, 122, 70), (90, 62, 38), stripe + speck)
    cv.paint(sl, lit(base, u, v, r, spec=0.3) * dark, cv.cover(d))

    sl, d, u, v, r = ellipse(cv, -11, 0, 11, 15 * flat)
    cv.paint(sl, lit((176, 132, 80), u, v, r) * dark, cv.cover(d))

    for bx, by, length in ((36, 7, 16), (22, 9, 17), (6, 7, 22), (-14, 5, 26)):
        for side in (-1, 1):
            p0 = (bx, by * side)
            p1 = (bx - length, (by + 4 + length * 0.12) * side)
            sl, d, _ = capsule(cv, p0, p1, 1.1, 0.3)
            cv.paint(sl, (26, 18, 14), cv.cover(d))
    for _ in range(22):
        bx, by = rng.uniform(-6, 50), rng.uniform(-24, 24)
        if (bx - 22) ** 2 / 34 ** 2 + by ** 2 / 29 ** 2 < 1:
            sl, d, _ = capsule(cv, (bx, by), (bx - 4, by + np.sign(by) * 1.2), 0.45, 0.2)
            cv.paint(sl, (40, 28, 20), cv.cover(d) * 0.8)

    sl, d, u, v, r = ellipse(cv, 64, 0, 17, 35 * flat)
    cv.paint(sl, lit((158, 110, 62), u, v, r) * dark, cv.cover(d))
    sl, d, u, v, r = ellipse(cv, 72, 0, 9, 11)
    cv.paint(sl, lit((196, 130, 64), u, v, r) * dark, cv.cover(d))
    sl, d, u, v, r = ellipse(cv, 62, 0, 4.5, 4.5)
    cv.paint(sl, lit((120, 80, 50), u, v, r) * dark, cv.cover(d) * 0.8)
    for ox, oy in ((64.5, 0), (60, -2.2), (60, 2.2)):
        sl, d, u, v, r = ellipse(cv, ox, oy, 1.0, 1.0)
        cv.paint(sl, (214, 170, 120), cv.cover(d) * 0.7)

    for side in (-1, 1):
        sl, d, u, v, r = ellipse(cv, 66, 22 * side, 16, 18 * flat, rot=0.25 * side)
        x, y = cv.x[sl], cv.y[sl]
        k = 1.7
        hexp = (np.cos(k * x) + np.cos(k * (0.5 * x + 0.866 * y)) + np.cos(k * (-0.5 * x + 0.866 * y)))
        facet = smooth(-1.5, 1.5, hexp)
        base = mixc((120, 14, 12), (206, 36, 28), 0.35 + 0.65 * facet)
        col = lit(base, u, v, r, amb=0.45, spec=0.55, shine=14)
        cv.paint(sl, col * dark, cv.cover(d))
        sl, d, u, v, r = ellipse(cv, 82, 8 * side, 5, 4)
        cv.paint(sl, lit((176, 124, 70), u, v, r) * dark, cv.cover(d))
        p0 = (85, 9 * side)
        p1 = (99, 22 * side)
        sl, d, _ = capsule(cv, p0, p1, 0.7, 0.3)
        cv.paint(sl, (60, 44, 30), cv.cover(d))
        for j in range(5):
            t = 0.2 + 0.16 * j
            bx, by = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
            for s2 in (-1, 1):
                sl, d, _ = capsule(cv, (bx, by), (bx + 3, by + 3.5 * s2 * side * 0.6 - 2 * s2), 0.3)
                cv.paint(sl, (70, 54, 38), cv.cover(d) * 0.7)


WING_ROOT = (16, 9)
WING_VEINS = [((6, 3), (150, 9)), ((6, 1), (158, 2)), ((6, -1), (152, -6)),
              ((8, -3), (140, -14)), ((12, -5), (112, -20)),
              ((66, 1.5), (68, -4.5)), ((100, -6), (104, -16))]


def draw_wing(cv, side, angle, alpha=1.0, length=1.0):
    """angle: radians from straight back, positive = away from the body."""
    rx, ry = WING_ROOT
    root = (rx, ry * side)
    heading = math.pi - angle * side
    dx, dy = math.cos(heading), math.sin(heading)
    px, py = -dy * side, dx * side

    def at(s, t):
        return (root[0] + (s * dx + t * px) * length, root[1] + (s * dy + t * py) * length)

    cx, cy = at(80, 4)
    sl, d, u, v, r = ellipse(cv, cx, cy, 80 * length, 23 * length, rot=heading)
    x, y = cv.x[sl], cv.y[sl]
    s_along = (x - root[0]) * dx + (y - root[1]) * dy
    taper = smooth(0, 30 * length, s_along)
    irid = 0.5 + 0.5 * np.sin(s_along * 0.06 + v * 2.0)
    tint = mixc((205, 222, 240), (236, 226, 246), irid)
    a = cv.cover(d) * (0.12 + 0.18 * smooth(-3, 0, d)) * alpha * (0.3 + 0.7 * taper)
    cv.paint(sl, tint * 0.92, a)
    sl2, d2, _, _, _ = ellipse(cv, cx, cy, 80 * length, 23 * length, rot=heading)
    edge = np.clip(1 - np.abs(d2) * 0.9, 0, 1) * 0.35 * alpha
    cv.paint(sl2, (120, 110, 100), edge)
    for (s0, t0), (s1, t1) in WING_VEINS:
        sl, d, _ = capsule(cv, at(s0, t0), at(s1, t1), 0.75 * length, 0.4 * length)
        cv.paint(sl, (90, 76, 60), cv.cover(d) * 0.6 * alpha)


def draw_proboscis(cv):
    sl, d, h = capsule(cv, (80, 0), (101, 0), 4.2, 3.4)
    u = cv.y[sl] / 4.2
    cv.paint(sl, lit((170, 124, 76), np.zeros_like(u), u, np.abs(u) * 0.9), cv.cover(d))
    for side in (-1, 1):
        sl, d, u, v, r = ellipse(cv, 104, 3.2 * side, 5.5, 3.8)
        cv.paint(sl, lit((196, 146, 92), u, v, r, spec=0.4), cv.cover(d))


def draw_splat(cv, seed=3):
    rng = np.random.default_rng(seed)
    for _ in range(26):
        ang = rng.uniform(0, 2 * math.pi)
        dist = abs(rng.normal(0, 38))
        rad = rng.uniform(4, 16) * (1 - min(dist / 120, 0.8))
        cx, cy = -20 + dist * math.cos(ang), dist * math.sin(ang)
        sl, d, u, v, r = ellipse(cv, cx, cy, rad, rad * rng.uniform(0.7, 1.0), rot=ang)
        cv.paint(sl, lit((92, 26, 20), u, v, r, amb=0.6, spec=0.45), cv.cover(d) * 0.9)


def build_fly(out_px):
    """All fly layers, smooth-scaled to out_px x out_px."""
    sprites = {}
    sprites["legs"] = []
    for i in range(GAIT_FRAMES):
        cv = Canvas()
        draw_legs(cv, phase=i / GAIT_FRAMES)
        sprites["legs"].append(cv.surface(out_px))
    cv = Canvas()
    draw_legs(cv, air=True)
    sprites["legs_air"] = cv.surface(out_px)

    cv = Canvas()
    draw_body(cv)
    sprites["body"] = cv.surface(out_px)

    cv = Canvas()
    draw_proboscis(cv)
    sprites["proboscis"] = cv.surface(out_px)

    cv = Canvas()
    draw_wing(cv, 1, 0.1)
    draw_wing(cv, -1, 0.1)
    sprites["wings_rest"] = cv.surface(out_px)

    sprites["wings_flight"] = []
    for frame in range(2):
        cv = Canvas()
        for side in (-1, 1):
            for a in np.linspace(0.5 + 0.3 * frame, 1.7 + 0.2 * frame, 6):
                draw_wing(cv, side, a, alpha=0.45, length=0.78)
        sprites["wings_flight"].append(cv.surface(out_px))

    cv = Canvas()
    draw_legs(cv, dead=True)
    draw_wing(cv, 1, 0.95)
    draw_wing(cv, -1, 0.75)
    draw_body(cv, dark=0.62, flat=1.12)
    sprites["dead"] = cv.surface(out_px)
    cv = Canvas()
    draw_splat(cv)
    sprites["splat"] = cv.surface(out_px)

    cv = Canvas()
    draw_legs(cv, phase=0)
    draw_body(cv)
    draw_wing(cv, 1, 0.1)
    draw_wing(cv, -1, 0.1)
    _, a = cv.arrays()
    a = box_blur(a, 5) * 0.55
    sprites["shadow"] = to_surface(np.zeros(a.shape + (3,)), a, out_px)
    return sprites


def build_swatter(side_px, color=(226, 64, 52)):
    """Square swatter head with a hole mesh, seen from above, plus its
    shadow (the same mesh in black). The handle leaves the bottom edge."""
    n = 360
    cv = Canvas(size=n, ss=2)
    x, y = cv.x, cv.y
    h = n * 0.44
    corner = n * 0.07
    qx, qy = np.abs(x) - (h - corner), np.abs(y) - (h - corner)
    d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - corner
    rim = n * 0.035
    pitch, hole = n * 0.052, n * 0.034
    gx = np.abs(((x + pitch / 2) % pitch) - pitch / 2) - hole / 2
    gy = np.abs(((y + pitch / 2) % pitch) - pitch / 2) - hole / 2
    ring = np.maximum(d, -(d + rim))
    mesh = np.maximum(-np.maximum(gx, gy), d)
    solid = np.minimum(ring, mesh)
    neck = capsule(cv, (0, h - 4), (0, n / 2 - 2), n * 0.03)
    grad = 0.8 + 0.25 * (-(x + y) / n)
    col = np.asarray(color, np.float32) * grad[..., None]
    rim_band = smooth(-rim, -rim * 0.4, d) * (d < 0)
    col = mixc(col, np.asarray(color, np.float32) * 0.62, rim_band * 0.6)
    edge_hl = np.exp(-((d + rim * 0.5) / 1.5) ** 2) * 0.35
    col = col + 255 * edge_hl[..., None] * 0.4
    cv.paint((slice(None), slice(None)), col, cv.cover(solid))
    sl, dn, _ = neck
    cv.paint(sl, np.asarray(color, np.float32) * 0.7, cv.cover(dn))
    rgb, a = cv.arrays()
    head = to_surface(rgb, a, side_px)
    shadow = to_surface(np.zeros(rgb.shape), box_blur(a, 2) * 0.8, side_px)
    return head, shadow, h / n


def build_drop(px, tint):
    """A liquid drop seen from above: mostly clear, a dark refracting rim, a
    bright caustic opposite the light and a specular highlight."""
    n = 128
    cv = Canvas(size=n, ss=2)
    rad = n * 0.44
    sl, d, u, v, r = ellipse(cv, 0, 0, rad, rad)
    tint = np.asarray(tint, np.float32)
    body = cv.cover(d) * (0.18 + 0.5 * smooth(0.55, 1.0, r))
    col = tint * (0.55 + 0.25 * (1 - r))[..., None]
    cv.paint(sl, col, body)
    caustic = np.exp(-(((u - 0.35) ** 2 + (v - 0.4) ** 2) / 0.08)) * cv.cover(d)
    cv.paint(sl, np.minimum(tint * 1.25 + 40, 255), caustic * 0.55)
    hl = np.exp(-(((u + 0.38) ** 2 + (v + 0.42) ** 2) / 0.012)) * cv.cover(d)
    cv.paint(sl, (255, 255, 255), hl * 0.95)
    rim = np.exp(-((d + 1.2) / 1.4) ** 2) * cv.cover(d - 1)
    cv.paint(sl, tint * 0.45, rim * 0.7)
    return cv.surface(px)
