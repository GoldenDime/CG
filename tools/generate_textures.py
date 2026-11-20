"""Генерация текстур для ЛР №3 и №4.

Запуск из корня проекта:  python tools/generate_textures.py
Результат пишется в assets/textures. Нужны numpy и Pillow.

Для каждого материала создаётся альбедо и, при необходимости, specular
(какие места блестят) и emissive (какие места светятся) текстуры.
Emissive текстура лавы построена на шуме Перлина.
"""

from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 512
OUT = Path(__file__).resolve().parent.parent / "assets" / "textures"
rng = np.random.default_rng(1)


# ---- Шум Перлина (тайлящийся) ------------------------------------------------

def perlin(size, period, seed):
    """Градиентный шум Перлина, бесшовно повторяющийся с периодом `period` ячеек."""
    r = np.random.default_rng(seed)
    angles = r.uniform(0, 2 * np.pi, (period + 1, period + 1))
    angles[-1, :] = angles[0, :]
    angles[:, -1] = angles[:, 0]
    gx, gy = np.cos(angles), np.sin(angles)

    coords = np.linspace(0, period, size, endpoint=False)
    x, y = np.meshgrid(coords, coords)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0

    def dot(ix, iy, dx, dy):
        return gx[iy, ix] * dx + gy[iy, ix] * dy

    n00 = dot(x0, y0, fx, fy)
    n10 = dot(x0 + 1, y0, fx - 1, fy)
    n01 = dot(x0, y0 + 1, fx, fy - 1)
    n11 = dot(x0 + 1, y0 + 1, fx - 1, fy - 1)

    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    u, v = fade(fx), fade(fy)
    return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v


def fbm(size, base_period, octaves, seed, persistence=0.5):
    result = np.zeros((size, size))
    amplitude, total = 1.0, 0.0
    for i in range(octaves):
        result += amplitude * perlin(size, base_period * 2 ** i, seed + i)
        total += amplitude
        amplitude *= persistence
    return result / total


def save(name, rgb):
    OUT.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), "RGB")
    if name.endswith(".jpg"):
        img.save(OUT / name, quality=92)
    else:
        img.save(OUT / name)
    print("written", OUT / name)


def gray(v):
    return np.repeat(v[..., None], 3, axis=2)


def lerp(a, b, t):
    return a + (b - a) * t[..., None]


coords = np.arange(SIZE) / SIZE
X, Y = np.meshgrid(coords, coords)


# ---- Деревянный ящик с металлической рамкой ----------------------------------

def crate():
    grain = fbm(SIZE, 4, 4, 10)
    stretched = np.sin((X * 40 + grain * 6) * np.pi) * 0.5 + 0.5
    plank = np.floor(Y * 5)
    plank_shade = 0.85 + 0.15 * np.sin(plank * 12.9898) ** 2
    wood_dark = np.array([0.35, 0.20, 0.09])
    wood_light = np.array([0.62, 0.40, 0.20])
    albedo = lerp(wood_dark, wood_light, stretched * 0.6 + 0.4 * (grain * 0.5 + 0.5)) * plank_shade[..., None]

    gap = (np.abs(Y * 5 - np.round(Y * 5)) < 0.03)
    albedo[gap] *= 0.35

    border = 0.09
    frame = (X < border) | (X > 1 - border) | (Y < border) | (Y > 1 - border)
    diag = (np.abs(X - Y) < border * 0.7) | (np.abs(X + Y - 1) < border * 0.7)
    metal_mask = frame | diag
    metal = np.array([0.55, 0.56, 0.58]) * (0.85 + 0.15 * fbm(SIZE, 16, 3, 20)[..., None])
    albedo[metal_mask] = metal[metal_mask]

    # Заклёпки
    for cx in (border / 2, 1 - border / 2):
        for cy in np.linspace(border / 2, 1 - border / 2, 6):
            for px, py in ((cx, cy), (cy, cx)):
                rivet = (X - px) ** 2 + (Y - py) ** 2 < 0.012 ** 2
                albedo[rivet] = [0.8, 0.8, 0.82]

    save("crate_albedo.png", albedo)

    spec = np.where(metal_mask, 0.9, 0.08) * (0.8 + 0.2 * fbm(SIZE, 32, 2, 30) * 0.5 + 0.1)
    save("crate_specular.png", gray(spec))


# ---- Кирпичная стена (JPEG, без specular/emissive) ---------------------------

def bricks():
    rows = 8
    row = np.floor(Y * rows)
    offset = (row % 2) * 0.5
    bx = (X * 4 + offset) % 1.0
    by = (Y * rows) % 1.0
    mortar = (bx < 0.04) | (by < 0.08)

    brick_id = np.floor(X * 4 + offset) + row * 7
    tint = 0.75 + 0.25 * np.sin(brick_id * 78.233) ** 2
    noise = fbm(SIZE, 8, 4, 40) * 0.5 + 0.5
    brick = np.array([0.55, 0.22, 0.15]) * (tint * (0.8 + 0.4 * noise))[..., None]
    mortar_color = np.array([0.7, 0.68, 0.62]) * (0.85 + 0.3 * noise)[..., None]
    albedo = np.where(mortar[..., None], mortar_color, brick)
    save("bricks_albedo.jpg", albedo)


# ---- Плитка для пола ----------------------------------------------------------

def tiles():
    n = 4
    tx = (X * n) % 1.0
    ty = (Y * n) % 1.0
    grout = (tx < 0.03) | (ty < 0.03) | (tx > 0.97) | (ty > 0.97)
    checker = ((np.floor(X * n) + np.floor(Y * n)) % 2)[..., None]
    marble = np.sin((X + Y) * 20 + fbm(SIZE, 4, 5, 50) * 8) * 0.5 + 0.5
    light = np.array([0.85, 0.84, 0.80])
    dark = np.array([0.25, 0.27, 0.30])
    base = light * checker + dark * (1 - checker)
    albedo = base * (0.85 + 0.15 * marble)[..., None]
    albedo[grout] = [0.15, 0.15, 0.15]
    save("tiles_albedo.png", albedo)

    spec = np.where(grout, 0.0, 0.6 + 0.3 * marble)
    save("tiles_specular.png", gray(spec))


# ---- Лавовый камень: emissive из шума Перлина --------------------------------

def lava():
    n = fbm(SIZE, 4, 6, 60)
    cracks = 1.0 - np.abs(n) * 6.0          # «ридж»-шум: светлые узкие линии там, где шум ~ 0
    cracks = np.clip(cracks, 0, 1) ** 3

    rock_noise = fbm(SIZE, 8, 5, 70) * 0.5 + 0.5
    rock = np.array([0.18, 0.16, 0.15]) * (0.6 + 0.8 * rock_noise)[..., None]
    albedo = lerp(rock, np.array([0.35, 0.12, 0.04]), cracks)
    save("lava_albedo.png", albedo)

    spec = (1 - cracks) * (0.25 + 0.5 * rock_noise)
    save("lava_specular.png", gray(spec))

    heat = cracks * (0.7 + 0.3 * (fbm(SIZE, 16, 3, 80) * 0.5 + 0.5))
    emissive = np.stack([heat * 1.0, heat ** 1.8 * 0.45, heat ** 4 * 0.1], axis=-1)
    save("lava_emissive.png", emissive)


# ---- Металлическая панель со светящимися индикаторами -------------------------

def scifi_panel():
    noise = fbm(SIZE, 16, 4, 90) * 0.5 + 0.5
    brushed = fbm(SIZE, 2, 3, 95)
    brushed = np.sin(Y * 400 + brushed * 20) * 0.03
    base = np.array([0.42, 0.45, 0.5]) * (0.85 + 0.15 * noise + brushed)[..., None]

    cells = 2
    cx = (X * cells) % 1.0
    cy = (Y * cells) % 1.0
    seam = (cx < 0.015) | (cy < 0.015) | (cx > 0.985) | (cy > 0.985)
    base[seam] *= 0.3

    # Светящиеся полосы и круглые индикаторы
    stripe = (np.abs(cy - 0.5) < 0.025) & (cx > 0.15) & (cx < 0.85)
    dots = np.zeros_like(X, dtype=bool)
    for px in (0.2, 0.8):
        for py in (0.2, 0.8):
            dots |= (cx - px) ** 2 + (cy - py) ** 2 < 0.04 ** 2
    glow = stripe | dots
    base[glow] = [0.1, 0.12, 0.14]
    save("panel_albedo.png", base)

    spec = np.where(seam, 0.05, 0.5 + 0.4 * noise)
    spec[glow] = 0.2
    save("panel_specular.png", gray(spec))

    emissive = np.zeros((SIZE, SIZE, 3))
    emissive[stripe] = [0.1, 0.8, 1.0]
    emissive[dots] = [1.0, 0.35, 0.1]
    save("panel_emissive.png", emissive)


# ---- Узор для нетривиального сэмплирования -----------------------------------

def rune():
    angle = np.arctan2(Y - 0.5, X - 0.5)
    radius = np.hypot(X - 0.5, Y - 0.5)
    rings = np.sin(radius * 60) * 0.5 + 0.5
    spokes = np.sin(angle * 8) * 0.5 + 0.5
    n = fbm(SIZE, 4, 4, 100) * 0.5 + 0.5
    r = rings * 0.6 + spokes * 0.4
    albedo = np.stack([0.2 + 0.6 * r * n, 0.15 + 0.4 * spokes, 0.5 + 0.5 * rings * (1 - n)], axis=-1)
    save("rune_albedo.png", albedo)


if __name__ == "__main__":
    crate()
    bricks()
    tiles()
    lava()
    scifi_panel()
    rune()
