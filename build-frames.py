#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build-frames.py — сборка прозрачной секвенции кадров маскота для index.html.

Что делает:
  1. ffmpeg: достаёт каждый STEP-й кадр из assets/mascot-original.mp4,
     масштабирует по высоте до FRAME_H, кладёт PNG в .build/png/
  2. Снимает фон ЗАЛИВКОЙ ОТ ГРАНИЦ КАДРА (связные с рамкой компоненты
     пикселей, близких к цвету фона), а НЕ глобальным color key —
     поэтому белки глаз, зубы и белые кроссовки остаются непрозрачными.
  3. Мягкая альфа в узкой полосе вокруг фоновой области + деконтаминация
     цвета — убирает белую кайму на волосах.
  4. Затухание альфы по нижним BOTTOM_FADE (6%) высоты кадра.
  5. Пишет assets/frames/fNNN.webp (lossy WebP с альфой) и assets/poster.webp.
  6. Печатает проверку альфы числами и итоговый размер.

Окружение (ffmpeg без webp-кодировщика, ImageMagick нет):
    python3 -m venv .venv && .venv/bin/pip install pillow numpy scipy
    .venv/bin/python build-frames.py

Запуск:  .venv/bin/python build-frames.py [--quality 78] [--step 2] [--budget 8]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "assets" / "mascot-original.mp4"
PNG_DIR = ROOT / ".build" / "png"
OUT_DIR = ROOT / "assets" / "frames"
POSTER = ROOT / "assets" / "poster.webp"

FRAME_H = 960          # высота кадра на выходе, ширина по пропорции 834/1112 -> 720
BOTTOM_FADE = 0.06     # доля высоты, по которой альфа гаснет к нулю

# Фон студии — стабильный #fbfbfb на всех 241 кадрах (проверено по угловым
# патчам 24x24: медиана (251,251,251) на каждом 10-м кадре без дрейфа).
# Берём константу, а не медиану рамки: на крупных планах шляпа и волосы
# пересекают верхнюю границу и медиану рамки сносит в (233,225,224).
BG = np.array([251.0, 251.0, 251.0])

T_FLOOD = 44   # допуск заливки от границ (17% по каналу)
T_HARD = 34    # внутри фоновой области пиксели ближе этого к фону — жёсткий 0
FEATHER = 12   # ширина полосы мягкой альфы вглубь силуэта, px
A_LO = 12      # d <= A_LO  -> alpha 0
A_HI = 90      # d >= A_HI  -> alpha 1
BLUR = 0.7     # сглаживание альфы ~1px

# Второй проход по столу: на общих планах передний край стола темнее допуска
# (rgb ~204,195,195) и остаётся непрозрачным блоком в y=828..907. Расширяем
# заливку в нижней полосе кадра. Ноутбук и кожа при этом целы: d у них >= 100,
# а мягкая альфа даёт таким пикселям 1 даже внутри фоновой области.
# Только на общих планах — на крупных планах в этой же полосе белые кроссовки.
T_DESK = 110
DESK_BAND = 0.84


def extract_png(step: int) -> list[Path]:
    """ffmpeg: каждый step-й кадр в PNG высотой FRAME_H."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(PNG_DIR.glob("f*.png"))
    if existing:
        print(f"[png] уже извлечено {len(existing)} кадров в {PNG_DIR}")
        return existing
    if not SRC.exists():
        sys.exit(f"нет исходника {SRC}")
    vf = f"select='not(mod(n\\,{step}))',scale=-2:{FRAME_H}"
    # -fps_mode vfr: в ffmpeg 8 опция -vsync удалена
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(SRC),
           "-vf", vf, "-fps_mode", "vfr", "-start_number", "0",
           str(PNG_DIR / "f%03d.png")]
    subprocess.run(cmd, check=True)
    files = sorted(PNG_DIR.glob("f*.png"))
    print(f"[png] извлечено {len(files)} кадров")
    return files


def background_region_from(mask: np.ndarray) -> np.ndarray:
    """Компоненты маски, связные с рамкой кадра (заливка ОТ ГРАНИЦ, не color key)."""
    lab, _ = ndimage.label(mask)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    edge.discard(0)
    return np.isin(lab, list(edge))


def background_region(d: np.ndarray) -> np.ndarray:
    return background_region_from(d <= T_FLOOD)


def matte(rgb: np.ndarray):
    """RGB float -> (rgb с деконтаминацией, alpha 0..1, фоновая область, общий ли план)."""
    d = np.abs(rgb - BG).max(axis=2)
    h, w = d.shape
    R = background_region(d)

    # общий план = верхняя строка кадра целиком фон (на крупных планах её
    # пересекают волосы и шляпа). Только там есть стол.
    wide = bool(R[0].mean() > 0.995)
    if wide:
        m = d <= T_FLOOD
        band = np.zeros_like(m)
        band[int(h * DESK_BAND):] = True
        R = background_region_from(m | ((d <= T_DESK) & band))

    # мягкая альфа только в полосе FEATHER вокруг фона: так белая кайма на
    # волосах становится полупрозрачной, а белые кроссовки (d=6, но >16px
    # от фона) остаются целыми.
    dist = ndimage.distance_transform_edt(~R)
    soft = np.clip((d - A_LO) / (A_HI - A_LO), 0.0, 1.0)
    alpha = np.ones(d.shape, dtype=np.float32)
    alpha[dist <= FEATHER] = soft[dist <= FEATHER]
    alpha[R & (d <= T_HARD)] = 0.0
    alpha = ndimage.gaussian_filter(alpha, BLUR)
    alpha[R & (d <= T_HARD - 6)] = 0.0

    # затухание по нижним 6% высоты — срез волос/тела растворяется
    nb = int(round(h * BOTTOM_FADE))
    alpha[h - nb:] *= np.linspace(1.0, 0.0, nb, dtype=np.float32)[:, None]

    # деконтаминация цвета: C_fg = (C - (1-a)*BG) / a
    out = rgb.copy()
    sel = (alpha > 0.04) & (alpha < 0.96)
    a = alpha[sel][:, None]
    out[sel] = np.clip((rgb[sel] - (1 - a) * BG) / a, 0, 255)
    return out, alpha, R, wide


def alpha255(alpha: np.ndarray, x: int, y: int) -> int:
    return int(round(float(alpha[y, x]) * 255))


def find_blob(mask: np.ndarray, min_px: int = 200):
    """Центроид крупнейшей связной компоненты маски -> (x, y, size)."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    i = int(np.argmax(sizes))
    if sizes[i] < min_px:
        return None
    cy, cx = ndimage.center_of_mass(lab == i + 1)
    return int(round(cx)), int(round(cy)), int(sizes[i])


def find_pupils(rgb: np.ndarray, alpha: np.ndarray):
    """Пара зрачков: два тёмных пятна близкой площади на одной высоте."""
    h, w, _ = rgb.shape
    dark = (rgb.max(axis=2) < 90) & (alpha > 200)
    dark[int(h * 0.66):] = False
    lab, n = ndimage.label(dark)
    if not n:
        return None
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    blobs = []
    for i in np.argsort(sizes)[::-1][:15]:
        s = int(sizes[i])
        if not (150 <= s <= 4000):
            continue
        ys, xs = np.nonzero(lab == i + 1)
        blobs.append((int(xs.mean()), int(ys.mean()), s))
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            a, b = blobs[i], blobs[j]
            if abs(a[1] - b[1]) <= 12 and 60 < abs(a[0] - b[0]) < w * 0.55:
                return sorted([a, b])
    return None


def sclera_near(rgb: np.ndarray, alpha: np.ndarray, px: int, py: int, r: int = 40):
    """Самый светлый пиксель рядом со зрачком = белок глаза."""
    h, w, _ = rgb.shape
    y0, y1 = max(0, py - r // 2), min(h, py + r // 2)
    x0, x1 = max(0, px - r), min(w, px + r)
    patch = rgb[y0:y1, x0:x1]
    m = patch.min(axis=2)
    k = int(np.argmax(m))
    dy, dx = divmod(k, patch.shape[1])
    return x0 + dx, y0 + dy


def verify_frame(path: Path, label: str):
    """Числа приёмки, замеренные на ГОТОВОМ .webp. Координаты ищутся, не выдумываются."""
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im)
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]
    h, w = alpha.shape
    print(f"\n  --- {path.name} ({label}) {w}x{h} ---")
    corners = [("TL", 0, 0), ("TR", w - 1, 0), ("BL", 0, h - 1), ("BR", w - 1, h - 1)]
    print("    углы: " + "  ".join(f"{n}({x},{y})={alpha[y, x]}" for n, x, y in corners))

    solid = alpha > 200
    inside = ndimage.distance_transform_edt(solid) > 12

    # лицо: кожа тёплая и светлая; G-B отделяет кожу от жёлтой футболки
    skin = ((rgb[:, :, 0] > 190) & (rgb[:, :, 0] - rgb[:, :, 1] > 35) &
            (rgb[:, :, 0] - rgb[:, :, 1] < 85) & (rgb[:, :, 1] - rgb[:, :, 2] > 12) &
            (rgb[:, :, 1] - rgb[:, :, 2] < 60) & inside)
    b = find_blob(skin, 1000)
    if b:
        x, y, n = b
        print(f"    центр лица  ({x},{y}) rgb={tuple(rgb[y, x])} alpha={alpha[y, x]}  [кожа {n} px]")

    # ноутбук: серый малонасыщенный массив в нижней части кадра
    grey = ((np.abs(rgb[:, :, 0] - rgb[:, :, 1]) < 14) & (np.abs(rgb[:, :, 1] - rgb[:, :, 2]) < 14) &
            (rgb[:, :, 0] > 120) & (rgb[:, :, 0] < 205) & inside)
    grey[: int(h * 0.55)] = False
    b = find_blob(grey, 5000)
    if b:
        x, y, n = b
        print(f"    ноутбук     ({x},{y}) rgb={tuple(rgb[y, x])} alpha={alpha[y, x]}  [серое {n} px]")
    else:
        print("    ноутбук     нет в кадре (крупный план)")

    # белки глаз: сначала пара зрачков, потом светлейший пиксель рядом
    pup = find_pupils(rgb, alpha)
    if pup:
        for (px, py, sz) in pup:
            sx, sy = sclera_near(rgb, alpha, px, py)
            print(f"    зрачок ({px},{py}) {sz}px -> белок ({sx},{sy}) "
                  f"rgb={tuple(rgb[sy, sx])} alpha={alpha[sy, sx]}")
    else:
        print("    зрачки не найдены (глаза закрыты / нет крупного плана)")

    # дырки: прозрачные компоненты, не связанные с рамкой кадра
    holes = alpha < 128
    lab, n = ndimage.label(holes)
    inner = []
    if n:
        edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        edge.discard(0)
        sizes = ndimage.sum(holes, lab, range(1, n + 1))
        inner = sorted((int(s) for j, s in enumerate(sizes) if (j + 1) not in edge), reverse=True)
    print(f"    дырки внутри фигуры: {len(inner)} шт, крупнейшие {inner[:3] if inner else '—'} px")


def verify_outputs():
    files = sorted(OUT_DIR.glob("f*.webp"))
    if not files:
        sys.exit("нет собранных кадров")
    n = len(files)
    print(f"\n=== ПРОВЕРКА АЛЬФЫ на готовых WebP ({n} кадров) ===")
    for idx, label in [(0, "старт, общий план"), (n // 2, "удивление, глаза открыты"),
                       (n - 1, "финал, крупный план")]:
        verify_frame(files[idx], label)
    total = sum(f.stat().st_size for f in files)
    print(f"\n[размер] {n} кадров = {total/1024/1024:.2f} MB, средний {total/n/1024:.1f} KB")
    if POSTER.exists():
        print(f"[размер] poster.webp = {POSTER.stat().st_size/1024:.1f} KB "
              f"({Image.open(POSTER).size[0]}x{Image.open(POSTER).size[1]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=78)
    ap.add_argument("--step", type=int, default=2, help="каждый N-й кадр исходника")
    ap.add_argument("--budget", type=float, default=8.0, help="бюджет кадров, MB")
    ap.add_argument("--verify", action="store_true", help="только проверить готовые кадры")
    args = ap.parse_args()

    if args.verify:
        verify_outputs()
        return

    files = extract_png(args.step)
    if not files:
        sys.exit("нет PNG кадров")

    if OUT_DIR.exists():
        for f in OUT_DIR.glob("*.webp"):
            f.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    print(f"\n[matte] BG={BG.astype(int)} T_FLOOD={T_FLOOD} T_HARD={T_HARD} "
          f"FEATHER={FEATHER} A_LO={A_LO} A_HI={A_HI} fade={BOTTOM_FADE:.0%}")

    wide_flags = []
    for i, p in enumerate(files):
        rgb = np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
        out, alpha, R, wide = matte(rgb)
        wide_flags.append(wide)
        rgba = np.dstack([out, alpha * 255]).astype(np.uint8)
        img = Image.fromarray(rgba, "RGBA")
        dst = OUT_DIR / f"f{i:03d}.webp"
        img.save(dst, "WEBP", quality=args.quality, method=6)
        total += dst.stat().st_size
        if i == 0:
            img.save(POSTER, "WEBP", quality=max(args.quality, 82), method=6)
        if i % 20 == 0:
            print(f"    ...{i + 1}/{len(files)}", flush=True)

    # общий план должен смениться крупным один раз, без мигания проходa по столу
    switches = sum(1 for a, b in zip(wide_flags, wide_flags[1:]) if a != b)
    last_wide = max((i for i, f in enumerate(wide_flags) if f), default=-1)
    print(f"\n[стол] общий план на кадрах 0..{last_wide}, переключений флага: {switches} "
          f"({'OK, без мигания' if switches <= 1 else 'ВНИМАНИЕ: флаг скачет'})")

    verify_outputs()
    mb = total / 1024 / 1024
    print(f"\n[итог] кадров: {len(files)}  ({OUT_DIR.relative_to(ROOT)}/f000..f{len(files)-1:03d}.webp)")
    print(f"[итог] размер кадров: {mb:.2f} MB (бюджет {args.budget} MB) — "
          f"{'OK' if mb <= args.budget else 'ПРЕВЫШЕН, взять --step 3 или --quality 72'}")
    ps = POSTER.stat().st_size / 1024
    print(f"[итог] poster.webp: {ps:.1f} KB — {'OK' if ps <= 120 else 'ПРЕВЫШЕН лимит 120 KB'}")
    print(f"[итог] средний кадр: {total / len(files) / 1024:.1f} KB, quality={args.quality}")
    if mb > args.budget:
        sys.exit(1)


if __name__ == "__main__":
    main()
