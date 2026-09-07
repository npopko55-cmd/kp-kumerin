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
  4. Затухание альфы по краям кадра — прячет прямые срезы исходника:
     нижние BOTTOM_FADE (6%), левый и правый по LR_FADE (7%) ширины,
     верхний по TOP_FADE (5%) высоты. Затухание работает только там,
     где непрозрачные пиксели реально доходят до края (ворота по строкам
     для боков и по колонкам для верха), иначе на общих планах гасла бы
     верхушка шапки, которая до края не достаёт.
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

# Лёгкий набор для мобильного (<= 1023px): каждый второй кадр десктопного
# набора, меньшая высота, ниже качество. Матирование то же самое — оно
# считается на полном разрешении, уменьшение идёт последним шагом.
OUT_DIR_M = ROOT / "assets" / "frames-m"
POSTER_M = ROOT / "assets" / "poster-m.webp"
FRAME_H_M = 640        # высота мобильного кадра, ширина по пропорции -> 480

FRAME_H = 960          # высота кадра на выходе, ширина по пропорции 834/1112 -> 720
BOTTOM_FADE = 0.06     # доля высоты, по которой альфа гаснет к нулю
LR_FADE = 0.07         # доля ширины: затухание по левому и правому краям
TOP_FADE = 0.05        # доля высоты: затухание по верхнему краю

# Ворота затухания: гасим край только там, где силуэт реально его достаёт.
EDGE_TOUCH = 0.15      # alpha, выше которой считаем «пиксель дошёл до края»
EDGE_PROBE = 3         # сколько пикселей от края щупаем
EDGE_DILATE = 41       # расширение зоны действия ворот вдоль края, px
EDGE_SIGMA = 9.0       # сглаживание ворот вдоль края, px (без него — ступеньки)

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


def _ramp(n: int) -> np.ndarray:
    """Smoothstep 0 -> 1 по n пикселям от края внутрь.

    Плавнее линейного: у внутренней границы полосы производная нулевая,
    поэтому шов «полоса / непрозрачное тело» не читается тонкой линией.
    """
    t = (np.arange(n, dtype=np.float32) + 0.5) / n
    return t * t * (3.0 - 2.0 * t)


def _gate(touch: np.ndarray) -> np.ndarray:
    """Ворота 0..1 вдоль края: 1 там, где силуэт доходит до края, 0 где пусто."""
    g = ndimage.grey_dilation(touch.astype(np.float32), size=EDGE_DILATE)
    g = ndimage.gaussian_filter1d(g, EDGE_SIGMA)
    return np.clip(g, 0.0, 1.0)


def edge_fade(alpha: np.ndarray) -> np.ndarray:
    """Гасит альфу у левого/правого/верхнего/нижнего краёв кадра.

    Пиксели за кадром не восстановить, поэтому прямой срез растворяем.
    Множитель = 1 - gate * (1 - ramp): где ворота 0 (край пуст) — множитель 1,
    и на общем плане верхушка шапки на y=20 остаётся целой.
    """
    h, w = alpha.shape
    nx = int(round(w * LR_FADE))
    ny = int(round(h * TOP_FADE))
    nb = int(round(h * BOTTOM_FADE))
    rx, ry = _ramp(nx), _ramp(ny)

    gl = _gate(alpha[:, :EDGE_PROBE].max(axis=1) > EDGE_TOUCH)
    alpha[:, :nx] *= 1.0 - gl[:, None] * (1.0 - rx[None, :])

    gr = _gate(alpha[:, w - EDGE_PROBE:].max(axis=1) > EDGE_TOUCH)
    alpha[:, w - nx:] *= 1.0 - gr[:, None] * (1.0 - rx[::-1][None, :])

    gt = _gate(alpha[:EDGE_PROBE, :].max(axis=0) > EDGE_TOUCH)
    alpha[:ny, :] *= 1.0 - gt[None, :] * (1.0 - ry[:, None])

    # низ гасим всегда и линейно: там срез тела совпадает с нижней кромкой
    # экрана, ворота не нужны, а линейный клин уже принят заказчиком в v1.
    alpha[h - nb:] *= np.linspace(1.0, 0.0, nb, dtype=np.float32)[:, None]
    return alpha


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

    # Деконтаминация цвета: C_fg = (C - (1-a)*BG) / a.
    # Считается ДО искусственного затухания краёв — там альфа занижена
    # намеренно, и деление на неё увело бы волосы в чёрный (тёмная кайма).
    out = rgb.copy()
    sel = (alpha > 0.04) & (alpha < 0.96)
    a = alpha[sel][:, None]
    out[sel] = np.clip((rgb[sel] - (1 - a) * BG) / a, 0, 255)

    # затухание по краям кадра — прячет прямые срезы исходника
    alpha = edge_fade(alpha)
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


def verify_edges(path: Path):
    """Числа затухания по краям на готовом .webp (раздел «Кадры» в ТЗ)."""
    arr = np.asarray(Image.open(path).convert("RGBA"))
    alpha = arr[..., 3]
    h, w = alpha.shape
    xin = int(round(0.12 * w))                       # 0.12*w — глубина «внутри»
    hair = (alpha[:, xin] > 200) & (alpha[:, w - 1 - xin] > 200)
    n = int(hair.sum())
    l2 = int(alpha[hair, 2].max()) if n else 0
    r3 = int(alpha[hair, w - 3].max()) if n else 0
    il = int(alpha[hair, xin].min()) if n else 0
    ir = int(alpha[hair, w - 1 - xin].min()) if n else 0
    top4 = int(alpha[:4, :].max())
    print(f"\n  --- {path.name}: края ({w}x{h}, строк с волосами до края: {n}) ---")
    print(f"    x=2      max alpha = {l2:3d}  (<= 40  {'OK' if l2 <= 40 else 'FAIL'})")
    print(f"    x=w-3    max alpha = {r3:3d}  (<= 40  {'OK' if r3 <= 40 else 'FAIL'})")
    print(f"    x=0.12w={xin:3d} min alpha = {il:3d}  (>= 200 {'OK' if il >= 200 else 'FAIL'})")
    print(f"    x=w-1-0.12w  min alpha = {ir:3d}  (>= 200 {'OK' if ir >= 200 else 'FAIL'})")
    print(f"    верхние 4 строки max alpha = {top4:3d}  (<= 40  {'OK' if top4 <= 40 else 'FAIL'})")
    return l2 <= 40 and r3 <= 40 and il >= 200 and ir >= 200 and top4 <= 40


def verify_outputs():
    files = sorted(OUT_DIR.glob("f*.webp"))
    if not files:
        sys.exit("нет собранных кадров")
    n = len(files)
    print(f"\n=== ПРОВЕРКА АЛЬФЫ на готовых WebP ({n} кадров) ===")
    for idx, label in [(0, "старт, общий план"), (n // 2, "удивление, глаза открыты"),
                       (n - 1, "финал, крупный план")]:
        verify_frame(files[idx], label)
    for idx in (0, min(90, n - 1)):
        verify_edges(files[idx])
    total = sum(f.stat().st_size for f in files)
    print(f"\n[размер] {n} кадров = {total/1024/1024:.2f} MB, средний {total/n/1024:.1f} KB")
    if POSTER.exists():
        print(f"[размер] poster.webp = {POSTER.stat().st_size/1024:.1f} KB "
              f"({Image.open(POSTER).size[0]}x{Image.open(POSTER).size[1]})")


def downscale(rgb: np.ndarray, alpha: np.ndarray, target_h: int):
    """Уменьшение RGBA с ПРЕДУМНОЖЕНИЕМ альфы.

    Без предумножения LANCZOS подмешивает в контур цвет прозрачных пикселей
    (у нас это почти белый фон студии) и на волосах снова появляется кайма.
    Полностью прозрачные пиксели заливаем цветом фона — lossy WebP хранит RGB
    и в нулевой альфе, чёрный там дал бы тёмный ореол на полупрозрачном крае.
    """
    h, w = alpha.shape
    tw = int(round(w * target_h / h))
    a_small = np.asarray(
        Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8), "L")
             .resize((tw, target_h), Image.LANCZOS)
    ).astype(np.float32) / 255.0
    prem = np.clip(rgb * alpha[..., None], 0, 255).astype(np.uint8)
    p_small = np.asarray(
        Image.fromarray(prem, "RGB").resize((tw, target_h), Image.LANCZOS)
    ).astype(np.float32)
    out = np.empty_like(p_small)
    m = a_small > 0.004
    out[m] = np.clip(p_small[m] / a_small[m][:, None], 0, 255)
    out[~m] = BG
    return out, a_small


def build_mobile(files: list[Path], quality: int, budget: float) -> None:
    """Режим --mobile: каждый второй кадр -> assets/frames-m + poster-m.webp."""
    sel = files[::2]
    if OUT_DIR_M.exists():
        for f in OUT_DIR_M.glob("*.webp"):
            f.unlink()
    OUT_DIR_M.mkdir(parents=True, exist_ok=True)

    print(f"\n[mobile] {len(sel)} кадров из {len(files)} (каждый второй), "
          f"высота {FRAME_H_M}, quality={quality}")
    total = 0
    for i, p in enumerate(sel):
        rgb = np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
        out, alpha, _R, _wide = matte(rgb)
        out, alpha = downscale(out, alpha, FRAME_H_M)
        rgba = np.dstack([out, alpha * 255]).astype(np.uint8)
        img = Image.fromarray(rgba, "RGBA")
        dst = OUT_DIR_M / f"f{i:03d}.webp"
        img.save(dst, "WEBP", quality=quality, method=6)
        total += dst.stat().st_size
        if i == 0:
            img.save(POSTER_M, "WEBP", quality=max(quality, 82), method=6)
        if i % 20 == 0:
            print(f"    ...{i + 1}/{len(sel)}", flush=True)

    verify_frame(OUT_DIR_M / "f000.webp", "мобильный старт, общий план")
    verify_frame(OUT_DIR_M / f"f{len(sel)-1:03d}.webp", "мобильный финал, крупный план")
    verify_edges(OUT_DIR_M / "f000.webp")
    verify_edges(OUT_DIR_M / f"f{min(45, len(sel)-1):03d}.webp")

    mb = total / 1024 / 1024
    ps = POSTER_M.stat().st_size / 1024
    print(f"\n[итог-m] кадров: {len(sel)}  ({OUT_DIR_M.relative_to(ROOT)}/f000..f{len(sel)-1:03d}.webp)")
    print(f"[итог-m] размер: {mb:.2f} MB (бюджет {budget} MB) — "
          f"{'OK' if mb <= budget else 'ПРЕВЫШЕН, взять --mobile-quality 70, затем FRAME_H_M 560'}")
    print(f"[итог-m] poster-m.webp: {ps:.1f} KB "
          f"({Image.open(POSTER_M).size[0]}x{Image.open(POSTER_M).size[1]}) — "
          f"{'OK' if ps <= 60 else 'ПРЕВЫШЕН лимит 60 KB'}")
    print(f"[итог-m] средний кадр: {total / len(sel) / 1024:.1f} KB")
    if mb > budget:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=78)
    ap.add_argument("--step", type=int, default=2, help="каждый N-й кадр исходника")
    ap.add_argument("--budget", type=float, default=8.0, help="бюджет кадров, MB")
    ap.add_argument("--verify", action="store_true", help="только проверить готовые кадры")
    ap.add_argument("--mobile", action="store_true",
                    help="собрать лёгкий набор assets/frames-m (каждый второй кадр)")
    ap.add_argument("--mobile-quality", type=int, default=74)
    ap.add_argument("--mobile-budget", type=float, default=2.5, help="бюджет frames-m, MB")
    args = ap.parse_args()

    if args.verify:
        verify_outputs()
        return

    files = extract_png(args.step)
    if not files:
        sys.exit("нет PNG кадров")

    if args.mobile:
        build_mobile(files, args.mobile_quality, args.mobile_budget)
        return

    if OUT_DIR.exists():
        for f in OUT_DIR.glob("*.webp"):
            f.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    print(f"\n[matte] BG={BG.astype(int)} T_FLOOD={T_FLOOD} T_HARD={T_HARD} "
          f"FEATHER={FEATHER} A_LO={A_LO} A_HI={A_HI}")
    print(f"[fade]  низ={BOTTOM_FADE:.0%} бока={LR_FADE:.0%} верх={TOP_FADE:.0%} "
          f"(ворота: touch>{EDGE_TOUCH}, dilate={EDGE_DILATE}px, sigma={EDGE_SIGMA})")

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
