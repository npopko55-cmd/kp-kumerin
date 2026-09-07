#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build-objects.py — вырезание фона у четырёх 3D-рендеров объектов в прозрачность.

Зачем: на сайте объекты стоят в белых карточках, светло-серый студийный фон
рендера читается грязным квадратом. Нужен объект, парящий на карточке.

Подход (тот же, что в build-frames.py, но фон здесь ГРАДИЕНТНЫЙ, не константа):
  1. Модель фона B(x,y): полином 3-й степени по каждому каналу, обученный
     в два прохода — сначала по рамке 48 px, потом по всей найденной фоновой
     области. Замер: фон идёт 222 (левый верх) -> 243 (правый низ), одной
     константы BG, как в build-frames.py, не хватает.
  2. Шум рендера (p99 остатка 9-10 уровней) сглаживается gaussian(SIGMA_D)
     ТОЛЬКО для карты расстояний d; цвет пишется из исходника.
     SIGMA_D=2.8, а не 1.4: при 1.4 у obj-4 заливка просачивалась тонким
     каналом через правый верх рамки телефона и выедала 14 тыс. px экрана
     рваной кляксой. 2.8 замазывает канал; силуэт при этом сдвигается на
     ~1 px (потери 6.5-11 тыс. px, все в кромке 3 px), тонкие детали —
     стилус, перфорация плёнки, зелёные квадраты — целы.
  3. Фон = ЗАЛИВКА ОТ ГРАНИЦ по d <= T_CORE. Не глобальный color key:
     белый экран телефона даёт 27 тыс. px «фоновых» по цвету, а при T=14 —
     221 тыс.; они не связаны с рамкой и потому уцелевают.
  4. Мягкая тень под объектом темнее фона на 5..20 уровней, малонасыщена
     и светлая. Второй проход заливки расширяется на пиксели
     S < S_MAX и V > V_MIN и -T_SHADOW <= ds <= T_BRIGHT (ds — знаковое
     отклонение яркости от модели фона), НО только в полосе пола.
     Полоса выводится из данных: последняя строка, где ещё есть >= DEEP_ROW
     «глубоких» пикселей (ds < -DEEP), плюс запас BAND_PAD. Замерено:
     низ объекта 1560/1642/1550/1660 из 2048, тень начинается ниже.
     Без полосы правило S/V/ds съедает матовое стекло планшета и белый
     корпус ноутбука — у них ds тоже в окне -26..+14 (проверено: 148 и
     158 тыс. px объекта). Ограничить заливку силой края нельзя: контуры
     здесь мягкие, p25 |grad| на контуре 0.24..1.7 против 0.3 шума фона.
  5. Мягкая альфа в полосе FEATHER вокруг фоновой области + деконтаминация
     цвета по модели фона.
  6. Кадрирование по непрозрачным пикселям с полем 6 %, квадрат 1200x1200,
     WebP q82 method 6 -> assets/img/obj-N-*-cut.webp.
  7. Числовая приёмка + контактный лист .build/objects-contact.png.

Запуск:  .venv/bin/python build-objects.py [--verify] [--quality 82]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "assets"
OUT_DIR = ROOT / "assets" / "img"
BUILD = ROOT / ".build"
CONTACT = BUILD / "objects-contact.png"

OBJECTS = ["obj-1-manager", "obj-2-design", "obj-3-video", "obj-4-site"]

# --- параметры матта (общие; per-image переопределения в OVERRIDE) ---
SIGMA_D = 2.8      # сглаживание шума рендера перед картой расстояний
POLY = 3           # степень полинома черновой модели фона
CELL = 32          # сторона ячейки сеточной модели фона, px
MARGIN = 96        # отступ от любого не-фона, ячейки ближе в модель не идут
INPAINT_ITERS = 60 # итераций диффузии при достройке пустых ячеек
T_CORE = 4         # допуск заливки от границ, уровней 0..255 (1.6 %)
T_SHADOW = 26      # насколько темнее фона может быть пиксель тени (10 %)
T_BRIGHT = 14      # насколько светлее фона (блик отражённого света на полу)
S_MAX = 0.12       # порог насыщенности для «фон/тень»
V_MIN = 0.72       # порог яркости для «фон/тень»
DEEP = 25          # ds < -DEEP -> пиксель заведомо объект, не тень
DEEP_ROW = 4       # столько глубоких пикселей в строке -> строка ещё объект
BAND_PAD = 10      # запас вниз от последней строки объекта, px
BAND_SPAN = 64     # полуокно бегущего максимума по столбцам, px
JUNK_MAX = 60000   # компонента меньше этого и «плоская» по цвету -> фон
FEATHER = 3        # полоса мягкой альфы вглубь силуэта, px (исходный масштаб)
A_LO = 3           # d <= A_LO -> alpha 0
A_HI = 22          # d >= A_HI -> alpha 1
BLUR = 0.8         # сглаживание альфы, ~1 px
MIN_BLOB = 600     # непрозрачные компоненты меньше — шум дизеринга, стираем
PAD = 0.06         # поле при кадрировании
SIDE = 1200        # сторона выходного квадрата

OVERRIDE: dict[str, dict] = {}


def poly_basis(x, y, order):
    return np.stack([(x ** i) * (y ** j)
                     for i in range(order + 1) for j in range(order + 1 - i)], axis=-1)


def fit_bg(rgb, mask, order=POLY):
    """Полиномиальная модель фона по каналам, обученная на mask."""
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A_all = poly_basis((xx / w).astype(np.float32), (yy / h).astype(np.float32), order)
    A = A_all[mask]
    model = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, rgb[..., c][mask], rcond=None)
        model[..., c] = A_all @ coef
    return model


def edge_region(mask):
    """Компоненты маски, связные с рамкой кадра (заливка ОТ ГРАНИЦ, не color key)."""
    lab, _ = ndimage.label(mask)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    edge.discard(0)
    return np.isin(lab, list(edge))


def sat_val(rgb):
    mx = rgb.max(axis=2) / 255.0
    mn = rgb.min(axis=2) / 255.0
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0), mx


def background_model(sm, cell=CELL, margin=MARGIN, iters=INPAINT_ITERS):
    """Модель фона в три шага.

    1) грубый полином по рамке -> черновая фоновая область R0;
    2) «безопасный фон» = R0 минус полоса margin вокруг всего не-фона —
       так в модель не попадают ни объект, ни его тень, ни ладонная
       площадка ноутбука, которая по яркости равна фону;
    3) медиана по ячейкам cell x cell безопасного фона, пустые ячейки
       (весь объект целиком) достраиваются диффузией Лапласа и
       разгоняются кубическим zoom.

    Зачем не полином: у полинома 3-й степени остаток в фоне p99 = 10
    уровней, и допуск приходится держать на 7, а тогда заливка
    просачивается в белые плоскости. У сеточной модели p99 = 1.6-2.4,
    допуск опускается до 4, ладонная площадка (d = 8) уцелевает.
    """
    h, w, _ = sm.shape
    frame = np.zeros((h, w), bool)
    frame[:48] = frame[-48:] = True
    frame[:, :48] = frame[:, -48:] = True
    R0 = edge_region(np.abs(sm - fit_bg(sm, frame, order=2)).max(axis=2) <= 10)
    safe = R0 & ~ndimage.binary_dilation(~R0, iterations=margin)

    gh, gw = h // cell, w // cell
    m = safe[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
    known = m.sum(axis=2) >= cell * cell * 0.5
    med = np.zeros((gh, gw, 3), np.float32)
    for c in range(3):
        v = sm[..., c][:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
        s = np.where(m, v, 0.0).sum(axis=2)
        n = np.maximum(m.sum(axis=2), 1)
        med[..., c] = s / n
    if not known.any():
        return fit_bg(sm, R0 & (np.random.RandomState(0).rand(h, w) < 0.05), order=POLY)

    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    f = med[idx[0], idx[1]].astype(np.float32)
    for _ in range(iters):
        f = ndimage.gaussian_filter(f, (1.5, 1.5, 0))
        f[known] = med[known]
    B = np.dstack([ndimage.zoom(f[..., c], (h / gh, w / gw), order=3) for c in range(3)])
    return B[:h, :w]


def matte(raw, p):
    """RGB float 0..255 -> (rgb деконтаминированный, alpha 0..1, диагностика)."""
    sm = np.dstack([ndimage.gaussian_filter(raw[..., c], p["sigma"]) for c in range(3)])
    B = background_model(sm)
    d = np.abs(sm - B).max(axis=2)
    ds = sm.mean(axis=2) - B.mean(axis=2)      # < 0 = темнее фона (тень)
    S, V = sat_val(sm)

    h, w = d.shape
    core = d <= p["t_core"]
    R_core = edge_region(core)

    # Полоса пола, ПОКОЛОНОЧНО: в каждом столбце тень начинается ниже
    # последнего «глубокого» пикселя объекта. Одна горизонтальная линия
    # на всю ширину оставляла бы серые куски тени слева от плёнки (obj-3)
    # и под карточкой (obj-4) — там объекта в столбце нет вовсе.
    deep = (~R_core) & (ds < -p["deep"])
    yy = np.arange(h)[:, None]
    bottom = np.where(deep.any(axis=0), np.max(np.where(deep, yy, -1), axis=0), -1)
    # бегущий максимум: у соседних с объектом столбцов граница не подскакивает
    bottom = ndimage.maximum_filter1d(bottom, size=2 * p["band_span"] + 1, mode="nearest")
    band = np.arange(h)[:, None] > (bottom + p["band_pad"])[None, :]
    band_top = int(band.argmax(axis=0).min())

    # тень/отражённый блик: малонасыщенные светлые пиксели рядом с яркостью фона
    soft_bg = ((S < p["s_max"]) & (V > p["v_min"]) &
               (ds >= -p["t_shadow"]) & (ds <= p["t_bright"]) & band)
    R = edge_region(core | soft_bg)

    # страховка: ядро объекта (d заведомо большое) не должно попасть в фон
    leak = int((R & (d > 45)).sum())

    # оставшиеся куски: мелочь дизеринга и «плоские» по цвету пятна пола — в фон
    obj = ~R
    lab, n = ndimage.label(obj)
    killed = 0
    if n:
        sizes = ndimage.sum(obj, lab, range(1, n + 1))
        drop = []
        for i in range(n):
            if sizes[i] < p["min_blob"]:
                drop.append(i + 1)
                continue
            if sizes[i] >= p["junk_max"]:
                continue
            m = (lab == i + 1)
            if (np.percentile(ds[m], 2) > -p["t_shadow"] and
                    np.median(S[m]) < p["s_max"] and np.median(V[m]) > p["v_min"]):
                drop.append(i + 1)
        if drop:
            small = np.isin(lab, drop)
            killed = int(small.sum())
            R = R | small

    # мягкая альфа только в полосе FEATHER вокруг фона: край объекта получает
    # честное частичное покрытие, а белые плоскости внутри остаются целыми
    dist = ndimage.distance_transform_edt(~R)
    soft = np.clip((d - p["a_lo"]) / (p["a_hi"] - p["a_lo"]), 0.0, 1.0).astype(np.float32)
    alpha = np.ones(d.shape, dtype=np.float32)
    band = (dist > 0) & (dist <= p["feather"])
    alpha[band] = soft[band]
    alpha[R] = 0.0
    alpha = ndimage.gaussian_filter(alpha, p["blur"])
    alpha[R & (d <= p["t_core"])] = 0.0        # чистый фон — жёсткий ноль

    # деконтаминация цвета: C_fg = (C - (1-a)*B) / a, B — локальная модель фона
    out = raw.copy()
    sel = (alpha > 0.04) & (alpha < 0.96)
    a = alpha[sel][:, None]
    out[sel] = np.clip((raw[sel] - (1 - a) * B[sel]) / a, 0, 255)

    diag = dict(bg_frac=float(R.mean()), leak=leak, killed=killed, band_top=band_top,
                bg_tl=B[0, 0].astype(int).tolist(), bg_br=B[-1, -1].astype(int).tolist(),
                core_frac=float(R_core.mean()), shadow_gain=float(R.mean() - R_core.mean()))
    return out, alpha, diag


def crop_square(rgb, alpha, pad=PAD, side=SIDE):
    """Кадрирование по непрозрачным пикселям с полем pad, квадрат side x side."""
    solid = alpha > 0.125
    ys, xs = np.nonzero(solid)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bw, bh = x1 - x0, y1 - y0
    s = int(round(max(bw, bh) * (1 + 2 * pad)))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left, top = int(round(cx - s / 2)), int(round(cy - s / 2))

    # премультипликация: иначе LANCZOS затянет серый фон в кромку
    canvas = np.zeros((s, s, 4), np.float32)
    h, w = alpha.shape
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + s), min(h, top + s)
    dx0, dy0 = sx0 - left, sy0 - top
    a = alpha[sy0:sy1, sx0:sx1]
    canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0), :3] = rgb[sy0:sy1, sx0:sx1] * a[..., None]
    canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0), 3] = a * 255.0

    im = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGBA")
    im = im.resize((side, side), Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32)
    al = arr[..., 3:4] / 255.0
    col = np.where(al > 0.002, arr[..., :3] / np.maximum(al, 0.002), 0.0)
    outa = np.clip(np.dstack([col, arr[..., 3]]), 0, 255).astype(np.uint8)
    return Image.fromarray(outa, "RGBA"), (int(bw), int(bh), s)


def verify(path):
    """Числа приёмки на ГОТОВОМ webp."""
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im)
    al = arr[..., 3]
    h, w = al.shape
    corners = [al[0, 0], al[0, w - 1], al[h - 1, 0], al[h - 1, w - 1]]

    solid = al > 200
    # «центр объекта» — самая глубокая точка силуэта, не выдуманная координата
    dt = ndimage.distance_transform_edt(solid)
    cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    centre = int(al[cy, cx])

    near = ndimage.binary_dilation(solid, ndimage.generate_binary_structure(2, 2), iterations=12)
    junk = ((al >= 20) & (al <= 120) & ~near)
    junk_frac = float(junk.sum()) / al.size

    # дырки: прозрачные компоненты внутри силуэта
    holes = al < 128
    lab, n = ndimage.label(holes)
    inner = []
    if n:
        edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        edge.discard(0)
        sizes = ndimage.sum(holes, lab, range(1, n + 1))
        inner = sorted((int(s) for j, s in enumerate(sizes) if (j + 1) not in edge), reverse=True)

    kb = path.stat().st_size / 1024
    return dict(size=(w, h), corners=[int(c) for c in corners], centre=centre,
                centre_xy=(int(cx), int(cy)), junk=junk_frac, kb=kb,
                holes=[s for s in inner if s > 60], opaque=float(solid.mean()))


def contact_sheet(files):
    """2 ряда x 4: объект на белом и на кислотно-зелёном #C2F53F."""
    cell = 460
    pad = 12
    sheet = Image.new("RGB", (cell * 4 + pad * 5, cell * 2 + pad * 3), (128, 128, 128))
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGBA").resize((cell, cell), Image.LANCZOS)
        for r, bg in enumerate([(255, 255, 255), (194, 245, 63)]):
            tile = Image.new("RGB", (cell, cell), bg)
            tile.paste(im, (0, 0), im)
            sheet.paste(tile, (pad + i * (cell + pad), pad + r * (cell + pad)))
    BUILD.mkdir(parents=True, exist_ok=True)
    sheet.save(CONTACT)
    return CONTACT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--verify", action="store_true", help="только проверить готовые файлы")
    args = ap.parse_args()

    outs = [OUT_DIR / f"{n}-cut.webp" for n in OBJECTS]

    if not args.verify:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, dst in zip(OBJECTS, outs):
            src = SRC_DIR / f"{name}.png"
            if not src.exists():
                sys.exit(f"нет исходника {src}")
            p = dict(sigma=SIGMA_D, t_core=T_CORE, t_shadow=T_SHADOW, t_bright=T_BRIGHT,
                     s_max=S_MAX, v_min=V_MIN, feather=FEATHER, a_lo=A_LO, a_hi=A_HI,
                     blur=BLUR, min_blob=MIN_BLOB, deep=DEEP, deep_row=DEEP_ROW,
                     band_pad=BAND_PAD, band_span=BAND_SPAN, junk_max=JUNK_MAX,
                     quality=args.quality)
            p.update(OVERRIDE.get(name, {}))
            raw = np.asarray(Image.open(src).convert("RGB")).astype(np.float32)
            rgb, alpha, diag = matte(raw, p)
            img, (bw, bh, s) = crop_square(rgb, alpha)
            img.save(dst, "WEBP", quality=p["quality"], method=6)
            print(f"[{name}] T_CORE={p['t_core']} T_SHADOW={p['t_shadow']} "
                  f"полоса пола y>={diag['band_top']}  фон={diag['bg_frac']:.4f} "
                  f"(+тень {diag['shadow_gain']:.4f}) утечка в объект={diag['leak']} px  "
                  f"снято ошмётков={diag['killed']} px")
            print(f"          модель фона TL={diag['bg_tl']} BR={diag['bg_br']}; "
                  f"bbox {bw}x{bh} -> квадрат {s} -> {SIDE}")

    print(f"\n=== ПРИЁМКА (quality={args.quality}, method=6) ===")
    rows = []
    for name, f in zip(OBJECTS, outs):
        if not f.exists():
            print(f"  {name}: НЕТ ФАЙЛА")
            continue
        v = verify(f)
        ok_c = all(c == 0 for c in v["corners"])
        ok_ctr = v["centre"] == 255
        ok_junk = v["junk"] < 0.003
        ok_kb = v["kb"] <= 140
        ok = ok_c and ok_ctr and ok_junk and ok_kb
        rows.append((name, ok))
        print(f"  {f.name}  {v['size'][0]}x{v['size'][1]}")
        print(f"    углы alpha = {v['corners']}  -> {'OK' if ok_c else 'FAIL'}")
        print(f"    центр объекта {v['centre_xy']} alpha={v['centre']} -> {'OK' if ok_ctr else 'FAIL'}"
              f"   (непрозрачных {v['opaque']*100:.1f}% кадра)")
        print(f"    мусор alpha 20..120 вне 12px полосы = {v['junk']*100:.4f}% -> "
              f"{'OK' if ok_junk else 'FAIL'} (лимит 0.30%)")
        print(f"    размер {v['kb']:.1f} KB -> {'OK' if ok_kb else 'FAIL'} (лимит 140 KB)")
        print(f"    дырки внутри силуэта >60px: {v['holes'][:5] if v['holes'] else '—'}")
        print(f"    ИТОГ: {'OK' if ok else 'FAIL'}")

    c = contact_sheet([f for f in outs if f.exists()])
    print(f"\n[контактный лист] {c}")
    if rows and not all(ok for _, ok in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
