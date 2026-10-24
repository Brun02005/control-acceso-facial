"""
BENCHMARK LFW - Mide FAR/FRR de buffalo_l sobre LFW y recomienda el umbral.

Por que LFW
-----------
Para fijar un umbral de ACCESO con fundamento hace falta medir el FAR (cuantos
impostores pasan) con MUCHAS identidades distintas. Un puñado de compañeros no
alcanza. LFW (Labeled Faces in the Wild) tiene ~13.000 fotos de ~5.700 personas
y es el benchmark estandar de verificacion facial.

Que hace
--------
1) Descarga LFW (deepfunneled, ~110 MB) una sola vez.
2) Pasa cada imagen por el MISMO motor (InsightFace buffalo_l) -> embedding 512-d.
   Cachea los embeddings en disco: si se corta, retoma donde iba.
3) Arma pares:
     - GENUINOS   = dos fotos de la MISMA persona   -> FRR
     - IMPOSTORES = fotos de personas DISTINTAS      -> FAR
   (muchos miles, muestreados de todas las identidades)
4) Barre umbrales, calcula FAR/FRR/EER y recomienda el umbral para un FAR objetivo.

Uso
---
    python benchmark_lfw.py                 # descarga + embebe + mide (~20-30 min CPU)
    python benchmark_lfw.py --descargar-solo
    python benchmark_lfw.py --limit 60      # prueba rapida (pocas imagenes)
    python benchmark_lfw.py --far-objetivo 0.001
    python benchmark_lfw.py --solo-medir    # usa el cache ya embebido

Dependencias: insightface onnxruntime opencv-contrib-python numpy requests
(matplotlib opcional para el grafico).
"""

import os
import sys
import time
import glob
import tarfile
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np
import cv2
from insightface.app import FaceAnalysis


_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "lfw_data")
IMG_ROOT = os.path.join(DATA_DIR, "lfw-deepfunneled")
TGZ_PATH = os.path.join(DATA_DIR, "lfw-deepfunneled.tgz")
CACHE = os.path.join(DATA_DIR, "lfw_embeddings.npz")
PLOT_PATH = os.path.join(_HERE, "calibracion_lfw.png")

# Mirrors de descarga (se prueban en orden).
TGZ_URLS = [
    "https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
    "https://ndownloader.figshare.com/files/5976015",   # mirror figshare
]

MODEL_NAME = "buffalo_l"
DET_SIZE = (320, 320)          # LFW ya viene centrada -> 320 alcanza y es rapido
MAX_GENUINOS = 120_000
MAX_IMPOSTORES = 600_000
PRINT_EVERY = 50               # mostrar progreso cada N imagenes
SAVE_EVERY = 300               # guardar cache cada N imagenes


# ---------------------------------------------------------------------------
# DESCARGA
# ---------------------------------------------------------------------------

def detectar_img_root():
    """Encuentra la carpeta que contiene <persona>/<foto>.jpg (sea cual sea el
    nombre que le haya puesto el mirror al extraer)."""
    if glob.glob(os.path.join(IMG_ROOT, "*", "*.jpg")):
        return IMG_ROOT
    for d in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*", "*.jpg")):
            return d
    hits = glob.glob(os.path.join(DATA_DIR, "**", "*.jpg"), recursive=True)
    if hits:
        return os.path.dirname(os.path.dirname(hits[0]))
    return None


def descargar():
    os.makedirs(DATA_DIR, exist_ok=True)
    if detectar_img_root():
        print(f"LFW ya esta en {detectar_img_root()}")
        return True

    if not (os.path.exists(TGZ_PATH) and os.path.getsize(TGZ_PATH) > 50_000_000):
        import requests
        ok = False
        for url in TGZ_URLS:
            try:
                print(f"Descargando LFW desde {url} ...")
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0))
                    done = 0
                    with open(TGZ_PATH, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                            done += len(chunk)
                            if total:
                                pct = done * 100 // total
                                print(f"\r  {done//1_000_000} / {total//1_000_000} MB "
                                      f"({pct}%)", end="", flush=True)
                print()
                ok = True
                break
            except Exception as exc:
                print(f"\n  fallo: {exc}")
        if not ok:
            print("No se pudo descargar LFW automaticamente. Descargalo a mano:")
            print("  https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz")
            print(f"  y dejalo en: {TGZ_PATH}")
            return False

    print("Extrayendo ...")
    with tarfile.open(TGZ_PATH, "r:gz") as tar:
        tar.extractall(DATA_DIR, filter="data")   # filter=data: extraccion segura
    print(f"Listo en {IMG_ROOT}")
    return True


# ---------------------------------------------------------------------------
# EMBEDDINGS  (con cache reanudable)
# ---------------------------------------------------------------------------

def cargar_cache():
    if not os.path.exists(CACHE):
        return {}
    data = np.load(CACHE)
    keys = data["keys"]
    embs = data["embs"]
    return {str(k): embs[i] for i, k in enumerate(keys)}


def guardar_cache(cache):
    keys = np.array(list(cache.keys()), dtype="U160")
    embs = np.array(list(cache.values()), dtype=np.float32)
    np.savez(CACHE, keys=keys, embs=embs)


def embeber(limit=None):
    app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)

    root = detectar_img_root()
    if not root:
        print("No encuentro las imagenes de LFW. Corré la descarga primero.")
        return {}
    imgs = sorted(glob.glob(os.path.join(root, "*", "*.jpg")))
    if limit:
        imgs = imgs[:limit]
    cache = cargar_cache()
    print(f"Imagenes: {len(imgs)} | ya en cache: {len(cache)}")

    t0 = time.time()
    hechas = 0
    fallos = 0
    for n, path in enumerate(imgs, 1):
        rel = os.path.relpath(path, root).replace("\\", "/")
        if rel in cache:
            continue
        img = cv2.imread(path)
        if img is None:
            fallos += 1
            continue
        faces = app.get(img)
        if not faces:
            fallos += 1
            continue
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        cache[rel] = face.normed_embedding.astype(np.float32)
        hechas += 1
        if hechas % PRINT_EVERY == 0:
            vel = hechas / max(1e-6, time.time() - t0)
            faltan = len(imgs) - n
            eta = faltan / max(1e-6, vel)
            print(f"\r  embebidas {len(cache)} | {vel:.1f} img/s | "
                  f"ETA {eta/60:.1f} min | fallos {fallos}   ", end="", flush=True)
        if hechas % SAVE_EVERY == 0:
            guardar_cache(cache)
    guardar_cache(cache)
    print(f"\nEmbeddings listos: {len(cache)} (fallos de deteccion: {fallos})")
    return cache


# ---------------------------------------------------------------------------
# METRICAS
# ---------------------------------------------------------------------------

def sims_de_pares(embs, I, J, chunk=100_000):
    """Similitud coseno (producto punto) de pares (I[k], J[k]) por chunks."""
    out = np.empty(len(I), dtype=np.float32)
    for s in range(0, len(I), chunk):
        e = s + chunk
        out[s:e] = np.einsum("ij,ij->i", embs[I[s:e]], embs[J[s:e]])
    return out


def medir(far_objetivo):
    cache = cargar_cache()
    if len(cache) < 100:
        print("Cache insuficiente. Corré primero la descarga + embebido.")
        return

    keys = list(cache.keys())
    embs = np.array([cache[k] for k in keys], dtype=np.float32)
    # identidad = carpeta (nombre de la persona)
    ids = np.array([k.split("/")[0] for k in keys])
    identidades, inv = np.unique(ids, return_inverse=True)

    # Indices por identidad.
    por_id = {}
    for i, lab in enumerate(inv):
        por_id.setdefault(lab, []).append(i)
    con_multiple = [v for v in por_id.values() if len(v) >= 2]

    rng = np.random.RandomState(0)

    # --- Pares GENUINOS: dentro de cada persona con >=2 fotos ---
    gi, gj = [], []
    for idxs in con_multiple:
        a = np.array(idxs)
        ii, jj = np.triu_indices(len(a), k=1)
        gi.append(a[ii]); gj.append(a[jj])
    gi = np.concatenate(gi) if gi else np.array([], int)
    gj = np.concatenate(gj) if gj else np.array([], int)
    if len(gi) > MAX_GENUINOS:
        sel = rng.choice(len(gi), MAX_GENUINOS, replace=False)
        gi, gj = gi[sel], gj[sel]

    # --- Pares IMPOSTORES: muestreo aleatorio de identidades distintas ---
    N = len(embs)
    ii = rng.randint(0, N, MAX_IMPOSTORES * 2)
    jj = rng.randint(0, N, MAX_IMPOSTORES * 2)
    dist = inv[ii] != inv[jj]
    ii, jj = ii[dist][:MAX_IMPOSTORES], jj[dist][:MAX_IMPOSTORES]

    if len(gi) == 0:
        print("No hay pares genuinos (ninguna persona con >=2 fotos).")
        return

    genuinos = sims_de_pares(embs, gi, gj)
    impostores = sims_de_pares(embs, ii, jj)

    # --- Barrido de umbrales ---
    ths = np.round(np.arange(0.20, 0.701, 0.01), 2)
    far = np.array([(impostores >= t).mean() for t in ths])
    frr = np.array([(genuinos < t).mean() for t in ths])

    eer_i = int(np.argmin(np.abs(far - frr)))
    eer = (far[eer_i] + frr[eer_i]) / 2

    cumplen = np.where(far <= far_objetivo)[0]
    rec_i = int(cumplen[0]) if len(cumplen) else int(np.argmin(far))
    recomendado = ths[rec_i]

    print("\n" + "=" * 64)
    print("  BENCHMARK LFW  -  buffalo_l")
    print("=" * 64)
    print(f"  Imagenes: {N}  |  identidades: {len(identidades)}")
    print(f"  Pares genuinos: {len(genuinos):>9,}  (sim media {genuinos.mean():.3f})")
    print(f"  Pares impostor: {len(impostores):>9,}  (sim media {impostores.mean():.3f})")
    print("-" * 64)
    print(f"  {'umbral':>7} {'FAR':>12} {'FRR':>10}")
    for t, a, r in zip(ths, far, frr):
        marca = "  <-- EER" if t == ths[eer_i] else ""
        marca += "  <<< RECOMENDADO" if t == recomendado else ""
        if t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60) or marca:
            print(f"  {t:>7.2f} {a*100:>11.4f}% {r*100:>9.3f}%{marca}")
    print("-" * 64)
    print(f"  EER (FAR=FRR): {eer*100:.3f}%  en umbral {ths[eer_i]:.2f}")
    print(f"  Objetivo FAR <= {far_objetivo*100:.4f}%")
    print(f"  >>> UMBRAL RECOMENDADO: {recomendado:.2f}  "
          f"(FAR {far[rec_i]*100:.4f}% / FRR {frr[rec_i]*100:.3f}%)")
    print("=" * 64)
    print(f"  Aplicalo en control_acceso.py:  RECOGNIZE_SIM = {recomendado:.2f}")
    print("=" * 64 + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        plt.semilogy(ths, far * 100 + 1e-6, label="FAR (impostores aceptados)",
                     color="#B23B2E")
        plt.semilogy(ths, frr * 100 + 1e-6, label="FRR (autorizados rechazados)",
                     color="#1C6B57")
        plt.axvline(recomendado, color="#A46036", ls="--",
                    label=f"recomendado {recomendado:.2f}")
        plt.xlabel("umbral de similitud")
        plt.ylabel("tasa (%) - escala log")
        plt.title("Benchmark LFW - buffalo_l")
        plt.legend()
        plt.grid(alpha=.3, which="both")
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=120)
        print(f"Grafico guardado en: {PLOT_PATH}\n")
    except Exception as exc:
        print(f"(No se pudo guardar el grafico: {exc})\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark LFW para calibrar umbral.")
    ap.add_argument("--descargar-solo", action="store_true")
    ap.add_argument("--solo-medir", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="limitar imagenes (prueba)")
    ap.add_argument("--far-objetivo", type=float, default=0.001)
    args = ap.parse_args()

    if args.solo_medir:
        medir(args.far_objetivo)
        return

    if not descargar():
        return
    if args.descargar_solo:
        return

    embeber(limit=args.limit)
    medir(args.far_objetivo)


if __name__ == "__main__":
    main()
