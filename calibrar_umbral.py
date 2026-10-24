"""
CALIBRAR UMBRAL - Mide FAR/FRR para elegir RECOGNIZE_SIM con datos, no a ojo.

Problema que resuelve
---------------------
En control_acceso.py el umbral de similitud (RECOGNIZE_SIM) estaba puesto a ojo
(0.50). Para un control de acceso eso es el riesgo central: no sabes tu FAR
(false accept rate: impostores que dejas pasar) ni tu FRR (false reject rate:
autorizados que no reconoces). Esta herramienta los MIDE sobre caras reales y
recomienda el umbral para un objetivo de FAR.

Como funciona
-------------
1) Capturas embeddings de VARIAS personas (varias tomas c/u) con el mismo motor
   InsightFace que usa la app -> se guardan en dataset_embeddings.npz.
2) Mide compara todos los pares:
     - GENUINOS   = misma persona   -> definen el FRR
     - IMPOSTORES = personas distintas -> definen el FAR
   barre umbrales y, en cada uno, calcula FAR y FRR. Elige el umbral mas bajo
   cuyo FAR <= objetivo (por defecto 0.1%).

Uso
---
    # capturar (repetir por cada persona; ideal 2-3 tomas en luces distintas)
    python calibrar_umbral.py capturar "Juan Perez"
    python calibrar_umbral.py capturar "Ana Gomez"
    ...
    # medir y obtener el umbral recomendado
    python calibrar_umbral.py medir
    python calibrar_umbral.py medir --far-objetivo 0.001

Dependencias: insightface onnxruntime opencv-contrib-python numpy  (matplotlib
opcional, solo para guardar el grafico).
"""

import os
import sys
import json
import time
import argparse
import warnings
import itertools

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
import numpy as np
from insightface.app import FaceAnalysis


_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_HERE, "dataset_embeddings.npz")
NOMBRES_PATH = os.path.join(_HERE, "dataset_nombres.json")
PLOT_PATH = os.path.join(_HERE, "calibracion.png")

MODEL_NAME = "buffalo_l"
DET_SIZE = (416, 416)
TARGET_SAMPLES = 40            # embeddings a capturar por toma
MAX_EMB_MEDIR = 4000           # cap para la matriz de similitud (memoria)


# ---------------------------------------------------------------------------
# MOTOR Y DATASET
# ---------------------------------------------------------------------------

def cargar_motor():
    print("Cargando motor de IA (InsightFace)...")
    app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    print("Motor listo.")
    return app


def cargar_dataset():
    """Devuelve (embeddings (N,512), labels (N,), nombres {id:nombre})."""
    if not os.path.exists(DATASET_PATH):
        return (np.empty((0, 512), np.float32), np.empty((0,), np.int32), {})
    data = np.load(DATASET_PATH)
    nombres = {}
    if os.path.exists(NOMBRES_PATH):
        with open(NOMBRES_PATH, "r", encoding="utf-8") as f:
            nombres = {int(k): v for k, v in json.load(f).items()}
    return (data["embeddings"].astype(np.float32),
            data["labels"].astype(np.int32), nombres)


def guardar_dataset(emb, labels, nombres):
    np.savez(DATASET_PATH, embeddings=emb, labels=labels)
    with open(NOMBRES_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in nombres.items()}, f, ensure_ascii=False)


def cara_mas_grande(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


# ---------------------------------------------------------------------------
# MODO CAPTURAR
# ---------------------------------------------------------------------------

def cmd_capturar(nombre):
    nombre = nombre.strip()
    if not nombre:
        print("Nombre vacio.")
        return
    app = cargar_motor()
    emb, labels, nombres = cargar_dataset()

    # id para este nombre (reusa si ya existe -> permite varias tomas)
    nombre_a_id = {v: k for k, v in nombres.items()}
    if nombre in nombre_a_id:
        pid = nombre_a_id[nombre]
        print(f"'{nombre}' ya existe (id {pid}); se suma otra toma.")
    else:
        pid = (max(nombres) + 1) if nombres else 0
        nombres[pid] = nombre

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la webcam.")
        return

    nuevos = []
    frame_idx = 0
    print(f"Capturando {TARGET_SAMPLES} muestras de '{nombre}'. "
          "Mové un poco la cabeza para variar angulo/luz. 'q' corta antes.")
    while len(nuevos) < TARGET_SAMPLES:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame_idx += 1
        if frame_idx % 2 == 0:
            face = cara_mas_grande(app.get(frame))
            if face is not None:
                nuevos.append(face.normed_embedding)
                x1, y1, x2, y2 = (int(v) for v in face.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 200, 90), 2)
        cv2.putText(frame, f"{nombre}: {len(nuevos)}/{TARGET_SAMPLES}",
                    (16, 34), cv2.FONT_HERSHEY_DUPLEX, 0.8, (90, 200, 90), 2)
        cv2.imshow("Capturar dataset", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    if not nuevos:
        print("No se capturó ningún rostro.")
        return
    nuevos = np.array(nuevos, dtype=np.float32)
    emb = np.vstack([emb, nuevos]) if len(emb) else nuevos
    labels = np.concatenate([labels, np.full(len(nuevos), pid, np.int32)])
    guardar_dataset(emb, labels, nombres)

    n_ids = len(set(labels.tolist()))
    print(f"Guardadas {len(nuevos)} muestras de '{nombre}'.")
    print(f"Dataset: {len(emb)} embeddings, {n_ids} persona(s): "
          f"{', '.join(nombres[i] for i in sorted(set(labels.tolist())))}")
    if n_ids < 2:
        print("AVISO: para medir FAR necesitas AL MENOS 2 personas distintas.")


# ---------------------------------------------------------------------------
# MODO MEDIR
# ---------------------------------------------------------------------------

def cmd_medir(far_objetivo):
    emb, labels, nombres = cargar_dataset()
    n_ids = len(set(labels.tolist()))
    if len(emb) == 0:
        print("No hay dataset. Capturá personas primero: "
              "python calibrar_umbral.py capturar \"Nombre\"")
        return
    if n_ids < 2:
        print(f"Solo hay {n_ids} persona(s). Para medir FAR necesitás >= 2.")
        return

    # Subsample si hay demasiados embeddings (la matriz es NxN).
    if len(emb) > MAX_EMB_MEDIR:
        idx = np.random.RandomState(0).choice(len(emb), MAX_EMB_MEDIR, replace=False)
        emb, labels = emb[idx], labels[idx]

    # Matriz de similitud (embeddings normalizados -> coseno = producto punto).
    S = emb @ emb.T
    same = labels[:, None] == labels[None, :]
    iu = np.triu_indices(len(emb), k=1)      # solo pares i<j, sin la diagonal
    sims = S[iu]
    es_genuino = same[iu]

    genuinos = sims[es_genuino]
    impostores = sims[~es_genuino]
    if len(genuinos) == 0 or len(impostores) == 0:
        print("Faltan pares genuinos o impostores (revisá los datos).")
        return

    # Barrido de umbrales.
    ths = np.round(np.arange(0.20, 0.701, 0.01), 2)
    far = np.array([(impostores >= t).mean() for t in ths])   # impostor aceptado
    frr = np.array([(genuinos < t).mean() for t in ths])      # genuino rechazado

    # EER: donde FAR y FRR se cruzan.
    eer_i = int(np.argmin(np.abs(far - frr)))
    eer = (far[eer_i] + frr[eer_i]) / 2

    # Umbral recomendado: el mas bajo que cumple el objetivo de FAR.
    cumplen = np.where(far <= far_objetivo)[0]
    if len(cumplen):
        rec_i = int(cumplen[0])
        recomendado = ths[rec_i]
    else:
        rec_i = int(np.argmin(far))
        recomendado = ths[rec_i]

    # ----- Reporte -----
    print("\n" + "=" * 60)
    print(f"  CALIBRACION DE UMBRAL  (buffalo_l)")
    print("=" * 60)
    print(f"  Personas: {n_ids}  |  embeddings: {len(emb)}")
    print(f"  Pares genuinos: {len(genuinos):>8}  (sim media {genuinos.mean():.3f})")
    print(f"  Pares impostor: {len(impostores):>8}  (sim media {impostores.mean():.3f})")
    print("-" * 60)
    print(f"  {'umbral':>7} {'FAR':>10} {'FRR':>10}")
    for t, a, r in zip(ths, far, frr):
        marca = "  <-- EER" if t == ths[eer_i] else ""
        marca += "  <<< RECOMENDADO" if t == recomendado else ""
        if t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60) or marca:
            print(f"  {t:>7.2f} {a*100:>9.3f}% {r*100:>9.3f}%{marca}")
    print("-" * 60)
    print(f"  EER (FAR=FRR): {eer*100:.2f}% en umbral {ths[eer_i]:.2f}")
    print(f"  Objetivo FAR <= {far_objetivo*100:.3f}%")
    print(f"  >>> UMBRAL RECOMENDADO: {recomendado:.2f}  "
          f"(FAR {far[rec_i]*100:.3f}% / FRR {frr[rec_i]*100:.3f}%)")
    print("=" * 60)
    print(f"  Aplicalo en control_acceso.py:  RECOGNIZE_SIM = {recomendado:.2f}")
    if n_ids < 5:
        print("  NOTA: con pocas personas el FAR es orientativo. Sumá mas gente")
        print("        y tomas en distintas luces para un numero confiable.")
    print("=" * 60 + "\n")

    # ----- Grafico (opcional) -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        plt.plot(ths, far * 100, label="FAR (impostores aceptados)", color="#B23B2E")
        plt.plot(ths, frr * 100, label="FRR (autorizados rechazados)", color="#1C6B57")
        plt.axvline(recomendado, color="#A46036", ls="--",
                    label=f"recomendado {recomendado:.2f}")
        plt.xlabel("umbral de similitud")
        plt.ylabel("tasa (%)")
        plt.title("Calibracion FAR / FRR - buffalo_l")
        plt.legend()
        plt.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=120)
        print(f"Grafico guardado en: {PLOT_PATH}\n")
    except Exception as exc:
        print(f"(No se pudo guardar el grafico: {exc})\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrar umbral FAR/FRR.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("capturar", help="capturar embeddings de una persona")
    pc.add_argument("nombre", help="nombre de la persona")
    pm = sub.add_parser("medir", help="medir FAR/FRR y recomendar umbral")
    pm.add_argument("--far-objetivo", type=float, default=0.001,
                    help="FAR maximo aceptable (default 0.001 = 0.1%%)")
    args = parser.parse_args()

    if args.cmd == "capturar":
        cmd_capturar(args.nombre)
    elif args.cmd == "medir":
        cmd_medir(args.far_objetivo)


if __name__ == "__main__":
    main()
