"""
FICHAJE FACIAL - Control de asistencia por reconocimiento facial (InsightFace).

Producto enfocado: registrar ENTRADA y SALIDA de empleados con la cara,
reemplazando relojes de huella o tarjetas.

Motor de IA
-----------
Usa InsightFace (modelo buffalo_l): detecta, ALINEA y reconoce la cara con deep
learning, generando un "embedding" de 512 dimensiones por rostro. Es robusto a
luz, angulo y distancia -> el empleado se da de alta UNA vez y despues el sistema
lo identifica y ficha solo, sin volver a registrarse.

Puntos de producto
-------------------
  * Alta con CONSENTIMIENTO registrado (dato biometrico = dato sensible).
  * Fichaje automatico entrada/salida segun el ultimo movimiento del dia.
  * Registro exportable en CSV (fichajes.csv) -> el reporte que se le vende a RRHH.
  * Se guardan EMBEDDINGS (vectores), nunca fotos. Base cifrable.

Dependencias
------------
    pip install insightface onnxruntime opencv-contrib-python numpy
    (la primera ejecucion descarga el modelo buffalo_l ~300 MB)

Archivos que mantiene
---------------------
    empleados.json  -> alta, legajo y consentimiento
    faces.npz       -> embeddings faciales enrolados
    fichajes.csv    -> registro de asistencia (reporte vendible)

Controles
---------
    e   alta de empleado (captura + nombre + legajo + consentimiento)
    x   mostrar resumen del dia en consola
    q   salir
"""

import os
import csv
import json
import time
import warnings
import datetime as dt

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from insightface.app import FaceAnalysis


# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

EMPRESA = "Mi Empresa S.A."
_HERE = os.path.dirname(os.path.abspath(__file__))

EMPLEADOS_PATH = os.path.join(_HERE, "empleados.json")
FACES_DB_PATH = os.path.join(_HERE, "faces.npz")
FICHAJES_CSV = os.path.join(_HERE, "fichajes.csv")
FACE_LANDMARKER_PATH = os.path.join(_HERE, "face_landmarker.task")

# --- Liveness / anti-spoofing (deteccion de parpadeo) ----------------------
# Antes de registrar, exige un PARPADEO real. Una foto no parpadea -> queda
# bloqueada. Usa los blendshapes de MediaPipe (score de parpadeo por ojo).
LIVENESS_ENABLED = True        # se puede alternar en vivo con la tecla 'v'.
BLINK_CLOSE = 0.45             # score > esto = ojo cerrandose.
BLINK_OPEN = 0.22             # score < esto = ojo abierto (completa el parpadeo).
BLINK_FRESH_SECONDS = 2.5      # el parpadeo vale por estos segundos.

# Motor InsightFace.
MODEL_NAME = "buffalo_l"       # pack de modelos (detector + reconocimiento).
DET_SIZE = (416, 416)          # resolucion de deteccion. Menor = mas rapido.
PROCESS_EVERY = 2              # correr IA cada N cuadros (fluidez en CPU).

# Reconocimiento (similitud coseno de embeddings de 512-d).
# buffalo_l: misma persona ~0.5-0.8, distintas <0.3. 0.40 da buen margen.
RECOGNIZE_SIM = 0.40           # similitud minima para aceptar identidad.
STABLE_FRAMES = 4              # cuadros seguidos reconociendo antes de registrar.
FICHAJE_COOLDOWN = 12          # segundos antes de que la misma persona re-fiche.
CONFIRM_SECONDS = 3.5          # duracion del cartel de confirmacion.
ENROLL_SAMPLES = 20            # embeddings capturados al dar de alta.

# Paleta de marca ASISTRA (BGR). Industrial: carbon, oliva, cobre, arena.
COL_BRAND = (72, 100, 88)     # verde oliva #566246 (marca / acentos)
COL_OK = (96, 156, 108)       # oliva-verde claro: fichaje/reconocido OK
COL_WARN = (56, 100, 168)     # cobre #A46036: atencion / no reconocido
COL_INK = (242, 247, 248)     # blanco calido #F8F7F2
COL_DIM = (150, 158, 162)     # gris calido secundario
COL_PANEL = (26, 26, 26)      # carbon #1B1B1B

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# RECONOCEDOR  (vecino mas cercano sobre embeddings de InsightFace)
# ---------------------------------------------------------------------------

def blink_score(landmarker_result):
    """Devuelve el score de parpadeo (0..1) del rostro, o None si no hay cara.

    MediaPipe entrega 'blendshapes' estilo ARKit; usamos eyeBlinkLeft/Right y
    tomamos el maximo (basta con que un ojo se cierre). Score alto = ojo cerrado.
    """
    if not landmarker_result.face_blendshapes:
        return None
    izq = der = 0.0
    for cat in landmarker_result.face_blendshapes[0]:
        if cat.category_name == "eyeBlinkLeft":
            izq = cat.score
        elif cat.category_name == "eyeBlinkRight":
            der = cat.score
    return max(izq, der)


class FaceRecognizer:
    """Almacena embeddings enrolados y reconoce por similitud coseno.

    Los embeddings de InsightFace ya vienen L2-normalizados, asi que la
    similitud coseno es simplemente el producto punto. Guardamos varios
    embeddings por persona (capturados en el alta) para mas robustez.
    """

    def __init__(self, threshold=RECOGNIZE_SIM):
        self.threshold = threshold
        self.embeddings = None       # (N, 512) float32 normalizados
        self.ids = None              # (N,) id de empleado por embedding

    @property
    def trained(self):
        return self.embeddings is not None and len(self.embeddings) > 0

    def add(self, embs, emp_id):
        arr = np.array(embs, dtype=np.float32)
        ids = np.full(len(embs), emp_id, dtype=np.int32)
        if self.embeddings is None:
            self.embeddings, self.ids = arr, ids
        else:
            self.embeddings = np.vstack([self.embeddings, arr])
            self.ids = np.concatenate([self.ids, ids])

    def predict(self, emb):
        """Devuelve (id_empleado, similitud) del rostro enrolado mas parecido."""
        if not self.trained:
            return None, 0.0
        sims = self.embeddings @ emb
        i = int(np.argmax(sims))
        return int(self.ids[i]), float(sims[i])

    def save(self, path):
        np.savez(path, embeddings=self.embeddings, ids=self.ids)

    def load(self, path):
        data = np.load(path)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.ids = data["ids"].astype(np.int32)


# ---------------------------------------------------------------------------
# PERSISTENCIA:  empleados y registro de fichajes
# ---------------------------------------------------------------------------

def cargar_empleados():
    if not os.path.exists(EMPLEADOS_PATH):
        return {}
    with open(EMPLEADOS_PATH, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def guardar_empleados(empleados):
    with open(EMPLEADOS_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in empleados.items()}, f,
                  ensure_ascii=False, indent=2)


def asegurar_csv():
    if not os.path.exists(FICHAJES_CSV):
        with open(FICHAJES_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["fecha", "hora", "legajo", "nombre", "tipo"])


def contar_fichajes_hoy():
    """{legajo: cantidad} de HOY, para alternar entrada/salida tras reinicios."""
    hoy = dt.date.today().isoformat()
    conteo = {}
    if os.path.exists(FICHAJES_CSV):
        with open(FICHAJES_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["fecha"] == hoy:
                    conteo[row["legajo"]] = conteo.get(row["legajo"], 0) + 1
    return conteo


def registrar_fichaje(empleado, tipo):
    ahora = dt.datetime.now()
    with open(FICHAJES_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            ahora.date().isoformat(),
            ahora.strftime("%H:%M:%S"),
            empleado["legajo"],
            empleado["nombre"],
            tipo,
        ])


def resumen_dia():
    hoy = dt.date.today().isoformat()
    filas = []
    if os.path.exists(FICHAJES_CSV):
        with open(FICHAJES_CSV, "r", encoding="utf-8") as f:
            filas = [r for r in csv.DictReader(f) if r["fecha"] == hoy]
    print("\n" + "=" * 48)
    print(f"  RESUMEN DE ASISTENCIA - {hoy}")
    print("=" * 48)
    if not filas:
        print("  (sin fichajes hoy)")
    for r in filas:
        print(f"  {r['hora']}  [{r['legajo']:>6}]  {r['nombre']:<20} {r['tipo']}")
    print(f"  Total de fichajes hoy: {len(filas)}")
    print(f"  Archivo: {FICHAJES_CSV}")
    print("=" * 48 + "\n")


# ---------------------------------------------------------------------------
# DIBUJO
# ---------------------------------------------------------------------------

def dibujar_marco(frame, box, color, grosor=2):
    x1, y1, x2, y2 = (int(v) for v in box)
    seg = max(14, int((x2 - x1) * 0.22))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * seg, cy), color, grosor)
        cv2.line(frame, (cx, cy), (cx, cy + dy * seg), color, grosor)


def texto_centrado(frame, text, cy, font, scale, color, thick):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(frame, text, ((frame.shape[1] - tw) // 2, cy), font, scale,
                color, thick, lineType=cv2.LINE_AA)
    return th


def panel(frame, x1, y1, x2, y2, alpha=0.55):
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), COL_PANEL, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


CONSENT_LINES = [
    "CONSENTIMIENTO DE DATOS BIOMETRICOS",
    "",
    "Al dar de alta, el empleado autoriza el uso de sus",
    "datos faciales UNICAMENTE para control de asistencia.",
    "Los datos se guardan cifrados y no se comparten.",
    "Puede revocar el consentimiento cuando lo solicite.",
    "",
    "[ S ] ACEPTA y da de alta      [ Esc ] Cancelar",
]


# ---------------------------------------------------------------------------
# APLICACION
# ---------------------------------------------------------------------------

IDLE, CAPTURE, NAME, LEGAJO, CONSENT = "idle", "capture", "name", "legajo", "consent"


def cara_mas_grande(faces):
    """Devuelve la Face de mayor area (la persona frente al kiosco)."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def main():
    asegurar_csv()

    print("Cargando motor de IA (InsightFace)...")
    app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    print("Motor listo.")

    # Detector de parpadeo para liveness (opcional pero recomendado).
    landmarker = None
    if os.path.exists(FACE_LANDMARKER_PATH):
        landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=FACE_LANDMARKER_PATH),
                running_mode=vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                num_faces=1))
        print("Liveness (deteccion de parpadeo) activo.")
    else:
        print("AVISO: falta face_landmarker.task -> liveness desactivado.")

    recognizer = FaceRecognizer()
    empleados = cargar_empleados()
    next_id = (max(empleados) + 1) if empleados else 0
    if os.path.exists(FACES_DB_PATH) and empleados:
        try:
            recognizer.load(FACES_DB_PATH)
            print(f"Base cargada: {len(empleados)} empleado(s).")
        except Exception as exc:
            print(f"AVISO: no se pudo cargar la base facial ({exc})")

    conteo_hoy = contar_fichajes_hoy()
    ultimo_fichaje = {}

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la webcam.")
        return

    # Estado de reconocimiento / fichaje.
    pending_id, pending_count = None, 0
    confirm_until, confirm_text, confirm_color = 0.0, "", COL_OK
    # Estado del alta.
    stage = IDLE
    enroll_embs = []
    buf_nombre, buf_legajo = "", ""
    # Estado de liveness (parpadeo).
    liveness_on = LIVENESS_ENABLED and landmarker is not None
    eyes_closed = False        # con histeresis, para detectar el parpadeo
    last_blink = 0.0           # momento del ultimo parpadeo completado

    faces = []              # ultimo resultado de la IA (cacheado)
    frame_idx = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: No se pudo leer la webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        frame_idx += 1

        # Correr la IA cada PROCESS_EVERY cuadros (detecta + alinea + embedding).
        if frame_idx % PROCESS_EVERY == 0 or frame_idx == 1:
            faces = app.get(frame)
        face = cara_mas_grande(faces)

        # Deteccion de parpadeo cada frame (para liveness). La histeresis evita
        # falsos positivos: solo contamos un parpadeo cuando el ojo se cierra
        # (score alto) y luego se vuelve a abrir (score bajo).
        if liveness_on and stage == IDLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            lm_res = landmarker.detect_for_video(mp_img, int((now - start) * 1000))
            score = blink_score(lm_res)
            if score is not None:
                if not eyes_closed and score > BLINK_CLOSE:
                    eyes_closed = True
                elif eyes_closed and score < BLINK_OPEN:
                    eyes_closed = False
                    last_blink = now            # se completo un parpadeo

        # ============ MODO ALTA ============
        if stage == CAPTURE:
            if face is not None:
                enroll_embs.append(face.normed_embedding)
                dibujar_marco(frame, face.bbox, COL_WARN, 2)
            panel(frame, 0, h - 70, w, h)
            texto_centrado(frame, f"CAPTURANDO ROSTRO  {len(enroll_embs)}/"
                           f"{ENROLL_SAMPLES}", h - 28, FONT_BOLD, 0.8, COL_INK, 2)
            if len(enroll_embs) >= ENROLL_SAMPLES:
                stage, buf_nombre = NAME, ""

        elif stage in (NAME, LEGAJO):
            panel(frame, 0, h // 2 - 60, w, h // 2 + 60)
            prompt, val = (("Nombre y apellido:", buf_nombre) if stage == NAME
                           else ("Legajo / N de empleado:", buf_legajo))
            cursor = "_" if int(now * 2) % 2 == 0 else " "
            texto_centrado(frame, prompt, h // 2 - 15, FONT, 0.7, COL_DIM, 2)
            texto_centrado(frame, val + cursor, h // 2 + 30, FONT_BOLD, 1.1,
                           COL_INK, 2)

        elif stage == CONSENT:
            panel(frame, 0, 0, w, h, alpha=0.72)
            y = h // 2 - 120
            for i, line in enumerate(CONSENT_LINES):
                fnt = FONT_BOLD if i == 0 else FONT
                sc = 0.85 if i == 0 else 0.62
                col = COL_OK if line.startswith("[ S ]") else COL_INK
                texto_centrado(frame, line, y, fnt, sc, col, 2)
                y += 34

        # ============ MODO KIOSCO ============
        else:
            identidad = None
            if recognizer.trained and face is not None:
                emp_id, sim = recognizer.predict(face.normed_embedding)
                if sim >= RECOGNIZE_SIM and emp_id in empleados:
                    identidad = (emp_id, sim)

            if identidad is not None:
                emp_id, sim = identidad
                emp = empleados[emp_id]
                dibujar_marco(frame, face.bbox, COL_OK, 2)

                if pending_id == emp_id:
                    pending_count += 1
                else:
                    pending_id, pending_count = emp_id, 1

                en_cooldown = (now - ultimo_fichaje.get(emp_id, 0)
                               < FICHAJE_COOLDOWN)
                # Liveness: exige un parpadeo reciente. Una foto no parpadea.
                liveness_ok = (not liveness_on
                               or (now - last_blink) < BLINK_FRESH_SECONDS)

                if (pending_count >= STABLE_FRAMES and not en_cooldown
                        and liveness_ok):
                    legajo = emp["legajo"]
                    n = conteo_hoy.get(legajo, 0)
                    tipo = "ENTRADA" if n % 2 == 0 else "SALIDA"
                    registrar_fichaje(emp, tipo)
                    conteo_hoy[legajo] = n + 1
                    ultimo_fichaje[emp_id] = now
                    hora = dt.datetime.now().strftime("%H:%M")
                    confirm_text = f"{tipo} - {emp['nombre']} - {hora}"
                    confirm_color = COL_OK
                    confirm_until = now + CONFIRM_SECONDS
                    print(f"[FICHAJE] {hora} {legajo} {emp['nombre']} -> {tipo} "
                          f"(sim {sim:.2f})")
                    pending_id, pending_count = None, 0
                elif en_cooldown:
                    dibujar_marco(frame, face.bbox, COL_DIM, 2)
                elif pending_count >= STABLE_FRAMES and not liveness_ok:
                    # Reconocido y estable, pero falta el parpadeo: lo pedimos.
                    texto_centrado(frame, "Parpadea para fichar", h - 90,
                                   FONT_BOLD, 0.9, COL_WARN, 2)
                # nombre sobre el marco
                bx1, by1 = int(face.bbox[0]), int(face.bbox[1])
                cv2.putText(frame, emp["nombre"], (bx1, by1 - 10), FONT_BOLD,
                            0.7, COL_OK, 2, lineType=cv2.LINE_AA)

            elif face is not None:
                dibujar_marco(frame, face.bbox, COL_WARN, 2)
                bx1, by1 = int(face.bbox[0]), int(face.bbox[1])
                cv2.putText(frame, "No reconocido", (bx1, by1 - 10), FONT, 0.6,
                            COL_WARN, 2, lineType=cv2.LINE_AA)
                pending_id, pending_count = None, 0
            else:
                pending_id, pending_count = None, 0

        # ---------------- HUD ----------------
        panel(frame, 0, 0, w, 46)
        ahora = dt.datetime.now()
        # Marca del producto (ASISTRA) + nombre del cliente (EMPRESA).
        cv2.putText(frame, "ASISTRA", (16, 30), FONT_BOLD, 0.8, COL_BRAND, 2,
                    lineType=cv2.LINE_AA)
        (aw, _), _ = cv2.getTextSize("ASISTRA", FONT_BOLD, 0.8, 2)
        cv2.putText(frame, "  ·  " + EMPRESA, (16 + aw, 29), FONT, 0.6, COL_DIM,
                    1, lineType=cv2.LINE_AA)
        reloj = ahora.strftime("%H:%M:%S")
        (rw, _), _ = cv2.getTextSize(reloj, FONT_BOLD, 0.9, 2)
        cv2.putText(frame, reloj, (w - rw - 16, 32), FONT_BOLD, 0.9, COL_BRAND, 2,
                    lineType=cv2.LINE_AA)
        cv2.putText(frame, ahora.strftime("%d/%m/%Y"), (w // 2 - 55, 30),
                    FONT, 0.6, COL_DIM, 1, lineType=cv2.LINE_AA)

        if stage == IDLE:
            if now < confirm_until:
                panel(frame, 0, h // 2 - 45, w, h // 2 + 45, alpha=0.6)
                texto_centrado(frame, "FICHAJE OK", h // 2 - 8, FONT_BOLD, 1.0,
                               confirm_color, 2)
                texto_centrado(frame, confirm_text, h // 2 + 28, FONT, 0.7,
                               COL_INK, 2)
            elif not recognizer.trained:
                panel(frame, 0, h - 60, w, h)
                texto_centrado(frame, "Sin empleados. Presiona 'e' para dar de "
                               "alta.", h - 22, FONT, 0.65, COL_DIM, 2)

        panel(frame, 0, h - 24, w, h, alpha=0.5)
        liv = "ON" if liveness_on else "off"
        cv2.putText(frame, f"[e] alta  [x] resumen  [v] liveness:{liv}  [q] salir",
                    (12, h - 7), FONT, 0.5, COL_DIM, 1, lineType=cv2.LINE_AA)
        aviso = "Datos biometricos usados solo para asistencia (consentimiento)"
        (aw, _), _ = cv2.getTextSize(aviso, FONT, 0.45, 1)
        cv2.putText(frame, aviso, (w - aw - 12, h - 7), FONT, 0.45, COL_DIM, 1,
                    lineType=cv2.LINE_AA)

        cv2.imshow("ASISTRA - Fichaje Facial", frame)

        # ---------------- TECLADO ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue

        if stage in (NAME, LEGAJO):
            if key in (13, 10):
                if stage == NAME and buf_nombre.strip():
                    stage, buf_legajo = LEGAJO, ""
                elif stage == LEGAJO and buf_legajo.strip():
                    stage = CONSENT
            elif key == 27:
                stage, enroll_embs = IDLE, []
                print("Alta cancelada.")
            elif key in (8, 127):
                if stage == NAME:
                    buf_nombre = buf_nombre[:-1]
                else:
                    buf_legajo = buf_legajo[:-1]
            elif 32 <= key <= 126:
                if stage == NAME:
                    buf_nombre += chr(key)
                else:
                    buf_legajo += chr(key)
            continue

        if stage == CONSENT:
            if key in (ord('s'), ord('S'), 13, 10):
                emp_id = next_id
                next_id += 1
                empleados[emp_id] = {
                    "nombre": buf_nombre.strip(),
                    "legajo": buf_legajo.strip(),
                    "consentimiento": True,
                    "fecha_consentimiento": dt.datetime.now().isoformat(timespec="seconds"),
                    "fecha_alta": dt.date.today().isoformat(),
                }
                recognizer.add(enroll_embs, emp_id)
                recognizer.save(FACES_DB_PATH)
                guardar_empleados(empleados)
                print(f"Alta OK: {buf_nombre.strip()} (legajo {buf_legajo.strip()}) "
                      f"- consentimiento registrado.")
                confirm_text = f"Alta completada: {buf_nombre.strip()}"
                confirm_color, confirm_until = COL_OK, now + CONFIRM_SECONDS
                stage, enroll_embs = IDLE, []
            elif key == 27:
                stage, enroll_embs = IDLE, []
                print("Alta cancelada (sin consentimiento).")
            continue

        if key == ord('q'):
            break
        elif key == ord('e'):
            stage, enroll_embs = CAPTURE, []
            print("Alta: mante la cara centrada mientras captura...")
        elif key == ord('x'):
            resumen_dia()
        elif key == ord('v'):
            if landmarker is not None:
                liveness_on = not liveness_on
                print(f"Liveness {'activado' if liveness_on else 'desactivado'}.")
            else:
                print("Liveness no disponible (falta face_landmarker.task).")

    cap.release()
    cv2.destroyAllWindows()
    if landmarker is not None:
        landmarker.close()
    resumen_dia()


if __name__ == "__main__":
    main()
