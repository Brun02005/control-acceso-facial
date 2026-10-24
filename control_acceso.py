"""
CONTROL DE ACCESO FACIAL - Reconocimiento facial para accesos autorizados.

Evolucion del motor de fichaje hacia un producto de CONTROL DE ACCESO: en vez de
registrar entrada/salida, decide si una persona puede PASAR o no.

  * Solo acceden personas ENROLADAS, AUTORIZADAS y VIGENTES.
  * A un desconocido (o a alguien dado de baja / vencido) -> ACCESO DENEGADO.
  * Cada intento queda en un LOG DE ACCESOS (auditoria): quien, cuando, resultado.
  * Prueba de vida por parpadeo: una foto no abre la puerta.
  * Hook abrir_puerta() listo para conectar un rele / cerradura electrica.

NO es vigilancia: no identifica desconocidos ni arma listas de personas que no
dieron su consentimiento. Es verificacion de identidad para acceso consentido.

Motor de IA
-----------
InsightFace (buffalo_l): detecta + alinea + embedding de 512-d. Robusto a luz,
angulo y distancia. Reconocimiento por similitud coseno (vecino mas cercano).

Dependencias
------------
    pip install insightface onnxruntime opencv-contrib-python mediapipe numpy
    (la primera ejecucion descarga el modelo buffalo_l ~300 MB)

Archivos que mantiene
---------------------
    personas.json      -> altas, documento, consentimiento, vigencia, autorizacion
    faces_acceso.npz   -> embeddings faciales enrolados
    accesos.csv        -> log de accesos (auditoria)

Controles
---------
    e   dar de alta una persona autorizada
    l   listar personas en la consola
    r   dar de baja / reactivar una persona (por id)
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

import cripto   # cifrado en reposo de la base biometrica


# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

INSTALACION = "Acceso Principal"       # se muestra en pantalla
_HERE = os.path.dirname(os.path.abspath(__file__))

# Base biometrica CIFRADA (.enc). Los .json/.npz en plano son solo para migrar
# datos viejos la primera vez; despues quedan como *.PLANO_BORRAR.
PERSONAS_PATH = os.path.join(_HERE, "personas.enc")
FACES_DB_PATH = os.path.join(_HERE, "faces_acceso.enc")
PERSONAS_PLANO = os.path.join(_HERE, "personas.json")
FACES_PLANO = os.path.join(_HERE, "faces_acceso.npz")
ACCESOS_CSV = os.path.join(_HERE, "accesos.csv")
FACE_LANDMARKER_PATH = os.path.join(_HERE, "face_landmarker.task")

# Motor InsightFace.
MODEL_NAME = "buffalo_l"
DET_SIZE = (416, 416)
PROCESS_EVERY = 2

# Reconocimiento. Para ACCESO conviene ESTRICTO: preferimos rechazar de mas
# (falso rechazo = molestia) antes que aceptar de mas (falso acceso = riesgo).
RECOGNIZE_SIM = 0.35           # MEDIDO en LFW (benchmark_lfw.py, 2308 identidades):
                               # FAR ~0.008% (1 en ~12.500), FRR ~1.4%. Punto del
                               # "valle": FAR bajo con margen y FRR en su minimo.
STABLE_FRAMES = 3              # cuadros seguidos reconociendo antes de conceder.
ACCESO_COOLDOWN = 6            # seg antes de que la misma persona re-dispare.
DENY_LOG_COOLDOWN = 5          # seg para no spamear el log con denegaciones.
BANNER_SECONDS = 3.5           # duracion del cartel AUTORIZADO/DENEGADO.
DOOR_OPEN_SECONDS = 4          # cuanto "queda abierta" la puerta (y el rele).
ENROLL_SAMPLES = 25            # embeddings capturados al dar de alta (mas = mejor).

# --- Liveness / anti-spoofing (parpadeo) -----------------------------------
LIVENESS_ENABLED = True
BLINK_CLOSE = 0.45
BLINK_OPEN = 0.22
BLINK_FRESH_SECONDS = 2.5
LM_FACES = 3                   # caras que rastrea el landmarker (para atar el parpadeo)
IOU_MIN = 0.35                 # solape minimo para considerar "la misma cara"

# Colores (BGR). Verdictos claros + acentos de marca.
COL_GRANT = (90, 200, 90)      # verde: acceso autorizado
COL_DENY = (60, 60, 225)       # rojo: acceso denegado
COL_BRAND = (72, 100, 88)      # oliva (acentos)
COL_INK = (242, 247, 248)
COL_DIM = (150, 158, 162)
COL_PANEL = (24, 24, 24)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# PUERTA  (hook de hardware)
# ---------------------------------------------------------------------------

def abrir_puerta(persona):
    """Dispara la apertura de la puerta.

    Hoy solo lo registra en consola (simulado). Para produccion, aca se activa
    el hardware: un rele USB, un GPIO (Raspberry Pi) o un comando a una
    controladora de cerradura. Ejemplo con pyserial a un Arduino con rele:

        import serial
        ser = serial.Serial("COM3", 9600, timeout=1)
        ser.write(b"OPEN\\n")

    Se mantiene abierta DOOR_OPEN_SECONDS y luego la cerradura vuelve a cerrar.
    """
    print(f"[PUERTA] >>> ABIERTA para {persona['nombre']} "
          f"({DOOR_OPEN_SECONDS}s)")


# ---------------------------------------------------------------------------
# LIVENESS
# ---------------------------------------------------------------------------

def blink_de_categorias(categorias):
    """Score de parpadeo (0..1) de UNA cara, desde sus blendshapes."""
    izq = der = 0.0
    for cat in categorias:
        if cat.category_name == "eyeBlinkLeft":
            izq = cat.score
        elif cat.category_name == "eyeBlinkRight":
            der = cat.score
    return max(izq, der)


def lm_bbox(landmarks, w, h):
    """Caja (x1,y1,x2,y2 px) de una cara a partir de sus landmarks normalizados."""
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    return (min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h)


def iou(a, b):
    """Interseccion sobre union de dos cajas (x1,y1,x2,y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# RECONOCEDOR  (vecino mas cercano sobre embeddings de InsightFace)
# ---------------------------------------------------------------------------

class FaceRecognizer:
    """Embeddings enrolados + reconocimiento por similitud coseno.

    Los embeddings de InsightFace vienen L2-normalizados -> la similitud coseno
    es el producto punto. Guardamos varios por persona (del alta) para robustez.
    """

    def __init__(self, threshold=RECOGNIZE_SIM):
        self.threshold = threshold
        self.embeddings = None       # (N, 512) float32
        self.ids = None              # (N,) id de persona por embedding

    @property
    def trained(self):
        return self.embeddings is not None and len(self.embeddings) > 0

    def add(self, embs, pid):
        arr = np.array(embs, dtype=np.float32)
        ids = np.full(len(embs), pid, dtype=np.int32)
        if self.embeddings is None:
            self.embeddings, self.ids = arr, ids
        else:
            self.embeddings = np.vstack([self.embeddings, arr])
            self.ids = np.concatenate([self.ids, ids])

    def predict(self, emb):
        """(id_persona, similitud) del rostro enrolado mas parecido."""
        if not self.trained:
            return None, 0.0
        sims = self.embeddings @ emb
        i = int(np.argmax(sims))
        return int(self.ids[i]), float(sims[i])

    def save(self, path):
        cripto.guardar_npz(path, embeddings=self.embeddings, ids=self.ids)

    def load(self, path):
        data = cripto.cargar_npz(path)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.ids = data["ids"].astype(np.int32)


# ---------------------------------------------------------------------------
# PERSISTENCIA
# ---------------------------------------------------------------------------

def cargar_personas():
    if not os.path.exists(PERSONAS_PATH):
        return {}
    return {int(k): v for k, v in cripto.cargar_json(PERSONAS_PATH).items()}


def guardar_personas(personas):
    cripto.guardar_json(PERSONAS_PATH, {str(k): v for k, v in personas.items()})


def migrar_plano_a_cifrado():
    """Migra datos viejos en texto plano a los archivos .enc (una sola vez).

    Lee personas.json / faces_acceso.npz si existen y aun no hay version .enc,
    los re-escribe CIFRADOS con la clave actual, y renombra los planos a
    *.PLANO_BORRAR para que el usuario los elimine tras verificar.
    """
    migrado = []
    if os.path.exists(FACES_PLANO) and not os.path.exists(FACES_DB_PATH):
        # 'with' cierra el npz antes del rename (en Windows no se puede
        # renombrar un archivo que sigue abierto).
        with np.load(FACES_PLANO) as d:
            cripto.guardar_npz(FACES_DB_PATH,
                               embeddings=d["embeddings"], ids=d["ids"])
        os.rename(FACES_PLANO, FACES_PLANO + ".PLANO_BORRAR")
        migrado.append(os.path.basename(FACES_PLANO))
    if os.path.exists(PERSONAS_PLANO) and not os.path.exists(PERSONAS_PATH):
        with open(PERSONAS_PLANO, "r", encoding="utf-8") as f:
            cripto.guardar_json(PERSONAS_PATH, json.load(f))
        os.rename(PERSONAS_PLANO, PERSONAS_PLANO + ".PLANO_BORRAR")
        migrado.append(os.path.basename(PERSONAS_PLANO))
    if migrado:
        print(f"MIGRACION: cifrados {', '.join(migrado)}.")
        print("  Los originales en texto plano quedaron como *.PLANO_BORRAR;")
        print("  borralos a mano cuando confirmes que el acceso anda bien.")


def asegurar_csv():
    if not os.path.exists(ACCESOS_CSV):
        with open(ACCESOS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["fecha", "hora", "nombre", "documento", "resultado",
                 "motivo", "similitud"])


def registrar_acceso(nombre, documento, resultado, motivo, sim):
    ahora = dt.datetime.now()
    with open(ACCESOS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            ahora.date().isoformat(), ahora.strftime("%H:%M:%S"),
            nombre, documento, resultado, motivo, f"{sim:.3f}"])


def vigente(persona):
    """True si la autorizacion no esta vencida (visitas con fecha limite)."""
    vh = persona.get("valido_hasta")
    if not vh:
        return True                       # permanente
    return dt.date.today().isoformat() <= vh


def listar_personas(personas):
    print("\n" + "=" * 60)
    print("  PERSONAS REGISTRADAS")
    print("=" * 60)
    if not personas:
        print("  (ninguna)")
    for pid, p in sorted(personas.items()):
        estado = "AUTORIZADA" if p.get("autorizado", True) else "DADA DE BAJA"
        vig = p.get("valido_hasta") or "permanente"
        if p.get("valido_hasta") and not vigente(p):
            estado += " (VENCIDA)"
        print(f"  id {pid:>3} · {p['nombre']:<22} doc {p.get('documento','-'):<12} "
              f"· {estado} · {vig}")
    print("=" * 60 + "\n")


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
    "La persona autoriza el uso de sus datos faciales",
    "UNICAMENTE para control de acceso a esta instalacion.",
    "Los datos se guardan CIFRADOS y no se comparten.",
    "Puede revocar el consentimiento cuando lo solicite.",
    "",
    "[ S ] ACEPTA y registra      [ Esc ] Cancelar",
]


# ---------------------------------------------------------------------------
# APLICACION
# ---------------------------------------------------------------------------

IDLE, CAPTURE, NAME, DOC, TIPO, CONSENT, REVOKE = \
    "idle", "capture", "name", "doc", "tipo", "consent", "revoke"


def cara_mas_grande(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def main():
    asegurar_csv()

    # --- Cifrado en reposo: sin clave no se arranca (fail-closed) ---
    if not cripto.hay_clave():
        print("=" * 60)
        print("  FALTA LA CLAVE DE CIFRADO (variable de entorno ASISTRA_KEY).")
        print("  La base biometrica va cifrada; sin la clave no se puede operar.")
        print("  1) Genera una:  python generar_clave.py")
        print('  2) Instalala:   setx ASISTRA_KEY "<clave>"')
        print("  3) Abri una terminal NUEVA y volve a correr.")
        print("=" * 60)
        return
    try:
        migrar_plano_a_cifrado()
    except cripto.ClaveInvalida as exc:
        print(f"ERROR de clave: {exc}")
        return

    print("Cargando motor de IA (InsightFace)...")
    app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    print("Motor listo.")

    landmarker = None
    if os.path.exists(FACE_LANDMARKER_PATH):
        landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=FACE_LANDMARKER_PATH),
                running_mode=vision.RunningMode.VIDEO,
                output_face_blendshapes=True, num_faces=LM_FACES))
        print("Liveness (parpadeo) activo.")
    else:
        print("AVISO: falta face_landmarker.task -> liveness desactivado.")

    recognizer = FaceRecognizer()
    try:
        personas = cargar_personas()
        if os.path.exists(FACES_DB_PATH) and personas:
            recognizer.load(FACES_DB_PATH)
            print(f"Base cargada (cifrada): {len(personas)} persona(s).")
    except cripto.ClaveInvalida as exc:
        print(f"ERROR: {exc}")
        print("  La clave ASISTRA_KEY no coincide con los datos cifrados.")
        return
    next_id = (max(personas) + 1) if personas else 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la webcam.")
        return

    # Estado de decision.
    pending_id, pending_count = None, 0
    ultimo_acceso = {}          # {pid: time} cooldown por persona
    last_deny_log = 0.0         # para no spamear denegaciones en el log
    banner_until = 0.0
    banner_text = banner_sub = ""
    banner_color = COL_GRANT
    door_until = 0.0            # hasta cuando mostramos "PUERTA ABIERTA"

    # Estado del alta / baja.
    stage = IDLE
    enroll_embs = []
    buf_nombre, buf_doc, buf_revoke = "", "", ""
    tipo_visita = False

    # Liveness.
    liveness_on = LIVENESS_ENABLED and landmarker is not None
    eyes_closed = False
    last_blink = 0.0

    faces = []
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

        if frame_idx % PROCESS_EVERY == 0 or frame_idx == 1:
            faces = app.get(frame)
        face = cara_mas_grande(faces)

        # Parpadeo de TODAS las caras del frame (hasta LM_FACES), cada una con su
        # caja. Mas abajo atamos el parpadeo a la cara que InsightFace reconocio:
        # si no, una foto de un autorizado + otra persona parpadeando al lado
        # abriria la puerta. Guardamos (bbox, score) por cara.
        lm_faces = []
        if liveness_on and stage == IDLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            lm_res = landmarker.detect_for_video(mp_img, int((now - start) * 1000))
            if lm_res.face_landmarks:
                for i, lms in enumerate(lm_res.face_landmarks):
                    sc = (blink_de_categorias(lm_res.face_blendshapes[i])
                          if lm_res.face_blendshapes
                          and i < len(lm_res.face_blendshapes) else None)
                    lm_faces.append((lm_bbox(lms, w, h), sc))

        # ============ MODO ALTA ============
        if stage == CAPTURE:
            if face is not None:
                enroll_embs.append(face.normed_embedding)
                dibujar_marco(frame, face.bbox, COL_BRAND, 2)
            panel(frame, 0, h - 70, w, h)
            texto_centrado(frame, f"CAPTURANDO ROSTRO  {len(enroll_embs)}/"
                           f"{ENROLL_SAMPLES}", h - 28, FONT_BOLD, 0.8, COL_INK, 2)
            if len(enroll_embs) >= ENROLL_SAMPLES:
                stage, buf_nombre = NAME, ""

        elif stage in (NAME, DOC, REVOKE):
            panel(frame, 0, h // 2 - 60, w, h // 2 + 60)
            if stage == NAME:
                prompt, val = "Nombre y apellido:", buf_nombre
            elif stage == DOC:
                prompt, val = "Documento / ID:", buf_doc
            else:
                prompt, val = "ID a dar de baja/reactivar:", buf_revoke
            cursor = "_" if int(now * 2) % 2 == 0 else " "
            texto_centrado(frame, prompt, h // 2 - 15, FONT, 0.7, COL_DIM, 2)
            texto_centrado(frame, val + cursor, h // 2 + 30, FONT_BOLD, 1.1,
                           COL_INK, 2)

        elif stage == TIPO:
            panel(frame, 0, h // 2 - 60, w, h // 2 + 60)
            texto_centrado(frame, "Tipo de acceso", h // 2 - 15, FONT, 0.7,
                           COL_DIM, 2)
            texto_centrado(frame, "[ P ] Permanente      [ V ] Visita (hoy)",
                           h // 2 + 30, FONT_BOLD, 0.9, COL_INK, 2)

        elif stage == CONSENT:
            panel(frame, 0, 0, w, h, alpha=0.72)
            y = h // 2 - 120
            for i, line in enumerate(CONSENT_LINES):
                fnt = FONT_BOLD if i == 0 else FONT
                sc = 0.85 if i == 0 else 0.62
                col = COL_GRANT if line.startswith("[ S ]") else COL_INK
                texto_centrado(frame, line, y, fnt, sc, col, 2)
                y += 34

        # ============ MODO CONTROL DE ACCESO ============
        else:
            if face is None:
                pending_id, pending_count = None, 0
            elif not recognizer.trained:
                dibujar_marco(frame, face.bbox, COL_DENY, 2)
            else:
                pid, sim = recognizer.predict(face.normed_embedding)
                reconocido = sim >= RECOGNIZE_SIM and pid in personas

                if not reconocido:
                    # Desconocido -> denegar (sin spamear el log).
                    pending_id, pending_count = None, 0
                    dibujar_marco(frame, face.bbox, COL_DENY, 2)
                    cv2.putText(frame, "No reconocido", (int(face.bbox[0]),
                                int(face.bbox[1]) - 10), FONT, 0.6, COL_DENY, 2,
                                lineType=cv2.LINE_AA)
                    if now < banner_until:
                        pass
                    elif now - last_deny_log >= DENY_LOG_COOLDOWN:
                        registrar_acceso("-", "-", "DENEGADO", "DESCONOCIDO", sim)
                        last_deny_log = now
                        banner_text, banner_sub = "ACCESO DENEGADO", "No autorizado"
                        banner_color, banner_until = COL_DENY, now + BANNER_SECONDS
                else:
                    persona = personas[pid]
                    # estabilidad temporal
                    if pending_id == pid:
                        pending_count += 1
                    else:
                        pending_id, pending_count = pid, 1
                        eyes_closed, last_blink = False, 0.0   # parpadeo nuevo por persona

                    # LIVENESS ATADO A ESTA CARA: tomamos el parpadeo de la cara
                    # del landmarker que mas solapa (IoU) con la bbox reconocida.
                    # Asi, una foto de un autorizado NO se valida con el parpadeo
                    # de otra persona parada al lado.
                    if liveness_on:
                        bbox_rec = tuple(float(x) for x in face.bbox)
                        mejor_sc, mejor_iou = None, 0.0
                        for bb, sc in lm_faces:
                            v = iou(bb, bbox_rec)
                            if v > mejor_iou:
                                mejor_iou, mejor_sc = v, sc
                        if mejor_sc is not None and mejor_iou >= IOU_MIN:
                            if not eyes_closed and mejor_sc > BLINK_CLOSE:
                                eyes_closed = True
                            elif eyes_closed and mejor_sc < BLINK_OPEN:
                                eyes_closed = False
                                last_blink = now

                    autorizada = persona.get("autorizado", True)
                    esta_vigente = vigente(persona)
                    en_cooldown = (now - ultimo_acceso.get(pid, 0)) < ACCESO_COOLDOWN
                    liveness_ok = (not liveness_on
                                   or (now - last_blink) < BLINK_FRESH_SECONDS)

                    col = COL_GRANT if (autorizada and esta_vigente) else COL_DENY
                    dibujar_marco(frame, face.bbox, col, 2)
                    cv2.putText(frame, persona["nombre"], (int(face.bbox[0]),
                                int(face.bbox[1]) - 10), FONT_BOLD, 0.7, col, 2,
                                lineType=cv2.LINE_AA)

                    if pending_count >= STABLE_FRAMES and not en_cooldown \
                            and now >= banner_until:
                        if not autorizada:
                            registrar_acceso(persona["nombre"],
                                             persona.get("documento", "-"),
                                             "DENEGADO", "DADA DE BAJA", sim)
                            banner_text, banner_sub = "ACCESO DENEGADO", "Dada de baja"
                            banner_color = COL_DENY
                            banner_until = now + BANNER_SECONDS
                            ultimo_acceso[pid] = now
                        elif not esta_vigente:
                            registrar_acceso(persona["nombre"],
                                             persona.get("documento", "-"),
                                             "DENEGADO", "VENCIDA", sim)
                            banner_text, banner_sub = "ACCESO DENEGADO", "Autorizacion vencida"
                            banner_color = COL_DENY
                            banner_until = now + BANNER_SECONDS
                            ultimo_acceso[pid] = now
                        elif not liveness_ok:
                            texto_centrado(frame, "Mire a la camara y parpadee",
                                           h - 90, FONT_BOLD, 0.9, COL_BRAND, 2)
                        else:
                            # AUTORIZADO
                            abrir_puerta(persona)
                            registrar_acceso(persona["nombre"],
                                             persona.get("documento", "-"),
                                             "AUTORIZADO", "OK", sim)
                            hora = dt.datetime.now().strftime("%H:%M")
                            banner_text = "ACCESO AUTORIZADO"
                            banner_sub = f"{persona['nombre']} - {hora}"
                            banner_color = COL_GRANT
                            banner_until = now + BANNER_SECONDS
                            door_until = now + DOOR_OPEN_SECONDS
                            ultimo_acceso[pid] = now
                            print(f"[ACCESO] {hora} {persona['nombre']} "
                                  f"AUTORIZADO (sim {sim:.2f})")
                            pending_count = 0

        # ---------------- HUD ----------------
        panel(frame, 0, 0, w, 46)
        ahora = dt.datetime.now()
        cv2.putText(frame, "CONTROL DE ACCESO", (16, 30), FONT_BOLD, 0.8,
                    COL_BRAND, 2, lineType=cv2.LINE_AA)
        (aw, _), _ = cv2.getTextSize("CONTROL DE ACCESO", FONT_BOLD, 0.8, 2)
        cv2.putText(frame, "  ·  " + INSTALACION, (16 + aw, 29), FONT, 0.6,
                    COL_DIM, 1, lineType=cv2.LINE_AA)
        reloj = ahora.strftime("%H:%M:%S")
        (rw, _), _ = cv2.getTextSize(reloj, FONT_BOLD, 0.9, 2)
        cv2.putText(frame, reloj, (w - rw - 16, 32), FONT_BOLD, 0.9, COL_BRAND, 2,
                    lineType=cv2.LINE_AA)

        # Banner de verdicto (centro).
        if stage == IDLE and now < banner_until:
            panel(frame, 0, h // 2 - 50, w, h // 2 + 50, alpha=0.65)
            texto_centrado(frame, banner_text, h // 2 - 8, FONT_BOLD, 1.1,
                           banner_color, 3)
            texto_centrado(frame, banner_sub, h // 2 + 30, FONT, 0.7, COL_INK, 2)

        # Indicador "PUERTA ABIERTA".
        if stage == IDLE and now < door_until:
            panel(frame, 0, 52, w, 92, alpha=0.6)
            texto_centrado(frame, "PUERTA ABIERTA", 80, FONT_BOLD, 0.9,
                           COL_GRANT, 2)

        if stage == IDLE and not recognizer.trained and now >= banner_until:
            panel(frame, 0, h - 60, w, h)
            texto_centrado(frame, "Sin personas. Presiona 'e' para dar de alta.",
                           h - 22, FONT, 0.65, COL_DIM, 2)

        panel(frame, 0, h - 24, w, h, alpha=0.5)
        cv2.putText(frame, "[e] alta  [l] listar  [r] baja  [q] salir",
                    (12, h - 7), FONT, 0.5, COL_DIM, 1, lineType=cv2.LINE_AA)
        aviso = "Acceso por consentimiento - datos biometricos cifrados"
        (aw2, _), _ = cv2.getTextSize(aviso, FONT, 0.45, 1)
        cv2.putText(frame, aviso, (w - aw2 - 12, h - 7), FONT, 0.45, COL_DIM, 1,
                    lineType=cv2.LINE_AA)

        cv2.imshow("Control de Acceso Facial", frame)

        # ---------------- TECLADO ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue

        # Entrada de texto (nombre / documento / id a revocar).
        if stage in (NAME, DOC, REVOKE):
            if key in (13, 10):
                if stage == NAME and buf_nombre.strip():
                    stage, buf_doc = DOC, ""
                elif stage == DOC and buf_doc.strip():
                    stage, tipo_visita = TIPO, False
                elif stage == REVOKE:
                    try:
                        rid = int(buf_revoke.strip())
                    except ValueError:
                        rid = None
                    if rid is not None and rid in personas:
                        personas[rid]["autorizado"] = not personas[rid].get(
                            "autorizado", True)
                        guardar_personas(personas)
                        est = "AUTORIZADA" if personas[rid]["autorizado"] else "DADA DE BAJA"
                        print(f"Persona id {rid} ahora: {est}")
                    else:
                        print("ID inexistente.")
                    stage, buf_revoke = IDLE, ""
            elif key == 27:
                stage = IDLE
                enroll_embs, buf_revoke = [], ""
                print("Operacion cancelada.")
            elif key in (8, 127):
                if stage == NAME:
                    buf_nombre = buf_nombre[:-1]
                elif stage == DOC:
                    buf_doc = buf_doc[:-1]
                else:
                    buf_revoke = buf_revoke[:-1]
            elif 32 <= key <= 126:
                ch = chr(key)
                if stage == NAME:
                    buf_nombre += ch
                elif stage == DOC:
                    buf_doc += ch
                elif ch.isdigit():
                    buf_revoke += ch
            continue

        # Seleccion de tipo de acceso.
        if stage == TIPO:
            if key in (ord('p'), ord('P')):
                tipo_visita = False
                stage = CONSENT
            elif key in (ord('v'), ord('V')):
                tipo_visita = True
                stage = CONSENT
            elif key == 27:
                stage, enroll_embs = IDLE, []
                print("Alta cancelada.")
            continue

        # Consentimiento -> guarda la persona.
        if stage == CONSENT:
            if key in (ord('s'), ord('S'), 13, 10):
                pid = next_id
                next_id += 1
                personas[pid] = {
                    "nombre": buf_nombre.strip(),
                    "documento": buf_doc.strip(),
                    "autorizado": True,
                    "valido_hasta": (dt.date.today().isoformat()
                                     if tipo_visita else None),
                    "consentimiento": True,
                    "fecha_consentimiento": dt.datetime.now().isoformat(timespec="seconds"),
                    "fecha_alta": dt.date.today().isoformat(),
                }
                recognizer.add(enroll_embs, pid)
                recognizer.save(FACES_DB_PATH)
                guardar_personas(personas)
                tipo = "visita (hoy)" if tipo_visita else "permanente"
                print(f"Alta OK: {buf_nombre.strip()} [{tipo}] - consentimiento "
                      f"registrado.")
                banner_text = "PERSONA REGISTRADA"
                banner_sub = buf_nombre.strip()
                banner_color, banner_until = COL_GRANT, now + BANNER_SECONDS
                stage, enroll_embs = IDLE, []
            elif key == 27:
                stage, enroll_embs = IDLE, []
                print("Alta cancelada (sin consentimiento).")
            continue

        # Teclas en modo operativo.
        if key == ord('q'):
            break
        elif key == ord('e'):
            stage, enroll_embs = CAPTURE, []
            print("Alta: mante la cara centrada mientras captura...")
        elif key == ord('l'):
            listar_personas(personas)
        elif key == ord('r'):
            stage, buf_revoke = REVOKE, ""
            listar_personas(personas)
            print("Escribi el ID a dar de baja/reactivar y Enter.")

    cap.release()
    cv2.destroyAllWindows()
    if landmarker is not None:
        landmarker.close()


if __name__ == "__main__":
    main()
