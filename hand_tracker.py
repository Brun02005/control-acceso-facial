"""
Real-time hand-tracking app: PROGRESSIVE TEXT REVEAL + fun extras.

Features
--------
1. Progressive text reveal driven by the distance between your two hands.
2. Dynamic box color: shifts red -> yellow -> green as you reveal more letters.
3. Particles / sparks that burst from your hands each time a new letter appears.
4. Multiple words: cycle them with the keyboard OR by making a fist with both hands.
5. Face pixelation (privacy) using MediaPipe's BlazeFace detector.
6. Chest/logo pixelation: a region below the face is pixelated to hide a shirt logo.
7. Video recording to an .mp4 file with a single key.

Model files (must sit next to this script)
-------------------------------------------
  hand_landmarker.task          -> hand tracking
  blaze_face_short_range.tflite -> face detection (for pixelation)

Dependencies
------------
    pip install opencv-contrib-python mediapipe numpy

Keyboard controls
-----------------
    q        -> quit
    n / .    -> next word
    p / ,    -> previous word
    1..9     -> jump directly to word #N (if it exists)
    f        -> toggle face pixelation on/off
    l        -> toggle logo/chest pixelation on/off
    r        -> start / stop video recording
"""

import os
import time
import json
import random
import hashlib

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------------------------
# CONFIGURATION  --  tweak these to customize behavior
# ---------------------------------------------------------------------------

# The words you can cycle through. The FIRST one is shown at startup. Change /
# add strings freely; the reveal and box auto-adapt to any length.
WORDS = ["PUTO", "CRACK", "GENIO", "LOCO", "PALABRA"]

# Model file paths (must sit next to this script).
_HERE = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(_HERE, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(_HERE, "blaze_face_short_range.tflite")

# --- Progressive-reveal distance thresholds --------------------------------
# Expressed as a normalized gap (palm distance / average hand size), so they
# behave consistently regardless of how close you are to the camera.
#   normalized_gap < GAP_MIN -> 0 characters, box NOT drawn
#   normalized_gap > GAP_MAX -> full word
#   in between               -> linearly interpolated character count
GAP_MIN = 0.8   # hands this close (or closer): show nothing
GAP_MAX = 2.2   # hands this far apart (or farther): show everything

# Smoothing factor for the box position (0..1). Lower = smoother but laggier.
SMOOTHING_ALPHA = 0.4

# Visual styling for the red sign.
BOX_COLOR = (0, 0, 255)          # Default/base red (BGR). Used if dynamic off.
BOX_BORDER_THICKNESS = 8         # Outer border thickness in pixels.
TEXT_COLOR = (255, 255, 255)     # White text for contrast.
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_THICKNESS = 6
BOX_PADDING_X = 45               # Horizontal padding around text (px).
BOX_PADDING_Y = 30               # Vertical padding around text (px).

# Toggle dynamic color (red->yellow->green with reveal progress).
DYNAMIC_COLOR = True

# --- Pixelation / censoring (privacy) --------------------------------------
PIXELATE_FACE = False            # Start OFF; toggle at runtime with the 'f' key.
PIXELATE_LOGO = False            # Start OFF; toggle at runtime with the 'l' key.
PIXEL_BLOCKS = 12                # Blocks for the LOGO. Fewer = chunkier.
FACE_DETECT_EVERY = 2            # Run face detection every N frames (perf).

# The FACE now uses a SMOOTH censor (aesthetic) instead of hard blocks:
#   1. downscale to FACE_CENSOR_RES so all identity detail is destroyed,
#   2. upscale back SMOOTHLY + Gaussian blur -> soft frosted blob (no hard edges),
#   3. feather it through an ELLIPSE mask so it blends into the surroundings.
FACE_CENSOR_RES = 8              # Internal resolution of the blob. Lower = more
                                 # abstract/hidden (6-10 looks good).
FACE_EXPAND = 0.35               # Grow the face box on each side (hair/chin/ears).
FACE_FEATHER = 0.12              # Softness of the oval edge (fraction of size).

# --- "Hacker" facial-scanner HUD -------------------------------------------
# A fake green targeting box that appears over any detected face, with garbled
# NAME / SURNAME / NATIONALITY fields (glitchy chars that keep failing to lock).
SHOW_SCANNER = True              # Start ON; toggle at runtime with the 's' key.
SCANNER_GREEN = (0, 255, 0)      # BGR bright green.
GARBLE_CHARS = "#@%&$?!*/\\0123456789ABCDEF"   # pool for corrupted values.

# --- REAL facial recognition (lightweight, NumPy-only) ---------------------
# You ENROLL faces (press 'e' and type a name); the scanner then shows the real
# name in green when it recognizes someone, or the glitch when it doesn't.
# The face database + name map persist to disk between sessions.
#
# How it works: each face crop is turned into a normalized vector; recognition
# is nearest-neighbor by cosine similarity. Basic (sensitive to big lighting /
# pose changes) but dependency-free and enough to recognize faces you enroll.
FACES_DB_PATH = os.path.join(_HERE, "faces.npz")       # enrolled face vectors
LABELS_PATH = os.path.join(_HERE, "labels.json")       # id -> name mapping
RECOGNIZE_SIM = 0.72             # cosine similarity to accept a match (0..1).
                                 # Higher = stricter. Lower if it won't match you.
ENROLL_SAMPLES = 30              # face crops captured per enrollment.
FACE_CROP_SIZE = 200             # detected-face crop size before vectorizing.
RECOG_VEC_SIZE = 100             # faces are compared at this resolution.

# The chest/logo box is derived from the detected face box. These multipliers
# place and size it. Tweak if the pixelation doesn't cover your shirt logo.
LOGO_W_SCALE = 1.7               # chest width  = face_w * this
LOGO_H_SCALE = 1.5               # chest height = face_h * this
LOGO_Y_OFFSET = 1.4              # top of chest = face_bottom + face_h * this
LOGO_PAD_FACTOR = 0.0            # extra vertical gap below chin (0 = none)

# --- Particles -------------------------------------------------------------
PARTICLES_PER_BURST = 14         # sparks spawned per hand when a letter reveals
PARTICLE_LIFE = 0.6              # seconds each spark lives
PARTICLE_GRAVITY = 900.0         # downward acceleration (px/s^2)


# ---------------------------------------------------------------------------
# LANDMARK CONSTANTS  (MediaPipe Hands has 21 landmarks per hand)
# ---------------------------------------------------------------------------

WRIST = 0
MIDDLE_TIP = 12
# Fingertip / PIP-joint pairs used to tell if a finger is extended or curled.
# (index, middle, ring, pinky) -> (tip_index, pip_index)
FINGER_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---------------------------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------------------------

def landmark_to_np(landmark):
    """Convert a single MediaPipe landmark to a numpy [x, y] array."""
    return np.array([landmark.x, landmark.y], dtype=np.float32)


def hand_size(hand):
    """Characteristic hand size = wrist -> middle-fingertip distance.

    This is our YARDSTICK: the gap between hands is divided by it so the reveal
    is invariant to how close the hands are to the camera.
    """
    wrist = landmark_to_np(hand[WRIST])
    middle_tip = landmark_to_np(hand[MIDDLE_TIP])
    return max(np.linalg.norm(middle_tip - wrist), 1e-6)


def hand_centroid(hand):
    """Average (x, y) of all 21 landmarks = the hand's center of mass."""
    pts = np.array([[lm.x, lm.y] for lm in hand], dtype=np.float32)
    return pts.mean(axis=0)


def compute_normalized_gap(hand_a, hand_b):
    """Gap between two hands divided by average hand size (scale-independent).

    1. yardstick = average of the two hands' wrist->middle-tip lengths.
    2. palm_gap  = distance between the two hand centroids.
    3. return palm_gap / yardstick.
    """
    yardstick = (hand_size(hand_a) + hand_size(hand_b)) / 2.0
    palm_gap = np.linalg.norm(hand_centroid(hand_a) - hand_centroid(hand_b))
    return palm_gap / yardstick


def gap_to_char_count(normalized_gap, text):
    """Map a normalized gap to how many characters of `text` to reveal.

    Below GAP_MIN -> 0; above GAP_MAX -> all; in between -> linear interpolation.
    """
    if normalized_gap <= GAP_MIN:
        return 0
    if normalized_gap >= GAP_MAX:
        return len(text)
    fraction = (normalized_gap - GAP_MIN) / (GAP_MAX - GAP_MIN)
    return max(0, min(len(text), int(round(fraction * len(text)))))


def is_fist(hand):
    """Return True if the hand looks like a closed fist.

    For each of the four fingers, the finger is "extended" when its TIP is
    farther from the wrist than its PIP joint. A fist has (almost) no extended
    fingers, so we call it a fist when 0 or 1 fingers are extended.
    """
    wrist = landmark_to_np(hand[WRIST])
    extended = 0
    for tip_idx, pip_idx in FINGER_TIP_PIP:
        tip_d = np.linalg.norm(landmark_to_np(hand[tip_idx]) - wrist)
        pip_d = np.linalg.norm(landmark_to_np(hand[pip_idx]) - wrist)
        if tip_d > pip_d:
            extended += 1
    return extended <= 1


# ---------------------------------------------------------------------------
# COLOR HELPER
# ---------------------------------------------------------------------------

def progress_color(fraction):
    """Return a BGR color that goes red -> yellow -> green as fraction 0->1.

    We build it in HSV (hue 0=red .. 60=green in OpenCV's 0-179 hue scale) and
    convert to BGR. This gives a smooth, saturated gradient.
    """
    hue = int(60 * max(0.0, min(1.0, fraction)))   # 0 (red) .. 60 (green)
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ---------------------------------------------------------------------------
# PIXELATION
# ---------------------------------------------------------------------------

def pixelate_region(frame, x1, y1, x2, y2, blocks=PIXEL_BLOCKS):
    """Pixelate the rectangle (x1,y1)-(x2,y2) of `frame` in place.

    Trick: shrink the region down to `blocks` x `blocks` (averaging colors),
    then scale it back up with NEAREST interpolation so each block is a solid
    square -> classic "censored" mosaic look.
    """
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = frame[y1:y2, x1:x2]
    small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(
        small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
    )


def _garble(rng, n):
    """Return a string of `n` random 'corrupted' characters."""
    return "".join(rng.choice(GARBLE_CHARS) for _ in range(n))


def get_face_gray(gray_frame, box, size=FACE_CROP_SIZE):
    """Crop a face from a grayscale frame and resize to the fixed LBPH size.

    Returns None if the box is empty/out of bounds. LBPH needs every training
    and query image to be the SAME size, hence the resize to size x size.
    """
    h, w = gray_frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return cv2.resize(gray_frame[y1:y2, x1:x2], (size, size))


class SimpleFaceRecognizer:
    """Minimal, dependency-free face recognizer (nearest-neighbor).

    Each enrolled face crop is converted to a normalized vector:
        resize -> equalizeHist (lighting) -> blur (denoise) -> flatten ->
        subtract mean -> L2-normalize.
    Recognition finds the enrolled vector with the highest cosine similarity
    (a dot product, since vectors are unit-length). If that similarity clears
    `threshold`, we accept it as a match.
    """

    def __init__(self, size=RECOG_VEC_SIZE, threshold=RECOGNIZE_SIM):
        self.size = size
        self.threshold = threshold
        self.vectors = None    # (N, D) float32, each L2-normalized
        self.ids = None        # (N,) int32 label per stored vector

    def _vec(self, gray_crop):
        """Turn a grayscale face crop into a normalized feature vector."""
        g = cv2.resize(gray_crop, (self.size, self.size))
        g = cv2.equalizeHist(g)                 # cancel overall brightness
        g = cv2.GaussianBlur(g, (3, 3), 0)      # smooth out sensor noise
        v = g.astype(np.float32).flatten()
        v -= v.mean()                           # remove DC component
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else v

    @property
    def trained(self):
        return self.vectors is not None and len(self.vectors) > 0

    def add(self, crops, label):
        """Enroll a batch of crops under an integer label."""
        vecs = np.array([self._vec(c) for c in crops], dtype=np.float32)
        ids = np.full(len(crops), label, dtype=np.int32)
        if self.vectors is None:
            self.vectors, self.ids = vecs, ids
        else:
            self.vectors = np.vstack([self.vectors, vecs])
            self.ids = np.concatenate([self.ids, ids])

    def predict(self, gray_crop):
        """Return (label, similarity) of the closest enrolled face."""
        if not self.trained:
            return None, 0.0
        v = self._vec(gray_crop)
        sims = self.vectors @ v                 # cosine similarity to each
        i = int(np.argmax(sims))
        return int(self.ids[i]), float(sims[i])

    def save(self, path):
        np.savez(path, vectors=self.vectors, ids=self.ids)

    def load(self, path):
        data = np.load(path)
        self.vectors = data["vectors"].astype(np.float32)
        self.ids = data["ids"].astype(np.int32)


def draw_face_scanner(frame, box, now, identity=None):
    """Draw a 'hacker' facial-recognition HUD over a detected face.

    If `identity` is None -> the glitch look (garbled fields, "NO IDENTIFICADO").
    If `identity` is (name, match_pct) -> a solid green "IDENTIFICADO" panel with
    the REAL recognized name. `box` is (x1, y1, x2, y2) px; `now` is time.time().
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 10 or bh < 10:
        return
    green = SCANNER_GREEN
    recognized = identity is not None

    # 1) Faint full outline (brighter when we have a real match).
    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  (0, 160, 0) if recognized else (0, 120, 0), 1)

    # 2) Bright green corner brackets (targeting-reticle look).
    seg = max(12, int(min(bw, bh) * 0.22))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * seg, cy), green, 2)
        cv2.line(frame, (cx, cy), (cx, cy + dy * seg), green, 2)

    # 3) Horizontal scan line sweeping down the face over time.
    phase = (now * 0.8) % 1.0
    scan_y = int(y1 + phase * bh)
    cv2.line(frame, (x1, scan_y), (x2, scan_y), (0, 255, 140), 1)

    # 4) Info panel text: real identity, or the glitch.
    if recognized:
        name, pct = identity
        # Stable fake ID derived from the name (same every time for that person).
        sid = hashlib.md5(name.encode()).hexdigest()[:6].upper()
        lines = [
            (f"SUBJECT ID: 0x{sid}", green),
            (f"NOMBRE: {name.upper()}", green),
            (f"MATCH: {pct}%", green),
            ("STATUS: IDENTIFICADO", green),
        ]
        header = "MATCH FOUND"
    else:
        # Values refresh ~3x/sec (a time "bucket") so the glitch flickers without
        # being an unreadable strobe. A private RNG avoids touching the particle
        # system's global random state.
        rng = random.Random(int(now * 3))
        blink_on = int(now * 2) % 2 == 0
        lines = [
            (f"SUBJECT ID: 0x{_garble(rng, 6)}", green),
            (f"NOMBRE: {_garble(rng, rng.randint(5, 8))}", green),
            (f"APELLIDO: {_garble(rng, rng.randint(5, 9))}", green),
            (f"NACIONALIDAD: {rng.choice(['???', '//ERR', 'UNKNOWN', 'N/A'])}", green),
            (f"MATCH: {rng.randint(0, 37)}.{rng.randint(0, 9)}%", green),
            ("STATUS: NO IDENTIFICADO",
             (0, 0, 255) if blink_on else (0, 140, 255)),
        ]
        header = None

    font, fs, th, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 22

    # Panel width from the widest line.
    panel_w = max(cv2.getTextSize(t, font, fs, th)[0][0] for t, _ in lines) + 16
    panel_h = lh * len(lines) + 12

    # Prefer the right of the face; flip to the left if it won't fit.
    px = x2 + 12
    if px + panel_w > w:
        px = x1 - panel_w - 12
    px = max(5, min(px, w - panel_w - 5))
    py = max(5, min(y1, h - panel_h - 5))

    # Semi-transparent dark backdrop for readability.
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (0, 90, 0), 1)

    # Header: solid "MATCH FOUND" when recognized, else blinking "SCANNING...".
    if recognized:
        head_text = header
    else:
        head_text = "SCANNING" + "." * (1 + int(now * 4) % 3)
    cv2.putText(frame, head_text, (px + 8, py - 6),
                font, 0.5, green, 1, lineType=cv2.LINE_AA)

    for i, (text, color) in enumerate(lines):
        ty = py + 8 + lh * (i + 1) - 6
        cv2.putText(frame, text, (px + 8, ty), font, fs, color, th,
                    lineType=cv2.LINE_AA)


def censor_face_smooth(frame, x1, y1, x2, y2):
    """Aesthetically censor a face region in place.

    Unlike hard pixelation, this destroys identity but LOOKS soft:
      1. Downscale the ROI to FACE_CENSOR_RES x FACE_CENSOR_RES (all detail gone).
      2. Upscale back with LINEAR interpolation (smooth gradient, not blocks)
         and add a Gaussian blur -> a soft "frosted" blob.
      3. Blend it back through a FEATHERED ELLIPSE mask, so the censored area is
         oval with soft edges that melt into the surrounding image instead of a
         harsh rectangle.
    """
    h, w = frame.shape[:2]
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(w, int(x2)), min(h, int(y2))
    rw, rh = x2i - x1i, y2i - y1i
    if rw < 8 or rh < 8:
        return

    roi = frame[y1i:y2i, x1i:x2i]

    # 1) Downscale to kill detail, 2) smooth upscale + blur -> frosted blob.
    small = cv2.resize(roi, (FACE_CENSOR_RES, FACE_CENSOR_RES),
                       interpolation=cv2.INTER_LINEAR)
    blob = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
    k = (max(11, min(rw, rh) // 3) | 1)          # odd kernel size
    blob = cv2.GaussianBlur(blob, (k, k), 0)

    # 3) Feathered elliptical mask (float 0..1) so edges blend smoothly.
    mask = np.zeros((rh, rw), dtype=np.float32)
    cv2.ellipse(mask, (rw // 2, rh // 2),
                (int(rw * 0.52), int(rh * 0.58)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3.0, rw * FACE_FEATHER))
    mask = mask[..., None]                        # add channel axis for blending

    frame[y1i:y2i, x1i:x2i] = (blob * mask + roi * (1.0 - mask)).astype(np.uint8)


def face_boxes_to_pixels(detection_result, w, h):
    """Convert a FaceDetector result to a list of (x1,y1,x2,y2) pixel boxes."""
    boxes = []
    if not detection_result or not detection_result.detections:
        return boxes
    for det in detection_result.detections:
        bb = det.bounding_box               # origin_x, origin_y, width, height
        x1, y1 = bb.origin_x, bb.origin_y
        boxes.append((x1, y1, x1 + bb.width, y1 + bb.height))
    return boxes


# ---------------------------------------------------------------------------
# PARTICLES
# ---------------------------------------------------------------------------

class Particle:
    """A single spark: position, velocity, remaining life and color."""
    __slots__ = ("x", "y", "vx", "vy", "life", "color")

    def __init__(self, x, y, color):
        self.x = float(x)
        self.y = float(y)
        # Random outward velocity.
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(120, 360)
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed - 120     # bias slightly upward
        self.life = PARTICLE_LIFE
        self.color = color

    def update(self, dt):
        """Advance physics by dt seconds. Returns True while still alive."""
        self.vy += PARTICLE_GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        return self.life > 0

    def draw(self, frame):
        # Fade + shrink as the spark dies.
        t = max(0.0, self.life / PARTICLE_LIFE)
        radius = max(1, int(5 * t))
        cv2.circle(frame, (int(self.x), int(self.y)), radius, self.color, -1,
                   lineType=cv2.LINE_AA)


def spawn_burst(particles, center_px, color):
    """Add a burst of sparks at a pixel position."""
    for _ in range(PARTICLES_PER_BURST):
        particles.append(Particle(center_px[0], center_px[1], color))


# ---------------------------------------------------------------------------
# RENDERING
# ---------------------------------------------------------------------------

def draw_hand_skeleton(frame, hand):
    """Draw the 21-landmark skeleton for one hand."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, lineType=cv2.LINE_AA)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)


def draw_sign(frame, center_xy, text, box_color):
    """Draw a solid rectangle (with border) and centered white text.

    The box is measured from the CURRENT text via cv2.getTextSize + padding, so
    it grows/shrinks to fit exactly the letters revealed this frame.
    """
    h, w = frame.shape[:2]
    cx, cy = int(center_xy[0]), int(center_xy[1])
    font_scale = max(0.8, w / 500.0)

    (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, FONT_THICKNESS)

    half_w = text_w // 2 + BOX_PADDING_X
    half_h = text_h // 2 + BOX_PADDING_Y
    x1, y1 = max(0, cx - half_w), max(0, cy - half_h)
    x2, y2 = min(w - 1, cx + half_w), min(h - 1, cy + half_h)

    # Fill + a darker border of the same hue for definition.
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness=cv2.FILLED)
    darker = tuple(int(c * 0.6) for c in box_color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), darker, BOX_BORDER_THICKNESS)

    text_x = cx - text_w // 2
    text_y = cy + text_h // 2
    cv2.putText(frame, text, (text_x, text_y), FONT, font_scale,
                TEXT_COLOR, FONT_THICKNESS, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    # --- Sanity-check model files ------------------------------------------
    if not os.path.exists(HAND_MODEL_PATH):
        print(f"ERROR: falta {HAND_MODEL_PATH}")
        return
    face_available = os.path.exists(FACE_MODEL_PATH)
    if not face_available:
        print("AVISO: falta blaze_face_short_range.tflite -> pixelado desactivado")

    # --- Build the HandLandmarker (Tasks API, VIDEO mode) ------------------
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(hand_options)

    # --- Build the FaceDetector (for pixelation) ---------------------------
    face_detector = None
    if face_available:
        face_options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
        )
        face_detector = vision.FaceDetector.create_from_options(face_options)

    # --- Build the face recognizer (real recognition) ----------------------
    recognizer = SimpleFaceRecognizer()
    labels = {}                # id (int) -> name (str)
    name_to_id = {}            # name -> id (reverse map)
    next_id = 0
    # Load a previously enrolled database + name map if they exist.
    if os.path.exists(FACES_DB_PATH) and os.path.exists(LABELS_PATH):
        try:
            recognizer.load(FACES_DB_PATH)
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                labels = {int(k): v for k, v in json.load(f).items()}
            name_to_id = {v: k for k, v in labels.items()}
            next_id = max(labels) + 1 if labels else 0
            print(f"Reconocedor cargado: {list(labels.values())}")
        except Exception as exc:
            print(f"AVISO: no se pudo cargar el reconocedor ({exc})")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la webcam.")
        return

    # --- Runtime state ------------------------------------------------------
    word_index = 0                 # which word from WORDS is active
    pixelate_face = PIXELATE_FACE
    pixelate_logo = PIXELATE_LOGO
    show_scanner = SHOW_SCANNER
    smoothed_center = None         # EMA-smoothed box position
    prev_char_count = 0            # to detect NEW letters (for particle bursts)
    particles = []                 # active sparks
    last_face_boxes = []           # cached face boxes between detections
    fist_was_active = False        # edge-trigger for the fist word-switch gesture
    recording = False
    writer = None
    enrolling = False              # currently capturing faces to enroll?
    enroll_crops = []              # captured grayscale face crops
    naming = False                 # typing the name for a just-captured face?
    name_buffer = ""               # the name being typed in-window
    frame_idx = 0
    start_time = time.time()
    prev_time = start_time

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: No se pudo leer un frame de la webcam.")
            break

        frame = cv2.flip(frame, 1)          # mirror for intuitive interaction
        h, w = frame.shape[:2]
        now = time.time()
        dt = max(1e-3, now - prev_time)     # seconds since last frame
        prev_time = now
        frame_idx += 1

        display_text = WORDS[word_index]

        # RGB image shared by both MediaPipe tasks.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((now - start_time) * 1000)

        # CLEAN grayscale (before any censoring) for recognition/enrollment.
        clean_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------- FACE: PIXELATION + SCANNER (on raw frame first) ----
        if face_detector is not None and (pixelate_face or pixelate_logo
                                          or show_scanner or enrolling):
            # Detect only every N frames for performance; reuse cached boxes.
            if frame_idx % FACE_DETECT_EVERY == 0:
                face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
                last_face_boxes = face_boxes_to_pixels(face_result, w, h)

            # The largest face is the one we enroll (ignore background faces).
            largest_box = None
            if last_face_boxes:
                largest_box = max(last_face_boxes,
                                  key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

            for (fx1, fy1, fx2, fy2) in last_face_boxes:
                fw, fh = fx2 - fx1, fy2 - fy1

                # --- REAL recognition: predict identity from the CLEAN crop ---
                identity = None
                if recognizer.trained and not enrolling:
                    crop = get_face_gray(clean_gray, (fx1, fy1, fx2, fy2))
                    if crop is not None:
                        label, sim = recognizer.predict(crop)
                        if sim >= RECOGNIZE_SIM:
                            pct = int(max(0, min(99, sim * 100)))
                            identity = (labels.get(label, "???"), pct)

                # --- Enrollment: collect crops of the largest face ------------
                if enrolling and (fx1, fy1, fx2, fy2) == largest_box:
                    crop = get_face_gray(clean_gray, (fx1, fy1, fx2, fy2))
                    if crop is not None and len(enroll_crops) < ENROLL_SAMPLES:
                        enroll_crops.append(crop)

                if pixelate_face:
                    # Expand the box outward so hair, ears and chin are covered,
                    # then apply the smooth (aesthetic) censor.
                    ex, ey = fw * FACE_EXPAND, fh * FACE_EXPAND
                    censor_face_smooth(frame, fx1 - ex, fy1 - ey,
                                       fx2 + ex, fy2 + ey)
                if pixelate_logo:
                    # Chest/logo box derived from the face box.
                    lw = fw * LOGO_W_SCALE
                    lh = fh * LOGO_H_SCALE
                    lcx = fx1 + fw / 2.0
                    ly1 = fy2 + fh * LOGO_PAD_FACTOR + fh * (LOGO_Y_OFFSET - 1.0)
                    pixelate_region(frame, lcx - lw / 2.0, ly1,
                                    lcx + lw / 2.0, ly1 + lh)
                if show_scanner:
                    # Drawn AFTER censoring so the green HUD sits on top.
                    draw_face_scanner(frame, (fx1, fy1, fx2, fy2), now, identity)

            # --- Enough samples captured -> switch to in-window name entry ---
            # (We DON'T use input() here: it would freeze the OpenCV window and
            # Windows would kill the app. Instead we type the name in-window.)
            if enrolling and len(enroll_crops) >= ENROLL_SAMPLES:
                enrolling = False
                naming = True
                name_buffer = ""
                print(f"Capturadas {len(enroll_crops)} muestras. "
                      "Escribi el nombre en la ventana y presiona Enter.")

        # ---------------- HAND TRACKING & TEXT REVEAL ------------------------
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        current_text = ""
        hands_center_norm = None
        char_count = 0
        normalized_gap = 0.0
        both_fists = False

        if result.hand_landmarks and len(result.hand_landmarks) == 2:
            hand_a, hand_b = result.hand_landmarks
            draw_hand_skeleton(frame, hand_a)
            draw_hand_skeleton(frame, hand_b)

            # Fist gesture (both hands) cycles to the next word (edge-triggered).
            both_fists = is_fist(hand_a) and is_fist(hand_b)

            normalized_gap = compute_normalized_gap(hand_a, hand_b)
            char_count = gap_to_char_count(normalized_gap, display_text)
            current_text = display_text[:char_count]

            c_a = hand_centroid(hand_a)
            c_b = hand_centroid(hand_b)
            hands_center_norm = (c_a + c_b) / 2.0

            # Spawn a spark burst at each hand whenever a NEW letter appears.
            if char_count > prev_char_count:
                color = progress_color(char_count / max(1, len(display_text)))
                spawn_burst(particles, (c_a[0] * w, c_a[1] * h), color)
                spawn_burst(particles, (c_b[0] * w, c_b[1] * h), color)

            cv2.putText(frame, f"gap: {normalized_gap:.2f}  chars: {char_count}",
                        (10, 60), FONT, 0.6, (0, 255, 255), 2,
                        lineType=cv2.LINE_AA)

        prev_char_count = char_count

        # Fist word-switch: fire once on the rising edge (fist just formed).
        if both_fists and not fist_was_active:
            word_index = (word_index + 1) % len(WORDS)
            prev_char_count = 0
        fist_was_active = both_fists

        # ---------------- PARTICLES: update + draw ---------------------------
        particles = [p for p in particles if p.update(dt)]
        for p in particles:
            p.draw(frame)

        # ---------------- SIGN (dynamic color) -------------------------------
        if current_text and hands_center_norm is not None:
            target_center = np.array(
                [hands_center_norm[0] * w, hands_center_norm[1] * h],
                dtype=np.float32,
            )
            if smoothed_center is None:
                smoothed_center = target_center
            else:
                smoothed_center = (SMOOTHING_ALPHA * target_center
                                   + (1.0 - SMOOTHING_ALPHA) * smoothed_center)

            if DYNAMIC_COLOR:
                box_color = progress_color(char_count / max(1, len(display_text)))
            else:
                box_color = BOX_COLOR
            draw_sign(frame, smoothed_center, current_text, box_color)
        else:
            smoothed_center = None

        # ---------------- HUD (hints + status) -------------------------------
        cv2.putText(frame, f"Palabra: {display_text}   (n/p cambia, puno cambia)",
                    (10, 30), FONT, 0.7, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        status = (f"[f]cara:{'ON' if pixelate_face else 'off'}  "
                  f"[l]logo:{'ON' if pixelate_logo else 'off'}  "
                  f"[s]scan:{'ON' if show_scanner else 'off'}  "
                  f"[e]enrolar  [r]rec:{'ON' if recording else 'off'}  [q]salir")
        cv2.putText(frame, status, (10, h - 15), FONT, 0.6, (255, 255, 0), 2,
                    lineType=cv2.LINE_AA)

        # Enrollment progress banner.
        if enrolling:
            msg = f"CAPTURANDO ROSTRO: {len(enroll_crops)}/{ENROLL_SAMPLES}"
            cv2.putText(frame, msg, (10, 90), FONT, 0.8, (0, 165, 255), 2,
                        lineType=cv2.LINE_AA)

        # In-window name entry (blinking cursor).
        if naming:
            cursor = "_" if int(now * 2) % 2 == 0 else " "
            cv2.putText(frame, f"NOMBRE: {name_buffer}{cursor}", (10, 90),
                        FONT, 0.9, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, "[Enter] guardar   [Esc] cancelar", (10, 122),
                        FONT, 0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Blinking red dot while recording.
        if recording:
            if int(now * 2) % 2 == 0:
                cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
            if writer is not None:
                writer.write(frame)

        cv2.imshow("Hand Tracker - Fun Edition", frame)

        # ---------------- KEYBOARD ------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        # While typing a name, ALL keys go to the text buffer (this branch fully
        # consumes the key, so shortcuts like q/n/f don't fire mid-typing).
        if naming:
            if key in (13, 10):                    # Enter -> save the face
                nm = name_buffer.strip()
                if nm:
                    if nm in name_to_id:
                        fid = name_to_id[nm]        # add to an existing person
                    else:
                        fid = next_id
                        next_id += 1
                        labels[fid] = nm
                        name_to_id[nm] = fid
                    recognizer.add(enroll_crops, fid)
                    recognizer.save(FACES_DB_PATH)
                    with open(LABELS_PATH, "w", encoding="utf-8") as f:
                        json.dump({str(k): v for k, v in labels.items()}, f)
                    print(f"Enrolado: '{nm}' -> reconocimiento activo.")
                else:
                    print("Enrolamiento cancelado (nombre vacio).")
                naming = False
                name_buffer = ""
                enroll_crops = []
            elif key == 27:                        # Esc -> cancel
                naming = False
                name_buffer = ""
                enroll_crops = []
                print("Enrolamiento cancelado.")
            elif key in (8, 127):                  # Backspace
                name_buffer = name_buffer[:-1]
            elif 32 <= key <= 126:                 # printable character
                name_buffer += chr(key)
            # (any other key ignored while naming)
            continue

        if key == ord('q'):
            break
        elif key in (ord('n'), ord('.')):
            word_index = (word_index + 1) % len(WORDS)
            prev_char_count = 0
        elif key in (ord('p'), ord(',')):
            word_index = (word_index - 1) % len(WORDS)
            prev_char_count = 0
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(WORDS):
                word_index = idx
                prev_char_count = 0
        elif key == ord('f'):
            pixelate_face = not pixelate_face
        elif key == ord('l'):
            pixelate_logo = not pixelate_logo
        elif key == ord('s'):
            show_scanner = not show_scanner
        elif key == ord('e'):
            # Start enrolling the currently-visible (largest) face.
            if face_detector is None:
                print("No hay detector facial disponible para enrolar.")
            elif not enrolling:
                enrolling = True
                enroll_crops = []
                print("Enrolando: mantene tu cara visible y centrada...")
        elif key == ord('r'):
            recording = not recording
            if recording:
                fname = time.strftime("grabacion_%Y%m%d_%H%M%S.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(fname, fourcc, 20.0, (w, h))
                print(f"Grabando -> {fname}")
            elif writer is not None:
                writer.release()
                writer = None
                print("Grabacion detenida y guardada.")

    # --- Cleanup ------------------------------------------------------------
    if writer is not None:
        writer.release()
    landmarker.close()
    if face_detector is not None:
        face_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
