"""
cripto.py - Cifrado en reposo de la base biometrica (Fernet / AES-128 autenticado).

La clave NO vive en el codigo ni junto a los datos: se lee de la variable de
entorno ASISTRA_KEY. Sin la clave, los archivos .enc son ilegibles, asi que un
robo de los archivos no sirve de nada.

Generá una clave con:  python generar_clave.py

Nota de alcance: esto protege los datos EN REPOSO (en disco). Un atacante con la
clave o con acceso al proceso en ejecucion sigue pudiendo leerlos; para subir el
nivel, la clave puede derivarse de una passphrase que el operador escribe al
arrancar (scrypt/PBKDF2) o guardarse en un secret manager. Ver README.
"""

import os
import io
import json

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

ENV_VAR = "ASISTRA_KEY"


class ClaveFaltante(Exception):
    """No hay clave en el entorno."""


class ClaveInvalida(Exception):
    """La clave no es valida o no coincide con los datos."""


def generar_clave() -> str:
    """Devuelve una clave Fernet nueva (string urlsafe base64)."""
    return Fernet.generate_key().decode()


def hay_clave() -> bool:
    return bool(os.environ.get(ENV_VAR, "").strip())


def _fernet() -> Fernet:
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        raise ClaveFaltante(
            f"Falta la variable de entorno {ENV_VAR}. "
            "Genera una con: python generar_clave.py")
    try:
        return Fernet(raw.encode())
    except Exception:
        raise ClaveInvalida(f"La clave en {ENV_VAR} no tiene formato Fernet valido.")


# --- JSON cifrado ----------------------------------------------------------

def guardar_json(path, obj):
    token = _fernet().encrypt(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    with open(path, "wb") as f:
        f.write(token)


def cargar_json(path):
    with open(path, "rb") as f:
        token = f.read()
    try:
        data = _fernet().decrypt(token)
    except InvalidToken:
        raise ClaveInvalida("No se pudo descifrar el JSON (la clave no coincide).")
    return json.loads(data.decode("utf-8"))


# --- NPZ (arrays numpy) cifrado -------------------------------------------

def guardar_npz(path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    token = _fernet().encrypt(buf.getvalue())
    with open(path, "wb") as f:
        f.write(token)


def cargar_npz(path):
    with open(path, "rb") as f:
        token = f.read()
    try:
        data = _fernet().decrypt(token)
    except InvalidToken:
        raise ClaveInvalida("No se pudo descifrar la base facial (clave incorrecta).")
    # allow_pickle=False (default): solo arrays numericos/texto, nada de pickle.
    return np.load(io.BytesIO(data))
