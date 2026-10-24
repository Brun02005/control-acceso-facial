"""
generar_clave.py - Genera la clave de cifrado de ASISTRA y muestra como instalarla.

Ejecutar UNA vez. Guardá la clave en un lugar seguro (gestor de contraseñas) y
cargala como variable de entorno ASISTRA_KEY. Sin ella no se puede leer la base
biometrica.
"""

import cripto

clave = cripto.generar_clave()

print("=" * 66)
print("  CLAVE DE CIFRADO GENERADA")
print("=" * 66)
print("  " + clave)
print("=" * 66)
print("  INSTALARLA (Windows) - pegá esto en una terminal:")
print()
print(f'    setx ASISTRA_KEY "{clave}"')
print()
print("  Despues ABRI UNA TERMINAL NUEVA (setx no afecta la terminal actual).")
print("  Verificá con:  echo %ASISTRA_KEY%   (cmd)  o  $env:ASISTRA_KEY  (PowerShell)")
print("=" * 66)
print("  IMPORTANTE")
print("  - Guardá esta clave en un gestor de contraseñas. Es la unica copia.")
print("  - Si la perdés, la base biometrica queda ilegible (hay que re-enrolar).")
print("  - NO la subas a git ni la dejes junto a los archivos .enc.")
print("=" * 66)
