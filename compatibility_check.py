"""compatibility_check.py

Comprobaciones para detectar incompatibilidades entre NumPy 2.x y extensiones compiladas.
Provee ensure_numpy_compatibility(exit_on_fail=True) que imprimirá recomendaciones y
opciones (crear entorno con numpy<2 o recompilar extensiones) si detecta problemas.
"""
import sys
import subprocess
import importlib
import os
from typing import List

def _print_box(msg: str):
    sep = "=" * max(60, len(msg))
    print(f"\n{sep}\n{msg}\n{sep}\n")

def _find_pyd_with_array_api(search_paths: List[str]) -> List[str]:
    """Busca archivos .pyd/.dll en las rutas indicadas que contengan la cadena '_ARRAY_API'."""
    matches = []
    try:
        for base in search_paths:
            if not base or not os.path.isdir(base):
                continue
            for root, _, files in os.walk(base):
                for f in files:
                    if f.lower().endswith(('.pyd', '.dll')):
                        full = os.path.join(root, f)
                        try:
                            with open(full, 'rb') as fh:
                                data = fh.read()
                                if b'_ARRAY_API' in data:
                                    matches.append(full)
                        except Exception:
                            # no leer archivos binarios bloqueados
                            continue
    except Exception:
        pass
    return matches

def ensure_numpy_compatibility(exit_on_fail: bool = False):
    """Verifica la instalación de numpy y detecta problemas conocidos con NumPy 2.x.

    Si detecta un numpy>=2 y posibles extensiones binarias que requieren recompilación,
    imprime recomendaciones y (opcionalmente) termina el proceso con código 1.
    """
    try:
        import numpy as np
    except Exception as e:
        _print_box("ERROR: NumPy no puede ser importado. Revise su instalación.")
        print(f"Detalle: {e}")
        if exit_on_fail:
            sys.exit(1)
        return

    v = np.__version__
    _print_box(f"NumPy detectado: {v} (ruta: {getattr(np, '__file__', 'desconocida')})")

    major = int(v.split('.')[0]) if v and v[0].isdigit() else 0
    if major >= 2:
        # buscar binarios en sys.path y site-packages
        import site
        search_paths = list(set(sys.path + site.getsitepackages() if hasattr(site, 'getsitepackages') else sys.path))
        matches = _find_pyd_with_array_api(search_paths)
        if matches:
            _print_box("POTENCIAL INCOMPATIBILIDAD: Se detectaron binarios que contienen '_ARRAY_API'.")
            print("Estos binarios pueden haber sido compilados contra NumPy 1.x y no ser compatibles con NumPy 2.x:")
            for m in matches[:20]:
                print(' -', m)
            if len(matches) > 20:
                print(f"  ...(+{len(matches)-20} más)")
            print("\nRecomendaciones:")
            print(" 1) La solución rápida: usar un entorno con 'numpy<2' (p.ej. numpy==1.26.4).")
            print(" 2) A largo plazo: reinstalar/recompilar los paquetes problemáticos desde la fuente con pybind11>=2.12.")
            print("    Ejemplo: pip install --force-reinstall --no-binary :all: matplotlib")
            if exit_on_fail:
                sys.exit(1)
        else:
            _print_box("NumPy >= 2 detectado, pero no se encontraron binarios con '_ARRAY_API' en los paths comunes.")
            print("Si aún obtiene errores, considere usar numpy<2 temporalmente o recompilar dependencias.")
    else:
        print("NumPy < 2 detectado — compatibilidad esperada con extensiones compiladas más antiguas.")

    return

if __name__ == '__main__':
    ensure_numpy_compatibility(exit_on_fail=False)

