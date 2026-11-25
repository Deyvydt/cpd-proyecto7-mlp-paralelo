"""
Validacion de correctitud: la version paralela final (BETA 3) debe seleccionar
exactamente el mismo modelo que la version secuencial de referencia (BETA 1).

Se comprueba para varios (p, n), incluyendo p que NO son potencia de 2.

Uso: python3 validacion.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from beta1_secuencial import entrenamiento_secuencial
from beta3_paralelo_final import ejecutar
from common import cargar_datos

CASOS = [(2, 3000), (3, 3000), (4, 5000), (6, 5000), (8, 5000)]


def main():
    print(f"{'p':>3} {'n':>7} {'s* sec':>7} {'s* par':>7} {'acc sec':>9} {'acc par':>9}  estado")
    todos_ok = True
    for p, n in CASOS:
        X_tr, X_te, y_tr, y_te = cargar_datos(n)
        _, A_sec, s_sec, _ = entrenamiento_secuencial(p, X_tr, y_tr, X_te, y_te)
        r = ejecutar(p, n)
        ok = (s_sec == r["semilla_ganadora"]) and abs(A_sec - r["accuracy"]) < 1e-12
        todos_ok &= ok
        print(f"{p:>3} {n:>7} {s_sec:>7} {r['semilla_ganadora']:>7} "
              f"{A_sec:>9.4f} {r['accuracy']:>9.4f}  {'OK' if ok else 'FALLA'}")
    print("\nResultado:", "todas las pruebas pasaron" if todos_ok else "hay diferencias")


if __name__ == "__main__":
    main()
