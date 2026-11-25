"""
BETA 2 - Primera version paralela (fase 1 del PRAM, sin reduccion en arbol).

Los p modelos se entrenan en paralelo con multiprocessing (un proceso por semilla),
pero el maximo de exactitud se obtiene con un max() secuencial, de costo O(p).
Limitacion detectada en esta version: la seleccion del ganador no corresponde al
PRAM propuesto (deberia ser una reduccion en arbol de costo O(log p)); eso se
corrige en la BETA 3.

Uso:  python3 beta2_paralelo_simple.py --p 4 --n 5000
"""
import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")       # 1 hilo BLAS por proceso
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp
import numpy as np
from common import cargar_datos, entrenar_y_evaluar

_DATOS = {}


def _contexto():
    """
    Devuelve el contexto de multiprocessing disponible.

    'fork' (Linux/macOS) es mas barato porque el hijo hereda la memoria del padre.
    En Windows no existe, asi que se usa 'spawn': cada proceso arranca de cero y
    recibe los datos por serializacion, lo que aumenta el tiempo de comunicacion.
    """
    import multiprocessing as _mp
    for nombre in ("fork", "spawn"):
        try:
            return _mp.get_context(nombre)
        except ValueError:
            continue
    return _mp.get_context()



def preparar_datos(n, dir_tmp):
    """
    El proceso padre genera la particion UNA sola vez y la deja en un archivo .npz,
    que hace las veces de la memoria compartida del modelo PRAM: todos los procesos
    leen de ahi (lectura concurrente, CREW) y ninguno la modifica.

    No se envian los arrays por el canal del Pool porque en Windows ese envio bloquea
    al padre hasta que cada hijo termina de leer, lo que serializa el arranque y
    contamina la medicion con un costo que no pertenece al algoritmo.
    """
    os.makedirs(dir_tmp, exist_ok=True)
    ruta = os.path.join(dir_tmp, f"datos_{n}.npz")
    X_tr, X_te, y_tr, y_te = cargar_datos(n)
    np.savez(ruta, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te)
    return ruta, len(X_tr)


def _init(ruta):
    """Cada proceso lee la particion compartida desde disco (CREW)."""
    d = np.load(ruta)
    _DATOS["d"] = (d["X_tr"], d["y_tr"], d["X_te"], d["y_te"])


def _tarea(semilla):
    X_tr, y_tr, X_te, y_te = _DATOS["d"]
    _, acc, epocas, t_fit = entrenar_y_evaluar(semilla, X_tr, y_tr, X_te, y_te)
    return semilla, acc, epocas, t_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    ruta, _ = preparar_datos(args.n, "modelos_tmp")

    t0 = time.perf_counter()
    with _contexto().Pool(
        processes=args.p, initializer=_init, initargs=(ruta,)
    ) as pool:
        resultados = pool.map(_tarea, range(args.p))          # fase 1: for all ... pardo
    # seleccion secuencial O(p)  <-- se reemplaza por reduccion en arbol en la beta 3
    s_best, A_best, _, _ = max(resultados, key=lambda r: r[1])
    t_total = time.perf_counter() - t0

    print(f"[BETA2 paralelo simple] p={args.p} n={args.n}")
    print(f"  mejor semilla s* = {s_best}   accuracy = {A_best:.4f}   T_p = {t_total:.3f} s")


if __name__ == "__main__":
    main()
