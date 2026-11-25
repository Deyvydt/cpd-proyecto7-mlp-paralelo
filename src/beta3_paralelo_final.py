"""
BETA 3 - Version final: implementacion fiel del PRAM propuesto.

Fase 1 (CREW) : los p procesos leen los mismos datos y entrenan un MLP cada uno.
Fase 2 (EREW) : reduccion en arbol del maximo de exactitud, con indice, en
                ceil(log2 p) rondas (sirve tambien si p no es potencia de 2).
Fase 3        : el ganador queda en la casilla 0 y se guarda en mlp_model.joblib.

Ademas mide por separado tiempo de computo y de comunicacion, y cuenta los FLOPs
de la region paralelizable.

Uso:  python3 beta3_paralelo_final.py --p 4 --n 5000 [--json salida.json]
"""
import argparse
import json
import math
import os
import shutil
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp
import numpy as np
import joblib

from common import cargar_datos, entrenar_y_evaluar, flops_entrenamiento

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

DIR_TMP = "modelos_tmp"


def preparar_datos(n, dir_tmp):
    """
    El proceso padre genera la particion UNA sola vez y la deja en un archivo .npz,
    que hace las veces de la memoria compartida del modelo PRAM: todos los procesos
    leen de ahi (lectura concurrente, CREW) y ninguno la modifica.

    No se envian los arrays por el canal del Pool porque en Windows ese envio bloquea
    al padre hasta que cada hijo termina de leerlos, lo que serializa el arranque de
    los procesos y contamina la medicion con un costo ajeno al algoritmo.
    """
    os.makedirs(dir_tmp, exist_ok=True)
    ruta = os.path.join(dir_tmp, f"datos_{n}.npz")
    X_tr, X_te, y_tr, y_te = cargar_datos(n)
    np.savez(ruta, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te)
    return ruta, len(X_tr)


def _init(ruta):
    """Cada proceso lee la particion compartida desde disco (lectura concurrente, CREW)."""
    d = np.load(ruta)
    _DATOS["d"] = (d["X_tr"], d["y_tr"], d["X_te"], d["y_te"])


def _tarea(s):
    """Cuerpo del 'for all s, 0 <= s <= p-1 pardo': el procesador P_s entrena M_s."""
    X_tr, y_tr, X_te, y_te = _DATOS["d"]
    modelo, acc, epocas, t_fit = entrenar_y_evaluar(s, X_tr, y_tr, X_te, y_te)  # semilla s
    joblib.dump(modelo, os.path.join(DIR_TMP, f"modelo_{s}.joblib"))            # escritura exclusiva
    return s, acc, epocas, t_fit


def reduccion_en_arbol(A, idx):
    """
    Fase 2 del PRAM: maximo con indice en ceil(log2 p) rondas.
    En la ronda k participan las posiciones s con s mod 2^k == 0 y su rival es j = s + 2^(k-1).
    El bucle sobre k es secuencial (do); el bucle sobre s es paralelo (pardo).
    """
    p = len(A)
    rondas = max(1, math.ceil(math.log2(p))) if p > 1 else 0
    comparaciones = 0
    for k in range(1, rondas + 1):          # secuencial: la ronda k usa los ganadores de k-1
        paso = 2 ** k
        mitad = 2 ** (k - 1)
        for s in range(0, p, paso):         # paralelo en el modelo PRAM
            j = s + mitad
            if j < p:                       # permite p no potencia de 2
                comparaciones += 1
                if A[j] > A[s]:
                    A[s], idx[s] = A[j], idx[j]
    return A[0], idx[0], rondas, comparaciones


def ejecutar(p, n):
    ruta, n_train = preparar_datos(n, DIR_TMP)      # linea 1 (division unica)

    t_ini = time.perf_counter()

    t0 = time.perf_counter()
    with _contexto().Pool(
        processes=p, initializer=_init, initargs=(ruta,)
    ) as pool:
        resultados = pool.map(_tarea, range(p))     # FASE 1 (CREW)
    t_fase1 = time.perf_counter() - t0

    resultados.sort(key=lambda r: r[0])
    A = [r[1] for r in resultados]
    idx = [r[0] for r in resultados]                # idx[s] <- s
    epocas_tot = sum(r[2] for r in resultados)
    t_fit_max = max(r[3] for r in resultados)       # computo puro del proceso mas lento
    t_fit_prom = sum(r[3] for r in resultados) / p

    t0 = time.perf_counter()
    A_best, s_best, rondas, comparaciones = reduccion_en_arbol(A, idx)   # FASE 2 (EREW)
    t_fase2 = time.perf_counter() - t0

    shutil.copy(os.path.join(DIR_TMP, f"modelo_{s_best}.joblib"), "mlp_model.joblib")  # FASE 3
    t_total = time.perf_counter() - t_ini

    # t_comm = todo lo que no es computo puro: creacion de procesos, envio de datos,
    # recoleccion de resultados y reduccion.
    t_comm = t_total - t_fit_max

    flops = flops_entrenamiento(n_train, epocas_tot)
    return {
        "p": p, "n": n, "n_train": n_train,
        "T_p": t_total,
        "t_fase1": t_fase1, "t_fase2": t_fase2,
        "t_comp": t_fit_max, "t_comp_prom": t_fit_prom, "t_comm": t_comm,
        "accuracy": A_best, "semilla_ganadora": s_best,
        "epocas_totales": epocas_tot, "rondas_reduccion": rondas,
        "comparaciones": comparaciones,
        "flops": flops, "flops_por_seg": flops / t_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    r = ejecutar(args.p, args.n)
    print(f"[BETA3 final] p={r['p']} n={r['n']}")
    print(f"  mejor semilla s* = {r['semilla_ganadora']}   accuracy = {r['accuracy']:.4f}")
    print(f"  T_p = {r['T_p']:.3f} s  (computo {r['t_comp']:.3f} s, comunicacion {r['t_comm']:.3f} s)")
    print(f"  reduccion: {r['rondas_reduccion']} rondas, {r['comparaciones']} comparaciones")
    print(f"  FLOPs = {r['flops']:.3e}   FLOPS = {r['flops_por_seg']:.3e}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
