"""
Barrido experimental del proyecto 7.

Para cada combinacion de p en {1,2,4,8,16,32} y n_samples en {5000,...,40000}:
  - mide T_p (version paralela beta 3), desglosado en computo y comunicacion
  - mide t_1(n): tiempo de entrenar UN modelo (promedio de varias semillas)
  - calcula T_s(p,n) = p * t_1(n), que es el costo del Algoritmo 1 secuencial
  - calcula speedup, eficiencia y FLOPS

Salida: resultados/resultados.csv
Uso:    python3 benchmark.py [--reps 3]
"""
import argparse
import csv
import os
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from common import cargar_datos, entrenar_y_evaluar
from beta3_paralelo_final import ejecutar

LISTA_P = [1, 2, 4, 8, 16, 32]
LISTA_N = [5000, 10000, 20000, 40000]
SEMILLAS_BASE = 4      # cuantas semillas se promedian para estimar t_1(n)


def medir_t1(n, reps):
    """t_1(n): tiempo medio de entrenar un unico MLP (base del costo secuencial)."""
    X_tr, X_te, y_tr, y_te = cargar_datos(n)
    tiempos = []
    for _ in range(reps):
        for s in range(SEMILLAS_BASE):
            _, _, _, t_fit = entrenar_y_evaluar(s, X_tr, y_tr, X_te, y_te)
            tiempos.append(t_fit)
    return statistics.median(tiempos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=str, default="../resultados/resultados.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    filas = []
    n_cpu = os.cpu_count()
    print(f"Nucleos disponibles: {n_cpu}")

    for n in LISTA_N:
        t1 = medir_t1(n, reps=1)
        print(f"\nn={n}: t_1 = {t1:.3f} s por modelo")
        for p in LISTA_P:
            corridas = []
            for _ in range(args.reps):
                corridas.append(ejecutar(p, n))
            # se toma la mediana de T_p entre repeticiones
            corridas.sort(key=lambda r: r["T_p"])
            r = corridas[len(corridas) // 2]

            T_s = p * t1
            S = T_s / r["T_p"]
            E = S / p
            fila = {
                "n": n, "p": p,
                "t_1": round(t1, 4),
                "T_s": round(T_s, 4),
                "T_p": round(r["T_p"], 4),
                "t_comp": round(r["t_comp"], 4),
                "t_comm": round(r["t_comm"], 4),
                "t_startup": round(r["t_startup"], 4),
                "t_fase1": round(r["t_fase1"], 4),
                "t_fase2_us": round(r["t_fase2"] * 1e6, 3),
                "speedup": round(S, 4),
                "eficiencia": round(E, 4),
                "accuracy": round(r["accuracy"], 4),
                "semilla_ganadora": r["semilla_ganadora"],
                "rondas_reduccion": r["rondas_reduccion"],
                "flops": r["flops"],
                "flops_por_seg": r["flops_por_seg"],
                "n_cpu": n_cpu,
            }
            filas.append(fila)
            print(f"  p={p:2d}  T_s={T_s:7.2f}  T_p={r['T_p']:7.2f}  "
                  f"S={S:5.2f}  E={E:5.2f}  acc={r['accuracy']:.4f}")

    import json, platform
    with open(os.path.join(os.path.dirname(args.out), "entorno.json"), "w") as f:
        json.dump({"cpu": platform.processor(), "sistema": platform.platform(),
                   "python": platform.python_version(), "n_cpu_logicos": n_cpu}, f, indent=2)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\nResultados guardados en {args.out}")


if __name__ == "__main__":
    main()
