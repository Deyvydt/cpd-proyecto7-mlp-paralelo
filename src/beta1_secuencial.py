"""
BETA 1 - Version secuencial de referencia (Algoritmo 1 del enunciado).

Entrena p modelos MLP, uno por semilla y uno detras de otro, y conserva el de
mayor exactitud. Sirve como linea base T_s y como validacion de correctitud:
el modelo ganador de esta version debe coincidir con el de las versiones paralelas.

Uso:  python3 beta1_secuencial.py --p 4 --n 5000
"""
import argparse
import time
import joblib

from common import cargar_datos, entrenar_y_evaluar, flops_entrenamiento


def entrenamiento_secuencial(p, X_tr, y_tr, X_te, y_te):
    A_best = float("-inf")       # linea 2: A_best <- -infinito
    s_best = -1                  # linea 3
    M_best = None                # linea 4
    epocas_tot = 0

    for s in range(p):                                        # linea 5: foreach s in S
        modelo, acc, epocas, _ = entrenar_y_evaluar(s, X_tr, y_tr, X_te, y_te)  # lineas 6-9
        epocas_tot += epocas
        if acc > A_best:                                      # linea 10
            A_best, s_best, M_best = acc, s, modelo           # lineas 11-13
    return M_best, A_best, s_best, epocas_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    X_tr, X_te, y_tr, y_te = cargar_datos(args.n)

    t0 = time.perf_counter()
    M, A, s, epocas_tot = entrenamiento_secuencial(args.p, X_tr, y_tr, X_te, y_te)
    t_total = time.perf_counter() - t0

    joblib.dump(M, "mlp_model.joblib")                        # linea 16

    flops = flops_entrenamiento(len(X_tr), epocas_tot)
    print(f"[BETA1 secuencial] p={args.p} n={args.n}")
    print(f"  mejor semilla s* = {s}   accuracy = {A:.4f}")
    print(f"  T_s = {t_total:.3f} s   FLOPs = {flops:.3e}   FLOPS = {flops/t_total:.3e}")


if __name__ == "__main__":
    main()
