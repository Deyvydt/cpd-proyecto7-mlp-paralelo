"""
Genera las graficas del informe a partir de resultados/resultados.csv.

Figuras producidas en figuras/:
  fig1_tiempos.png        tiempo de ejecucion vs p, una curva por n
  fig2_desglose.png       computo vs comunicacion vs p
  fig3_teoria_vs_exp.png  curva teorica superpuesta a la experimental
  fig4_speedup.png        speedup vs p (con la recta ideal S = p)
  fig5_eficiencia.png     eficiencia vs p
  fig6_flops.png          FLOPS vs p
  fig7_weak_scaling.png   escalabilidad debil: n y p crecen juntos

El modelo teorico ajustado se describe en el informe:
      T_p(n,p) = c1 * ceil(p / P) * T_max*n*d*h  +  c2 * log2(p)  +  c3
con P = numero de nucleos fisicos. Los coeficientes c1, c2, c3 se obtienen por
minimos cuadrados sobre los datos experimentales (puntos 4 del enunciado).

Uso: python3 graficas.py
"""
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import T_MAX, D_FEATURES, H_HIDDEN

CSV = "../resultados/resultados.csv"
DIR = "../figuras"
TICKS = [1, 2, 4, 6, 8, 12, 16, 32]

# Numero de nucleos FISICOS del procesador donde se midio. Es un dato del hardware,
# no un parametro que se ajuste: el i5-11400H tiene 6 nucleos fisicos y 12 logicos.
# Si se mide en otra maquina, cambiar este valor (o ponerlo en None para estimarlo).
P_FISICOS = 6
COLORES = {5000: "#1f77b4", 10000: "#2ca02c", 20000: "#ff7f0e", 40000: "#d62728"}


def cargar():
    with open(CSV) as f:
        filas = [{k: (float(v) if k not in ("semilla_ganadora",) else v) for k, v in r.items()}
                 for r in csv.DictReader(f)]
    return filas


def _ajuste(filas, P):
    """Minimos cuadrados no negativos para un P dado. Devuelve (coef, error_rel_medio)."""
    A, b = [], []
    for r in filas:
        p, n = r["p"], r["n"]
        W = T_MAX * (n * 0.75) * D_FEATURES * H_HIDDEN
        A.append([max(1.0, p / P) * W, math.log2(p) if p > 1 else 0.0, 1.0])
        b.append(r["T_p"])
    A, b = np.array(A), np.array(b)
    try:
        from scipy.optimize import nnls
        coef, _ = nnls(A, b)
    except Exception:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        coef = np.clip(coef, 0, None)
    pred = A @ coef
    err = float(np.mean(np.abs(pred - b) / b)) * 100
    return coef, err


def ajustar_modelo(filas, n_cpu):
    """
    Ajusta T_p = c1*ceil(p/P)*W(n) + c2*log2(p) + c3.

    P es el numero de nucleos FISICOS de la maquina (P_FISICOS), que es un dato del
    hardware y no un parametro libre. Para contrastar, se reporta tambien que valor de P
    minimizaria el error si se dejara libre.
    """
    errores = {Pv: _ajuste(filas, Pv)[1] for Pv in range(1, max(2, n_cpu) + 1)}
    P_libre = min(errores, key=errores.get)
    P = P_FISICOS if P_FISICOS else P_libre
    coef, err = _ajuste(filas, P)
    print(f"  P = {P} (nucleos fisicos), error {err:.1f}%"
          f"   [P libre que minimizaria el error: {P_libre}, {errores[P_libre]:.1f}%]")
    return coef, P


def t_teorico(p, n, coef, n_cpu):
    n_train = n * 0.75
    W = T_MAX * n_train * D_FEATURES * H_HIDDEN
    return coef[0] * max(1.0, p / n_cpu) * W + coef[1] * (math.log2(p) if p > 1 else 0) + coef[2]


def serie(filas, n, campo):
    s = sorted([r for r in filas if r["n"] == n], key=lambda r: r["p"])
    return [r["p"] for r in s], [r[campo] for r in s]


def guardar(fig, nombre):
    os.makedirs(DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, nombre), dpi=150)
    plt.close(fig)
    print("  ", nombre)


def main():
    filas = cargar()
    ns = sorted({r["n"] for r in filas})
    n_cpu = int(filas[0]["n_cpu"])
    coef, P_eff = ajustar_modelo(filas, n_cpu)
    print(f"Procesadores logicos: {n_cpu}   coeficientes ajustados: c1={coef[0]:.3e} c2={coef[1]:.3e} c3={coef[2]:.3e}")

    # --- Fig 1: tiempos de ejecucion ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        x, y = serie(filas, n, "T_p")
        ax.plot(x, y, "o-", color=COLORES[n], label=f"n = {int(n)}")
    ax.set_xscale("log", base=2); ax.set_xticks(TICKS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.axvline(P_eff, color="gray", ls=":", lw=1)
    ax.set_xlabel("Numero de procesos p"); ax.set_ylabel("Tiempo de ejecucion $T_p$ [s]")
    ax.set_title(f"Tiempo de ejecucion paralelo (linea punteada: P = {P_eff})")
    ax.grid(alpha=.3); ax.legend()
    guardar(fig, "fig1_tiempos.png")

    # --- Fig 2: computo vs comunicacion ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for n in ns:
        x, yc = serie(filas, n, "t_comp")
        _, ym = serie(filas, n, "t_comm")
        axes[0].plot(x, yc, "o-", color=COLORES[n], label=f"n = {int(n)}")
        axes[1].plot(x, ym, "s--", color=COLORES[n], label=f"n = {int(n)}")
    for n in ns:
        x, ys = serie(filas, n, "t_startup")
        axes[1].plot(x, ys, "^:", color=COLORES[n], alpha=.5)
    for ax, t in zip(axes, ["Tiempo de computo",
                            "Comunicacion (cuadrados) y arranque del pool (triangulos)"]):
        ax.set_xscale("log", base=2); ax.set_xticks(TICKS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Numero de procesos p"); ax.set_ylabel("Tiempo [s]")
        ax.set_title(t); ax.grid(alpha=.3); ax.legend()
    guardar(fig, "fig2_desglose.png")

    # --- Fig 3: teoria vs experimento ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        x, y = serie(filas, n, "T_p")
        ax.plot(x, y, "o", color=COLORES[n], label=f"n = {int(n)} (exp.)")
        xt = np.logspace(0, 5, 400, base=2)
        ax.plot(xt, [t_teorico(v, n, coef, P_eff) for v in xt], "-", color=COLORES[n],
                alpha=.6, label=f"n = {int(n)} (teorico)")
    ax.set_xscale("log", base=2); ax.set_xticks(TICKS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Numero de procesos p"); ax.set_ylabel("$T_p$ [s]")
    ax.set_title("Curva teorica superpuesta a la experimental")
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    guardar(fig, "fig3_teoria_vs_exp.png")

    # --- Fig 4: speedup ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ps = sorted({r["p"] for r in filas})
    ax.plot(ps, ps, "k--", label="ideal $S = p$")
    xt = np.logspace(0, 5, 400, base=2)
    ax.plot(xt, [v / max(1.0, v / P_eff) for v in xt], color="gray", lw=1.2,
            label=f"teorico $S = p/\\max(1,p/P)$, P = {P_eff}")
    ax.axhline(P_eff, color="gray", ls=":", label=f"techo por nucleos efectivos (P = {P_eff})")
    ax.axvline(P_eff, color="gray", ls=":", lw=1)
    for n in ns:
        x, y = serie(filas, n, "speedup")
        ax.plot(x, y, "o-", color=COLORES[n], label=f"n = {int(n)}")
    ax.set_xscale("log", base=2); ax.set_xticks(TICKS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Numero de procesos p"); ax.set_ylabel("Speedup $S_p$")
    ax.set_title("Speedup"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    guardar(fig, "fig4_speedup.png")

    # --- Fig 5: eficiencia ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(1.0, color="k", ls="--", label="ideal $E = 1$")
    xt = np.logspace(0, 5, 400, base=2)
    ax.plot(xt, [1.0 / max(1.0, v / P_eff) for v in xt], color="gray", lw=1.2,
            label=f"teorico $E = 1/\\max(1,p/P)$, P = {P_eff}")
    for n in ns:
        x, y = serie(filas, n, "eficiencia")
        ax.plot(x, y, "o-", color=COLORES[n], label=f"n = {int(n)}")
    ax.set_xscale("log", base=2); ax.set_xticks(TICKS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Numero de procesos p"); ax.set_ylabel("Eficiencia $E_p$")
    ax.set_title("Eficiencia"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    guardar(fig, "fig5_eficiencia.png")

    # --- Fig 6: FLOPS ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        x, y = serie(filas, n, "flops_por_seg")
        ax.plot(x, [v / 1e9 for v in y], "o-", color=COLORES[n], label=f"n = {int(n)}")
    ax.set_xscale("log", base=2); ax.set_xticks(TICKS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Numero de procesos p"); ax.set_ylabel("GFLOP/s")
    ax.set_title("Rendimiento: FLOPs por segundo vs p")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    guardar(fig, "fig6_flops.png")

    # --- Fig 7: escalabilidad debil (n y p crecen juntos) ---
    pares = [(1, 5000), (2, 10000), (4, 20000), (8, 40000)]
    xs, tp, ef = [], [], []
    for p, n in pares:
        for r in filas:
            if r["p"] == p and r["n"] == n:
                xs.append(p); tp.append(r["T_p"]); ef.append(r["eficiencia"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(xs, tp, "o-", color="#6a3d9a")
    axes[0].set_ylabel("$T_p$ [s]"); axes[0].set_title("Weak scaling: tiempo (ideal = constante)")
    axes[1].plot(xs, ef, "o-", color="#6a3d9a"); axes[1].axhline(1, color="k", ls="--")
    axes[1].set_ylabel("$E_p$"); axes[1].set_title("Weak scaling: eficiencia")
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("p (con n creciendo proporcionalmente)"); ax.grid(alpha=.3)
    guardar(fig, "fig7_weak_scaling.png")

    with open("../resultados/ajuste_teorico.json", "w") as f:
        json.dump({"c1": coef[0], "c2": coef[1], "c3": coef[2],
                   "n_cpu": n_cpu, "P_eff": P_eff}, f, indent=2)


if __name__ == "__main__":
    main()
