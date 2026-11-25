# Proyecto 7 — Redes Neuronales: entrenamiento paralelo de MLPs

**Curso:** Computación Paralela y Distribuida (CS4052) — UTEC, 2026-II
**Profesor:** José Fiestas
**Integrantes:** David Marcelo Quispe Velásquez · Johannes Albert Loayza Huamán

---

## Qué hace

Paraleliza el **Algoritmo 1** del enunciado: entrenar `p` perceptrones multicapa (MLP)
con distintas semillas y quedarse con el de mayor *accuracy*.

Como el entrenamiento de cada modelo es **independiente** de los demás
(*embarrassingly parallel*), se asigna un modelo a cada proceso y al final se aplica
una **reducción en árbol** (máximo con índice) de costo `O(log p)`.

| Fase | Modelo PRAM | Costo |
|---|---|---|
| 1. Entrenamiento de los p modelos | CREW | `O(T_max · n · d · h)` |
| 2. Reducción en árbol del máximo | EREW | `O(log p)` |
| 3. Recuperar el modelo ganador | — | `O(1)` |

**Tiempo paralelo:** `T_p = O(T_max · n · d · h + log p)`

---

## Estructura

```
src/
  common.py                  datos, hiperparámetros y conteo de FLOPs
  beta1_secuencial.py        BETA 1 — Algoritmo 1 del enunciado (línea base T_s)
  beta2_paralelo_simple.py   BETA 2 — fase 1 paralela, selección secuencial O(p)
  beta3_paralelo_final.py    BETA 3 — versión final: reducción en árbol O(log p)
  benchmark.py               barrido p = 1..32, n = 5000..40000
  graficas.py                genera las figuras del informe
resultados/
  resultados.csv             mediciones crudas
  ajuste_teorico.json        coeficientes del modelo teórico ajustado
figuras/                     fig1..fig7 (.png)
informe/                     informe final en LaTeX
```

## Cómo ejecutar

```bash
pip install scikit-learn numpy matplotlib joblib

cd src
python3 beta1_secuencial.py --p 8 --n 10000      # versión secuencial
python3 beta3_paralelo_final.py --p 8 --n 10000  # versión paralela final
python3 benchmark.py --reps 2                    # barrido completo
python3 graficas.py                              # figuras
```

## Validación

La BETA 3 fue validada contra la BETA 1: para las mismas `(p, n)` ambas devuelven
**la misma semilla ganadora y la misma accuracy**, lo que confirma que la reducción
en árbol selecciona el mismo modelo que el `max` secuencial del algoritmo original.

## Parámetros del estudio

| Símbolo | Significado | Valor |
|---|---|---|
| `p` | procesos (= número de semillas) | 1, 2, 4, 8, 16, 32 |
| `n` | muestras (`n_samples`) | 5000, 10000, 20000, 40000 |
| `d` | features | 20 |
| `h` | neuronas de la capa oculta | 64 |
| `c` | clases | 10 |
| `T_max` | épocas (`E` en el enunciado) | 50 |

> Se usa `T_max` para las épocas y se reserva `E` para la eficiencia, como en el curso.
