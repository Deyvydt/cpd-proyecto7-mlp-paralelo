"""
Proyecto 7 - Redes Neuronales (Computacion Paralela y Distribuida, CS4052, 2026-II)
Utilidades compartidas: generacion de datos, hiperparametros y conteo de FLOPs.
"""
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ---- Hiperparametros fijos del estudio (alpha, eta, h, T_max) ----
D_FEATURES = 20     # d: numero de features
N_CLASSES = 10      # c: numero de clases
H_HIDDEN = 64       # h: neuronas de la capa oculta
T_MAX = 50          # T_max: numero de epocas (E en el enunciado)
ALPHA = 1e-4        # alpha: regularizacion L2
ETA = 1e-3          # eta: learning rate
TEST_SIZE = 0.25


def cargar_datos(n_samples, seed_data=42):
    """Genera el dataset y lo divide UNA sola vez en train/test (linea 1 del PRAM)."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=D_FEATURES,
        n_informative=15,
        n_redundant=2,
        n_classes=N_CLASSES,
        random_state=seed_data,
    )
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed_data, stratify=y)


def construir_modelo(semilla):
    """Construye un MLP con una capa oculta de h neuronas y la semilla dada."""
    return MLPClassifier(
        hidden_layer_sizes=(H_HIDDEN,),
        alpha=ALPHA,
        learning_rate_init=ETA,
        max_iter=T_MAX,
        n_iter_no_change=T_MAX,   # sin early stopping: T_max epocas efectivas
        tol=0.0,
        random_state=semilla,
    )


def entrenar_y_evaluar(semilla, X_tr, y_tr, X_te, y_te):
    """Entrena un MLP con la semilla dada y devuelve (accuracy, epocas, t_entrenamiento)."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    t0 = time.perf_counter()
    modelo = construir_modelo(semilla)
    modelo.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0
    acc = modelo.score(X_te, y_te)
    return modelo, acc, modelo.n_iter_, t_fit


def flops_entrenamiento(n_train, epocas):
    """
    FLOPs de la region paralelizable (entrenamiento de UN modelo).

    Por muestra y epoca, con W1 en R^{h x d} y W2 en R^{c x h}:
      forward : 2*d*h + 2*h*c
      backward: 4*d*h + 4*h*c   (gradientes + actualizacion de pesos)
    Total por muestra y epoca: 6*h*(d + c)
    """
    return epocas * n_train * 6.0 * H_HIDDEN * (D_FEATURES + N_CLASSES)
