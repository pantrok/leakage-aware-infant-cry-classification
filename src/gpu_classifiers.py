"""
Clasificadores con soporte opcional de GPU (cuML/RAPIDS) + grids de
hiperparámetros para optimización con nested cross-validation
================================================================================
scikit-learn (MLPClassifier, SVC, KNeighborsClassifier) es CPU-only por
diseño. Para aprovechar GPU NVIDIA, este módulo intenta usar cuML
(RAPIDS), que reimplementa SVC y KNeighborsClassifier con interfaz
compatible con sklearn pero ejecutando en GPU. cuML no tiene un
MLPClassifier equivalente; para ese caso se usa PyTorch con un MLP
simple entrenado en GPU si CUDA está disponible.

Si cuML/PyTorch+CUDA no están instalados o no hay GPU disponible, se
cae automáticamente a sklearn/CPU sin romper el flujo -- el pipeline es
idéntico, solo cambia dónde se ejecuta el cómputo.

IMPORTANTE: cuML no es binariamente idéntico a sklearn (distintos
solvers/inicializaciones internas); los resultados pueden diferir
ligeramente de corridas previas en CPU. Esto se documenta en consola al
inicio de la ejecución para que quede claro en qué modo corrió cada
resultado.

Instalación de cuML (requiere GPU NVIDIA + CUDA):
    pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12
(ver https://docs.rapids.ai/install/ para la versión exacta de tu CUDA)
"""

import warnings

import numpy as np

GPU_AVAILABLE = False
GPU_BACKEND = "none"

try:
    import cuml  # noqa: F401
    from cuml.svm import SVC as cuSVC
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    GPU_AVAILABLE = True
    GPU_BACKEND = "cuml"
except ImportError:
    pass

try:
    import torch
    if torch.cuda.is_available():
        TORCH_CUDA_AVAILABLE = True
    else:
        TORCH_CUDA_AVAILABLE = False
except ImportError:
    TORCH_CUDA_AVAILABLE = False

# Lista mutable de un solo elemento (truco para tener un "global" mutable
# simple): una vez que se detecta que CUDA está "disponible" según
# torch.cuda.is_available() pero en realidad falla al ejecutar cualquier
# operación real (ej. arquitectura de GPU muy nueva sin kernels
# compilados en el build estable de PyTorch, como sm_120/Blackwell),
# se marca aquí para que TODAS las instancias futuras de
# TorchMLPClassifier en el mismo proceso usen CPU directamente sin
# reintentar GPU en cada fold (eso sería lento y generaría el mismo
# aviso repetidamente).
_CUDA_RUNTIME_BROKEN = [False]

from sklearn.svm import SVC as skSVC
from sklearn.neighbors import KNeighborsClassifier as skKNN
from sklearn.neural_network import MLPClassifier as skMLP
from sklearn.base import BaseEstimator, ClassifierMixin


def report_backend():
    """Imprime claramente qué backend de cómputo se está usando, para
    que los resultados queden documentados (CPU vs GPU pueden no ser
    binariamente idénticos)."""
    if GPU_BACKEND == "cuml":
        print("[INFO] Backend de cómputo: cuML (GPU NVIDIA, vía RAPIDS) para SVM/kNN.")
    else:
        print("[INFO] Backend de cómputo: scikit-learn (CPU). cuML no disponible "
              "-- instala con 'pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12' "
              "si tienes GPU NVIDIA+CUDA y quieres acelerar SVM/kNN (nota: cuML no tiene build "
              "para Windows nativo, solo Linux/WSL2).")
    if TORCH_CUDA_AVAILABLE and not _CUDA_RUNTIME_BROKEN[0]:
        print("[INFO] PyTorch detecta GPU CUDA disponible para el MLP.")
    elif TORCH_CUDA_AVAILABLE and _CUDA_RUNTIME_BROKEN[0]:
        print("[INFO] PyTorch detecta GPU CUDA, pero ya falló en runtime en este proceso "
              "(arquitectura de GPU probablemente no soportada por este build de PyTorch) "
              "-- el MLP está corriendo en CPU como fallback automático.")
    else:
        print("[INFO] PyTorch sin GPU CUDA disponible (o no instalado) -- MLP corre en sklearn/CPU.")


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    MLP simple en PyTorch, entrenado en GPU si está disponible, con
    interfaz compatible con sklearn (fit/predict/predict_proba) para
    poder usarse dentro de un imblearn.Pipeline y GridSearchCV igual que
    cualquier estimador de sklearn.

    Arquitectura equivalente a MLPClassifier(hidden_layer_sizes=(64,32))
    de sklearn: dos capas ocultas, ReLU, salida softmax, optimizador Adam.
    """

    def __init__(self, hidden1=64, hidden2=32, lr=1e-3, max_iter=2000,
                 alpha=1e-4, random_state=42, device=None, class_weight=None):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lr = lr
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        self.device = device
        self.class_weight = class_weight

    def _get_device(self):
        if self.device is not None:
            return self.device
        return "cuda" if (TORCH_CUDA_AVAILABLE and not _CUDA_RUNTIME_BROKEN[0]) else "cpu"

    def _build_model(self, n_features, n_classes, device):
        import torch.nn as nn

        class _MLP(nn.Module):
            def __init__(self, n_in, h1, h2, n_out):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_in, h1), nn.ReLU(),
                    nn.Linear(h1, h2), nn.ReLU(),
                    nn.Linear(h2, n_out),
                )

            def forward(self, x):
                return self.net(x)

        return _MLP(n_features, self.hidden1, self.hidden2, n_classes).to(device)

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        device = self._get_device()

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        model = self._build_model(n_features, n_classes, device)

        # Prueba defensiva: si la GPU detectada no es realmente utilizable
        # por este build de PyTorch (ej. arquitectura muy nueva, como
        # Blackwell/sm_120, sin kernels compilados en el build estable),
        # un forward pass mínimo lo revela aquí, ANTES de comprometernos
        # a entrenar en ese device. Si falla, se recompila en CPU y se
        # avisa una sola vez por proceso (no en cada fold del grid search).
        if device == "cuda":
            try:
                with torch.no_grad():
                    probe = torch.zeros((1, n_features), dtype=torch.float32, device=device)
                    model(probe)
            except RuntimeError as e:
                if not _CUDA_RUNTIME_BROKEN[0]:
                    print(f"[AVISO] GPU detectada pero no utilizable por este build de PyTorch "
                          f"({e}). Cayendo a CPU para el MLP en el resto de la ejecución.")
                _CUDA_RUNTIME_BROKEN[0] = True
                device = "cpu"
                model = self._build_model(n_features, n_classes, device)

        self._model = model
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.alpha)

        class_weights_tensor = None
        if self.class_weight == "balanced":
            # Misma fórmula que sklearn: n_samples / (n_classes * count_por_clase)
            y_idx_for_weights = np.searchsorted(self.classes_, y)
            counts = np.bincount(y_idx_for_weights, minlength=n_classes)
            counts = np.maximum(counts, 1)  # evitar división por cero si alguna clase no aparece
            weights = len(y) / (n_classes * counts)
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_idx = np.searchsorted(self.classes_, y)
        y_t = torch.tensor(y_idx, dtype=torch.long, device=device)

        self._model.train()
        for _ in range(self.max_iter // 10):  # iteraciones más cortas que sklearn (full-batch Adam converge antes)
            optimizer.zero_grad()
            out = self._model(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()

        self._device = device
        return self

    def predict_proba(self, X):
        import torch
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
            out = self._model(X_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]


def make_classifier(name: str, class_weight=None, random_state: int = 42, **params):
    """
    Construye un clasificador por nombre, usando GPU (cuML/PyTorch) si
    está disponible, con fallback a sklearn/CPU. 'params' son
    hiperparámetros específicos del clasificador (ej. C, gamma, etc.).
    """
    if name == "MLP":
        if TORCH_CUDA_AVAILABLE:
            return TorchMLPClassifier(random_state=random_state, class_weight=class_weight, **params)
        return skMLP(max_iter=2000, random_state=random_state, **params)

    if name == "SVM-lineal":
        if GPU_AVAILABLE:
            return cuSVC(kernel="linear", class_weight=class_weight, **params)
        return skSVC(kernel="linear", class_weight=class_weight, **params)

    if name == "SVM-polinomial":
        if GPU_AVAILABLE:
            return cuSVC(kernel="poly", degree=3, class_weight=class_weight, **params)
        return skSVC(kernel="poly", degree=3, class_weight=class_weight, **params)

    if name == "SVM-RBF":
        if GPU_AVAILABLE:
            return cuSVC(kernel="rbf", class_weight=class_weight, **params)
        return skSVC(kernel="rbf", class_weight=class_weight, **params)

    if name == "kNN":
        if GPU_AVAILABLE:
            return cuKNN(**params)
        return skKNN(**params)

    raise ValueError(f"Clasificador desconocido: {name}")


# ---------------------------------------------------------------------
# Grids de hiperparámetros para nested CV (GridSearchCV interno)
# ---------------------------------------------------------------------
# Grid "exhaustivo" (v2): ampliado a partir de los resultados del grid
# "mediano" original. Decisiones basadas en gridsearch_comparison_results
# _best_params.csv del usuario (configs B y D, 159 muestras reales):
#
#   - SVM-RBF: C=100 (el valor MÁS ALTO del grid anterior) fue elegido
#     con mucha frecuencia -> el óptimo probablemente está por encima;
#     se extiende el rango hacia arriba (hasta 1000). gamma=0.001 (el
#     valor MÁS BAJO) también dominó -> se extiende hacia abajo (hasta
#     0.0001).
#   - kNN: n_neighbors NO convergía (variaba entre 3-11 sin patrón claro
#     entre folds/semillas) -> se ensancha el rango (hasta 15) por si el
#     óptimo está fuera de lo ya explorado, aunque la falta de
#     convergencia también puede ser inestabilidad inherente con la
#     clase asphyxia (n=6), no necesariamente un problema de rango.
#   - MLP: alpha y arquitectura tampoco convergían fuertemente -> se
#     agregan más valores intermedios.
#   - SVM-lineal y SVM-polinomial: ya convergían bien (C=0.1 dominante,
#     degree=2 dominante) -> se amplían de todas formas por pedido
#     explícito del usuario de ampliar TODOS los clasificadores por
#     igual, no porque haya evidencia fuerte de que lo necesiten.
#
# Este grid es considerablemente más caro que el "mediano": nested CV
# multiplica el costo por cada combinación nueva, así que se documenta
# el conteo de combinaciones por clasificador para poder estimar tiempo.

HYPERPARAM_GRIDS = {
    "MLP": {
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2] if TORCH_CUDA_AVAILABLE else [1e-5, 1e-4, 1e-3, 1e-2],
        "hidden1": [16, 32, 64, 128] if TORCH_CUDA_AVAILABLE else None,
        "hidden2": [8, 16, 32, 64] if TORCH_CUDA_AVAILABLE else None,
        "hidden_layer_sizes": [(16, 8), (32, 16), (64, 32), (128, 64)] if not TORCH_CUDA_AVAILABLE else None,
        "learning_rate_init": [1e-4, 1e-3, 1e-2] if not TORCH_CUDA_AVAILABLE else None,
    },
    "SVM-lineal": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },
    "SVM-polinomial": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "degree": [2, 3, 4, 5],
        "coef0": [0.0, 0.5, 1.0, 2.0],
    },
    "SVM-RBF": {
        # Ampliado respecto al grid mediano: C hacia arriba (el valor
        # máximo anterior, 100, fue elegido frecuentemente) y gamma
        # hacia abajo (el valor mínimo anterior, 0.001, también
        # dominaba) -- ambos bordes sugerían que el grid anterior no
        # bracketaba bien el óptimo real.
        "C": [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0],
        "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1.0],
    },
    "kNN": {
        # Ampliado: el grid mediano (3-11) no mostraba convergencia
        # clara entre folds/semillas. Se agregan vecinos más numerosos
        # (13, 15) por si el óptimo cae ahí, aunque con la clase
        # asphyxia teniendo solo 6 muestras totales, n_neighbors muy
        # altos pueden no tener sentido práctico dentro de esa clase.
        "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],  # distancia Manhattan (p=1) vs Euclidiana (p=2), no explorado antes
    },
}

# Conteo de combinaciones por clasificador, para estimar tiempo antes de
# correr (cada combinación implica un fit completo dentro de cada fold
# interno de GridSearchCV, multiplicado por outer_folds x n_seeds x
# inner_folds):
#   MLP:            4 (alpha) x 4 (hidden1) x 4 (hidden2)         = 64
#   SVM-lineal:     6 (C)                                          = 6
#   SVM-polinomial: 5 (C) x 4 (degree) x 4 (coef0)                = 80
#   SVM-RBF:        6 (C) x 7 (gamma)                             = 42
#   kNN:            7 (n_neighbors) x 2 (weights) x 2 (p)         = 28
# vs. grid mediano anterior: MLP=8, SVM-lineal=4, SVM-poly=18,
# SVM-RBF=16, kNN=10 -- el más caro (SVM-polinomial y MLP) crece ~4-8x.


def get_hyperparam_grid(name: str) -> dict:
    """Regresa el grid de hiperparámetros para un clasificador, limpiando
    entradas None (que solo aplican según el backend disponible)."""
    grid = HYPERPARAM_GRIDS[name]
    return {k: v for k, v in grid.items() if v is not None}
