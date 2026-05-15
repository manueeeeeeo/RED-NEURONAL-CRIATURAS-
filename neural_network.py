# =============================================================================
# neural_network.py — Red neuronal feedforward con NumPy (sin frameworks)
# Arquitectura: 10 entradas → capas ocultas (tanh) → 4 salidas (tanh)
# Todas las capas usan tanh para que los outputs estén en [-1, 1],
# lo que es ideal para controlar velocidades de motores.
# =============================================================================

import numpy as np
from config import ENTRADAS, CAPAS_OCULTAS, SALIDAS


# -----------------------------------------------------------------------------
# Función de activación
# -----------------------------------------------------------------------------

def tanh_act(x: np.ndarray) -> np.ndarray:
    """Tangente hiperbólica: salidas en [-1, 1]. Ideal para control de motores."""
    return np.tanh(x)


# -----------------------------------------------------------------------------
# Clase RedNeuronal
# -----------------------------------------------------------------------------

class RedNeuronal:
    """
    Red neuronal densa hacia adelante (feedforward) implementada con NumPy.
    No usa TensorFlow ni PyTorch: los pesos son arrays NumPy puros que
    se serializan directamente como 'genoma' del algoritmo genético.

    Arquitectura:
        ENTRADAS (10) → [CAPAS_OCULTAS] → SALIDAS (4)
        Todas las capas usan activación tanh.

    Convención de pesos:
        Cada matriz w tiene forma (n_entradas + 1, n_salidas),
        donde la última fila es el vector de bias.
        Multiplicación: z = [activacion | 1.0] @ w
    """

    def __init__(self, pesos: list | None = None):
        # Topología completa: [entradas, ocultas..., salidas]
        self.topologia: list[int] = [ENTRADAS] + CAPAS_OCULTAS + [SALIDAS]

        # Inicializar o cargar pesos
        if pesos is not None:
            self.pesos = [w.copy() for w in pesos]
        else:
            self.pesos = self._inicializar_aleatorio()

        # Activaciones del último forward pass (para visualización futura)
        self.activaciones_ultimas: list[np.ndarray] = []

    # -------------------------------------------------------------------------
    # Inicialización de pesos
    # -------------------------------------------------------------------------

    def _inicializar_aleatorio(self) -> list[np.ndarray]:
        """
        Inicialización Xavier/Glorot para tanh.
        Escala = sqrt(2 / (n_in + n_out)) — equilibra la varianza de entrada/salida.
        La última fila de cada matriz (bias) se inicializa en 0.
        """
        pesos = []
        for i in range(len(self.topologia) - 1):
            n_in  = self.topologia[i]
            n_out = self.topologia[i + 1]
            # Xavier: mejor para tanh
            escala = np.sqrt(2.0 / (n_in + n_out))
            # Forma: (n_in + 1, n_out) — fila extra = bias
            w = np.random.randn(n_in + 1, n_out) * escala
            w[-1, :] = 0.0   # Bias inicial = 0
            pesos.append(w)
        return pesos

    # -------------------------------------------------------------------------
    # Inferencia (forward pass)
    # -------------------------------------------------------------------------

    def predecir(self, entradas: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante.
        Guarda activaciones intermedias en self.activaciones_ultimas
        para posible visualización de la red en tiempo real.

        Args:
            entradas: array de shape (ENTRADAS,), valores en [-1, 1]
        Returns:
            array de shape (SALIDAS,), valores en [-1, 1]
        """
        x = np.asarray(entradas, dtype=np.float64)
        self.activaciones_ultimas = [x.copy()]  # capa de entrada

        for w in self.pesos:
            # Añadir término de bias (valor 1.0) al vector de activación
            x_bias = np.append(x, 1.0)
            z = x_bias @ w                     # (n_in+1,) @ (n_in+1, n_out) → (n_out,)
            x = tanh_act(z)                    # tanh en todas las capas
            self.activaciones_ultimas.append(x.copy())

        return x

    # -------------------------------------------------------------------------
    # Serialización del genoma (para el algoritmo genético)
    # -------------------------------------------------------------------------

    def obtener_genoma(self) -> list[np.ndarray]:
        """Devuelve copia de los pesos (genoma) para el algoritmo genético."""
        return [w.copy() for w in self.pesos]

    def establecer_genoma(self, genoma: list[np.ndarray]) -> None:
        """Carga un genoma externo en la red."""
        self.pesos = [w.copy() for w in genoma]

    @staticmethod
    def aplanar_genoma(genoma: list[np.ndarray]) -> np.ndarray:
        """Convierte la lista de matrices en un único vector 1D (para guardar en disco)."""
        return np.concatenate([w.flatten() for w in genoma])

    @staticmethod
    def reconstruir_genoma(vector: np.ndarray, topologia: list[int]) -> list[np.ndarray]:
        """Reconstruye la lista de matrices desde un vector 1D guardado."""
        pesos = []
        idx = 0
        for i in range(len(topologia) - 1):
            n_in  = topologia[i] + 1   # +1 por el bias
            n_out = topologia[i + 1]
            tam   = n_in * n_out
            pesos.append(vector[idx: idx + tam].reshape(n_in, n_out))
            idx += tam
        return pesos

    def __repr__(self) -> str:
        return f"RedNeuronal(topologia={self.topologia}, activacion=tanh)"
