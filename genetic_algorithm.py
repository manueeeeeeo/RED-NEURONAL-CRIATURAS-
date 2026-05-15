# =============================================================================
# genetic_algorithm.py — Algoritmo genético con mutación adaptativa
# Evoluciona la población de criaturas generación a generación.
# Fitness = distancia horizontal recorrida (sin checkpoints).
# =============================================================================

import numpy as np
import random
from neural_network import RedNeuronal
from config import (
    TAMANO_POBLACION,
    TASA_MUTACION, TASA_MUTACION_MIN, TASA_MUTACION_DECAY,
    MAGNITUD_MUTACION, MAGNITUD_MUTACION_MIN,
    ELITISMO, PORCENTAJE_PADRES, PORCENTAJE_ALEATORIOS,
)


class AlgoritmoGenetico:
    """
    Gestiona la evolución de la población de criaturas generación a generación.

    ► Selección:  Torneo (elige el mejor de k candidatos aleatorios)
    ► Cruce:      Uniforme ponderado — el padre con más fitness aporta más genes
    ► Mutación:   Gaussiana adaptativa — se reduce con mejora, sube con estancamiento
    ► Élite:      Top-N pasan intactos a la siguiente generación

    MUTACIÓN ADAPTATIVA:
        Generación con mejora  → reducir tasa/magnitud (explotar la solución)
        Generación sin mejora  → aumentar tasa/magnitud (explorar más espacio)

    CRUCE PONDERADO:
        P(gen_a) = fitness_a / (fitness_a + fitness_b)
        El padre con mejor fitness contribuye con más genes al hijo.
    """

    def __init__(self):
        self.generacion:              int   = 0
        self.mejor_fitness_historico: float = 0.0
        self.mejor_genoma:            list | None = None
        self.historial_fitness:       list[float] = []  # mejor por generación
        self.historial_media:         list[float] = []  # media por generación

        # Parámetros adaptativos (cambian con la evolución)
        self._tasa_mutacion     = TASA_MUTACION
        self._magnitud_mutacion = MAGNITUD_MUTACION
        self._gens_sin_mejora   = 0   # contador de estancamiento

    # =========================================================================
    # Punto de entrada principal
    # =========================================================================

    def evolucionar(self, criaturas: list) -> list:
        """
        Recibe la lista de criaturas de la generación actual.
        Devuelve una lista de genomas para la siguiente generación.
        Cada elemento puede ser list[np.ndarray] o None (→ criatura aleatoria).
        """
        self.generacion += 1

        # Ordenar por fitness (mejor primero)
        criaturas_ord = sorted(criaturas, key=lambda c: c.fitness, reverse=True)

        mejor_fitness = criaturas_ord[0].fitness
        media_fitness = float(np.mean([c.fitness for c in criaturas]))

        self.historial_fitness.append(mejor_fitness)
        self.historial_media.append(media_fitness)

        # Actualizar mejor histórico y ajustar mutación
        umbral_mejora = self.mejor_fitness_historico * 1.005  # 0.5% de mejora mínima
        if mejor_fitness > umbral_mejora:
            self.mejor_fitness_historico = mejor_fitness
            self.mejor_genoma            = criaturas_ord[0].obtener_genoma()
            self._gens_sin_mejora        = 0
            # Reducir mutación: explotar la mejora
            self._tasa_mutacion = max(
                TASA_MUTACION_MIN,
                self._tasa_mutacion * TASA_MUTACION_DECAY
            )
            self._magnitud_mutacion = max(
                MAGNITUD_MUTACION_MIN,
                self._magnitud_mutacion * TASA_MUTACION_DECAY
            )
        else:
            self._gens_sin_mejora += 1
            # Si lleva muchas generaciones sin mejorar → aumentar exploración
            if self._gens_sin_mejora > 8:
                self._tasa_mutacion = min(
                    TASA_MUTACION,
                    self._tasa_mutacion * 1.08
                )
                self._magnitud_mutacion = min(
                    MAGNITUD_MUTACION,
                    self._magnitud_mutacion * 1.08
                )

        # Construir nueva generación
        nuevos_genomas: list = []

        # 1. Élite: los mejores pasan sin modificación
        n_elite = min(ELITISMO, len(criaturas_ord))
        for i in range(n_elite):
            nuevos_genomas.append(criaturas_ord[i].obtener_genoma())

        # 2. Pool de padres: los mejores PORCENTAJE_PADRES de la población
        n_padres = max(2, int(len(criaturas_ord) * PORCENTAJE_PADRES))
        padres   = criaturas_ord[:n_padres]

        # 3. Reservar hueco para individuos completamente aleatorios
        n_random  = max(1, int(TAMANO_POBLACION * PORCENTAJE_ALEATORIOS))
        n_hijos   = TAMANO_POBLACION - n_elite - n_random

        # 4. Completar con hijos de cruce + mutación
        for _ in range(n_hijos):
            padre_a = self._seleccion_torneo(padres)
            padre_b = self._seleccion_torneo(padres)
            intentos = 0
            while padre_a is padre_b and len(padres) > 1 and intentos < 5:
                padre_b = self._seleccion_torneo(padres)
                intentos += 1
            hijo = self._cruce_ponderado(padre_a, padre_b)
            hijo = self._mutar(hijo)
            nuevos_genomas.append(hijo)

        # 5. Individuos aleatorios (None → Criatura los inicializa aleatoriamente)
        for _ in range(n_random):
            nuevos_genomas.append(None)

        return nuevos_genomas

    # =========================================================================
    # Selección por torneo
    # =========================================================================

    def _seleccion_torneo(self, padres: list, k: int = 3):
        """
        Selección por torneo: escoge el mejor de k candidatos aleatorios.
        k=3 equilibra presión selectiva y diversidad.
        """
        candidatos = random.sample(padres, min(k, len(padres)))
        return max(candidatos, key=lambda c: c.fitness)

    # =========================================================================
    # Cruce ponderado por fitness
    # =========================================================================

    def _cruce_ponderado(self, padre_a, padre_b) -> list[np.ndarray]:
        """
        Cruce uniforme PONDERADO: cada gen del hijo se toma de padre_a con
        probabilidad p = fitness_a / (fitness_a + fitness_b).
        El padre mejor contribuye proporcionalmente más genes al hijo.
        """
        genoma_a = padre_a.obtener_genoma()
        genoma_b = padre_b.obtener_genoma()

        # Calcular probabilidad de heredar del padre A
        # Asegurarse de que los fitness sean positivos para el peso
        fit_a = max(0.0, padre_a.fitness)
        fit_b = max(0.0, padre_b.fitness)
        fit_total = fit_a + fit_b
        prob_a    = 0.5 if fit_total <= 0 else fit_a / fit_total

        hijo = []
        for wa, wb in zip(genoma_a, genoma_b):
            mascara = np.random.rand(*wa.shape) < prob_a
            w_hijo  = np.where(mascara, wa, wb)
            hijo.append(w_hijo)
        return hijo

    # =========================================================================
    # Mutación gaussiana adaptativa
    # =========================================================================

    def _mutar(self, genoma: list[np.ndarray]) -> list[np.ndarray]:
        """
        Mutación gaussiana con tasa y magnitud que se ajustan dinámicamente.
        La perturbación sigue Normal(0, _magnitud_mutacion).
        Cada peso se muta de forma independiente con probabilidad _tasa_mutacion.
        """
        mutado = []
        for w in genoma:
            mascara = np.random.rand(*w.shape) < self._tasa_mutacion
            ruido   = np.random.randn(*w.shape) * self._magnitud_mutacion
            mutado.append(w + mascara * ruido)
        return mutado

    # =========================================================================
    # Estadísticas
    # =========================================================================

    def estadisticas(self, criaturas: list) -> dict:
        """Devuelve un dict con las métricas de la generación actual."""
        fitnesses       = [c.fitness for c in criaturas]
        mejor_actual    = max(fitnesses) if fitnesses else 0.0
        # El récord histórico es el máximo entre guardado y actual
        mejor_historico = max(self.mejor_fitness_historico, mejor_actual)
        return {
            "generacion":        self.generacion,
            "mejor_fitness":     mejor_actual,
            "media_fitness":     float(np.mean(fitnesses)) if fitnesses else 0.0,
            "mejor_historico":   mejor_historico,
            "tasa_mutacion":     self._tasa_mutacion,
            "magnitud_mutacion": self._magnitud_mutacion,
            "gens_sin_mejora":   self._gens_sin_mejora,
        }

    # =========================================================================
    # Persistencia del mejor agente
    # =========================================================================

    def guardar_mejor(self, ruta: str):
        """Guarda el genoma del mejor agente histórico en disco."""
        if self.mejor_genoma is None:
            return
        vector = RedNeuronal.aplanar_genoma(self.mejor_genoma)
        np.save(ruta, vector)
        print(f"[GA] Mejor agente guardado en '{ruta}'")

    def cargar_mejor(self, ruta: str) -> list | None:
        """Intenta cargar un agente guardado. Devuelve None si no existe."""
        try:
            vector  = np.load(ruta)
            red_tmp = RedNeuronal()
            genoma  = RedNeuronal.reconstruir_genoma(vector, red_tmp.topologia)
            print(f"[GA] Agente cargado desde '{ruta}'")
            return genoma
        except Exception as e:
            print(f"[GA] No se pudo cargar '{ruta}': {e}. Iniciando desde cero.")
            return None
