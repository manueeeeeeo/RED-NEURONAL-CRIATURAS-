# =============================================================================
# main.py — Punto de entrada del simulador Criatura-Evolution
# Ejecuta: python main.py
#
# CONTROLES:
#   ESPACIO      Pausar / reanudar
#   1/2/3/5      Velocidad ×1 / ×2 / ×3 / ×5
#   R            Reiniciar simulación desde cero
#   ESC          Salir
# =============================================================================

import sys
import math
import pygame
import numpy as np

from config import (
    ANCHO_VENTANA, ALTO_VENTANA, FPS, TITULO,
    TAMANO_POBLACION, MAX_GENERACIONES,
    DURACION_SIMULACION,
    COLOR_FONDO, COLOR_SUELO, COLOR_HUD, COLOR_ACENTO, COLOR_MEJOR,
    Y_SUELO, X_INICIO_CRIATURA,
    MOSTRAR_GRAFICA, GRAFICA_GENS,
)
from simulation import Simulacion
from creature import Criatura
from genetic_algorithm import AlgoritmoGenetico


# =============================================================================
# Paleta de colores para la generación
# =============================================================================

PALETA_COLORES = [
    (0,   190, 255),   # Azul eléctrico
    (0,   235, 130),   # Verde neón
    (200,  80, 255),   # Púrpura
    (255, 165,   0),   # Naranja
    (255,  70,  70),   # Rojo
    (80,  255, 200),   # Turquesa
    (185, 255,  80),   # Lima
    (255,  50, 185),   # Rosa
    (50,  200, 255),   # Cian
    (255, 220,  50),   # Amarillo
]


# =============================================================================
# Clase principal del simulador
# =============================================================================

class Simulador:
    """
    Gestiona el bucle principal:
      - Evaluación SECUENCIAL: una criatura a la vez
      - Renderizado en tiempo real de la criatura evaluada
      - Cámara que sigue al torso
      - HUD con estadísticas
      - Gráfica de evolución del fitness
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITULO)
        self.pantalla = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
        self.reloj    = pygame.time.Clock()

        # Fuentes
        self.fuente_grande  = pygame.font.SysFont("consolas", 20, bold=True)
        self.fuente_normal  = pygame.font.SysFont("consolas", 15)
        self.fuente_pequena = pygame.font.SysFont("consolas", 12)
        self.fuente_titulo  = pygame.font.SysFont("consolas", 13, bold=True)

        # Núcleo de física y evolución
        self.sim = Simulacion()
        self.ga  = AlgoritmoGenetico()

        # Estado de simulación
        self.generacion_actual   = 0
        self.pausa               = False
        self.velocidad_sim       = 1     # steps de física por frame
        self._frame_cnt          = 0     # para skip de render a alta velocidad

        # Genomas de la generación actual (list de list[np.ndarray] o None)
        self._genomas_generacion: list = []

        # Índice de la criatura que se está evaluando ahora
        self._idx_evaluando: int = 0

        # Criatura actual en evaluación
        self._criatura_actual: Criatura | None = None
        self._frame_criatura: int = 0   # frames que lleva la criatura actual

        # Mejor fitness de la generación actual (para el HUD)
        self._mejor_esta_gen: float = 0.0
        # Lista de fitness acumulados en esta generación (para evolucionar al final)
        self._criaturas_evaluadas: list[Criatura] = []

        # Iniciar la primera generación
        self._nueva_generacion()

    # =========================================================================
    # Gestión de generaciones
    # =========================================================================

    def _nueva_generacion(self, genomas: list | None = None):
        """
        Prepara los genomas para evaluar la nueva generación.
        Si genomas es None → primera generación (todo aleatorio).
        """
        self.generacion_actual += 1

        if genomas is None:
            # Primera generación: genomas = None para cada individuo
            self._genomas_generacion = [None] * TAMANO_POBLACION
        else:
            self._genomas_generacion = genomas

        # Resetear estado de evaluación
        self._idx_evaluando       = 0
        self._frame_criatura      = 0
        self._mejor_esta_gen      = 0.0
        self._criaturas_evaluadas = []

        # Lanzar la primera criatura
        self._iniciar_criatura(0)

    def _iniciar_criatura(self, idx: int):
        """
        Resetea el espacio físico y crea la criatura con índice idx.
        """
        # Limpiar la criatura anterior del espacio
        self.sim.resetear()

        genoma = self._genomas_generacion[idx]

        # El élite 0 lleva color especial
        if idx == 0 and genoma is not None:
            color = COLOR_MEJOR
        else:
            color = PALETA_COLORES[idx % len(PALETA_COLORES)]

        self._criatura_actual = Criatura(
            self.sim.espacio,
            genoma=genoma,
            color=color,
        )
        self._frame_criatura = 0

    def _finalizar_criatura_actual(self):
        """
        Guarda la criatura evaluada y pasa a la siguiente.
        Si ya se evaluaron todas, llama al GA para evolucionar.
        """
        if self._criatura_actual is not None:
            self._criaturas_evaluadas.append(self._criatura_actual)
            fit = self._criatura_actual.fitness
            if fit > self._mejor_esta_gen:
                self._mejor_esta_gen = fit

        self._idx_evaluando += 1

        if self._idx_evaluando < TAMANO_POBLACION:
            # Siguiente criatura
            self._iniciar_criatura(self._idx_evaluando)
        else:
            # Generación completa → evolucionar
            self._evolucionar()

    def _evolucionar(self):
        """Llama al GA y prepara la siguiente generación."""
        nuevos_genomas = self.ga.evolucionar(self._criaturas_evaluadas)
        self._nueva_generacion(nuevos_genomas)

    # =========================================================================
    # Bucle principal
    # =========================================================================

    def ejecutar(self):
        corriendo = True

        while corriendo:
            # --- Eventos de teclado y cierre ---
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    corriendo = False
                elif evento.type == pygame.KEYDOWN:
                    corriendo = self._manejar_teclado(evento.key)

            if not self.pausa:
                self._frame_cnt += 1

                # Realizar N pasos de física + N actualizaciones de criatura
                for _ in range(self.velocidad_sim):
                    if self._criatura_actual is None:
                        break

                    # Avanzar física
                    self.sim.paso(1 / 60)

                    # Actualizar la criatura (sensores → NN → motores → fitness)
                    self._criatura_actual.actualizar()
                    self._frame_criatura += 1

                    # Comprobar si hay que pasar a la siguiente criatura
                    tiempo_agotado = self._frame_criatura >= DURACION_SIMULACION
                    muerta         = not self._criatura_actual.esta_viva()

                    if tiempo_agotado or muerta:
                        self._finalizar_criatura_actual()
                        break  # resetear al exterior del bucle de velocidad

                # Comprobar límite de generaciones
                if MAX_GENERACIONES > 0 and self.generacion_actual > MAX_GENERACIONES:
                    corriendo = False

            # --- Renderizado (puede saltar frames a altas velocidades) ---
            if self._debe_renderizar():
                self._renderizar()

            self.reloj.tick(FPS)

        pygame.quit()
        sys.exit()

    def _debe_renderizar(self) -> bool:
        """
        A velocidades altas, saltamos algunos renders para ganar rendimiento.
          ×1–×3  → render cada frame
          ×5     → render cada 2 frames
        En pausa siempre renderiza.
        """
        if self.pausa:
            return True
        if self.velocidad_sim <= 3:
            return True
        return self._frame_cnt % 2 == 0

    # =========================================================================
    # Teclado
    # =========================================================================

    def _manejar_teclado(self, tecla: int) -> bool:
        """Procesa una tecla. Devuelve False si hay que salir."""
        if tecla == pygame.K_ESCAPE:
            return False
        elif tecla == pygame.K_SPACE:
            self.pausa = not self.pausa
        elif tecla == pygame.K_1:
            self.velocidad_sim = 1
        elif tecla == pygame.K_2:
            self.velocidad_sim = 2
        elif tecla == pygame.K_3:
            self.velocidad_sim = 3
        elif tecla == pygame.K_5:
            self.velocidad_sim = 5
        elif tecla == pygame.K_r:
            # Reiniciar desde cero
            self.sim             = Simulacion()
            self.ga              = AlgoritmoGenetico()
            self.generacion_actual = 0
            self._frame_cnt        = 0
            self._nueva_generacion()
        return True

    # =========================================================================
    # Renderizado principal
    # =========================================================================

    def _renderizar(self):
        """Dibuja el frame completo."""
        self.pantalla.fill(COLOR_FONDO)

        # Calcular offset de cámara (sigue al torso de la criatura actual)
        camera_x = 0.0
        if self._criatura_actual is not None:
            tx       = self._criatura_actual.torso.position.x
            camera_x = -(tx - ANCHO_VENTANA * 0.35)

        # 1. Fondo y suelo
        self._dibujar_suelo(camera_x)

        # 2. Criatura actual
        if self._criatura_actual is not None:
            self._criatura_actual.dibujar(self.pantalla, camera_x)

        # 3. HUD
        self._dibujar_hud()

        # 4. Gráfica de fitness
        if MOSTRAR_GRAFICA:
            self._dibujar_grafica_fitness()

        # 5. Controles (esquina inferior izquierda)
        self._dibujar_ayuda()

        # 6. Indicador de pausa
        if self.pausa:
            p_surf = self.fuente_grande.render(
                "PAUSADO  —  ESPACIO para continuar", True, (255, 200, 50)
            )
            px = ANCHO_VENTANA // 2 - p_surf.get_width() // 2
            pygame.draw.rect(
                self.pantalla, (20, 20, 40),
                (px - 10, 16, p_surf.get_width() + 20, 30), border_radius=6
            )
            self.pantalla.blit(p_surf, (px, 18))

        pygame.display.flip()

    # =========================================================================
    # Dibujo del suelo
    # =========================================================================

    def _dibujar_suelo(self, camera_x: float):
        """
        Dibuja el suelo como un rectángulo horizontal con marcas de distancia.
        """
        y_suelo_px = Y_SUELO

        # Rectángulo del suelo
        pygame.draw.rect(
            self.pantalla,
            COLOR_SUELO,
            (0, y_suelo_px, ANCHO_VENTANA, ALTO_VENTANA - y_suelo_px)
        )
        # Línea superior del suelo
        pygame.draw.line(
            self.pantalla,
            (100, 120, 140),
            (0, y_suelo_px), (ANCHO_VENTANA, y_suelo_px), 2
        )

        # Marcas de distancia cada 100 px en el mundo
        # La marca del mundo está en x_mundo; en pantalla → x_mundo + camera_x
        inicio_marca = int((-camera_x) // 100) * 100  # primera marca visible
        for i in range(30):
            x_mundo  = inicio_marca + i * 100
            x_pant   = int(x_mundo + camera_x)
            if -20 < x_pant < ANCHO_VENTANA + 20:
                pygame.draw.line(
                    self.pantalla, (80, 95, 110),
                    (x_pant, y_suelo_px - 8), (x_pant, y_suelo_px + 4), 1
                )
                # Etiqueta de distancia (en píxeles desde el inicio)
                dist  = x_mundo - X_INICIO_CRIATURA
                label = self.fuente_pequena.render(f"{int(dist)}", True, (90, 105, 120))
                self.pantalla.blit(label, (x_pant - label.get_width() // 2, y_suelo_px + 6))

    # =========================================================================
    # HUD
    # =========================================================================

    def _dibujar_hud(self):
        """Panel informativo en la esquina superior izquierda."""
        stats = self.ga.estadisticas(self._criaturas_evaluadas) if self._criaturas_evaluadas else {
            "generacion":       self.generacion_actual,
            "mejor_fitness":    0.0,
            "media_fitness":    0.0,
            "mejor_historico":  0.0,
            "tasa_mutacion":    0.18,
            "magnitud_mutacion": 0.40,
            "gens_sin_mejora":  0,
        }

        fit_actual = self._criatura_actual.fitness if self._criatura_actual else 0.0

        vel_texto = {1: "x1", 2: "x2", 3: "x3", 5: "x5"}
        vel_str   = vel_texto.get(self.velocidad_sim, f"x{self.velocidad_sim}")

        lineas = [
            ("GENERACION",        f"{self.generacion_actual:>4}",                     COLOR_ACENTO),
            ("EVALUANDO",         f"{self._idx_evaluando + 1:>2} / {TAMANO_POBLACION}", COLOR_HUD),
            ("FITNESS ACTUAL",    f"{fit_actual:>8.1f} px",                           (255, 190, 60)),
            ("MEJOR ESTA GEN",    f"{self._mejor_esta_gen:>8.1f} px",                 (200, 220, 255)),
            ("RECORD HISTORICO",  f"{stats['mejor_historico']:>8.1f} px",             COLOR_MEJOR),
            ("MUTACION",          f"{stats['tasa_mutacion']:.3f} | "
                                  f"{stats['magnitud_mutacion']:.3f}",                (140, 180, 200)),
            ("VELOCIDAD",         vel_str,                                             (255, 160, 50)),
        ]

        padding  = 12
        ancho    = 310
        alto     = padding * 2 + 28 + len(lineas) * 20 + 8
        hud_surf = pygame.Surface((ancho, alto), pygame.SRCALPHA)
        hud_surf.fill((14, 14, 32, 215))
        pygame.draw.rect(hud_surf, COLOR_ACENTO, (0, 0, ancho, alto), 2, border_radius=6)
        self.pantalla.blit(hud_surf, (10, 10))

        # Título
        t_surf = self.fuente_grande.render("CRIATURA-EVOLUTION", True, COLOR_ACENTO)
        self.pantalla.blit(t_surf, (16, 15))

        # Filas de estadísticas
        for i, (etiqueta, valor, c_val) in enumerate(lineas):
            y = 15 + 26 + i * 20
            et_s  = self.fuente_normal.render(etiqueta + ":", True, (140, 155, 185))
            val_s = self.fuente_normal.render(valor,           True, c_val)
            self.pantalla.blit(et_s,  (16, y))
            self.pantalla.blit(val_s, (10 + ancho - val_s.get_width() - 14, y))

        # Barra de progreso de la evaluación actual
        barra_y  = 15 + 26 + len(lineas) * 20 + 4
        barra_w  = ancho - 20
        progreso = self._frame_criatura / max(DURACION_SIMULACION, 1)
        pygame.draw.rect(self.pantalla, (35, 35, 60), (16, barra_y, barra_w, 7), border_radius=3)
        if progreso > 0:
            pygame.draw.rect(
                self.pantalla, (80, 160, 255),
                (16, barra_y, int(barra_w * progreso), 7), border_radius=3
            )

    # =========================================================================
    # Gráfica de evolución del fitness
    # =========================================================================

    def _dibujar_grafica_fitness(self):
        """
        Muestra la historia de fitness por generación (esquina inferior derecha).
        Línea verde = mejor fitness. Línea gris = media.
        """
        if len(self.ga.historial_fitness) < 2:
            return

        panel_w = 310
        panel_h = 160
        panel_x = ANCHO_VENTANA - panel_w - 10
        panel_y = ALTO_VENTANA  - panel_h - 10

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((12, 12, 28, 215))
        pygame.draw.rect(panel, (55, 60, 100), (0, 0, panel_w, panel_h), 1, border_radius=5)

        tit = self.fuente_titulo.render("EVOLUCION DEL FITNESS", True, (140, 150, 195))
        panel.blit(tit, (10, 5))

        hf = self.ga.historial_fitness[-GRAFICA_GENS:]
        hm = self.ga.historial_media[-GRAFICA_GENS:]

        max_val = max(max(hf), 1.0)
        min_val = min(min(hm), 0.0)
        rango   = max(max_val - min_val, 1.0)

        mx, my = 10, 22
        gw = panel_w - mx * 2
        gh = panel_h - my - 14
        n  = len(hf)

        # Líneas de cuadrícula
        for frac in (0.25, 0.5, 0.75):
            yg = my + int(gh * (1 - frac))
            pygame.draw.line(panel, (35, 38, 65), (mx, yg), (panel_w - mx, yg), 1)
            lbl = self.fuente_pequena.render(
                f"{int(min_val + rango * frac)}", True, (70, 80, 110)
            )
            panel.blit(lbl, (mx, yg - 8))

        def puntos_de(valores):
            pts = []
            for i, v in enumerate(valores):
                x = mx + int(gw * i / max(n - 1, 1))
                y = my + int(gh * (1 - (v - min_val) / rango))
                pts.append((x, y))
            return pts

        pts_mejor = puntos_de(hf)
        pts_media = puntos_de(hm)

        # Relleno bajo la curva del mejor fitness
        if len(pts_mejor) > 1:
            fill = [pts_mejor[0]] + pts_mejor + [(pts_mejor[-1][0], my + gh), (pts_mejor[0][0], my + gh)]
            pygame.draw.polygon(panel, (0, 70, 50), fill)

        # Curva media (gris)
        if len(pts_media) > 1:
            pygame.draw.lines(panel, (90, 100, 130), False, pts_media, 1)

        # Curva mejor (verde)
        if len(pts_mejor) > 1:
            pygame.draw.lines(panel, (0, 220, 140), False, pts_mejor, 2)

        # Punto actual
        if pts_mejor:
            pygame.draw.circle(panel, (0, 255, 160), pts_mejor[-1], 3)
            val_txt = self.fuente_pequena.render(f"{hf[-1]:.0f}", True, (0, 220, 140))
            panel.blit(val_txt, (
                pts_mejor[-1][0] - val_txt.get_width() // 2,
                pts_mejor[-1][1] - 14
            ))

        # Leyenda
        pygame.draw.line(panel, (0, 200, 130), (panel_w - 100, 5), (panel_w - 80, 5), 2)
        panel.blit(self.fuente_pequena.render("mejor", True, (0, 200, 130)), (panel_w - 78, 0))
        pygame.draw.line(panel, (90, 100, 130), (panel_w - 100, 14), (panel_w - 80, 14), 1)
        panel.blit(self.fuente_pequena.render("media", True, (90, 100, 130)), (panel_w - 78, 10))

        self.pantalla.blit(panel, (panel_x, panel_y))

    # =========================================================================
    # Ayuda de teclado
    # =========================================================================

    def _dibujar_ayuda(self):
        atajos = [
            "[SPACE]   Pausa",
            "[1/2/3/5] Velocidad",
            "[R]       Reiniciar",
            "[ESC]     Salir",
        ]
        y_base = ALTO_VENTANA - len(atajos) * 15 - 8
        for i, txt in enumerate(atajos):
            surf = self.fuente_pequena.render(txt, True, (85, 95, 125))
            self.pantalla.blit(surf, (10, y_base + i * 15))


# =============================================================================
# Punto de entrada
# =============================================================================

if __name__ == "__main__":
    sim = Simulador()
    sim.ejecutar()
