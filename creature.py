# =============================================================================
# creature.py — Criatura bípeda controlada por red neuronal
# Anatomía: Torso → 2 muslos → 2 espinillas, con motores en cada articulación.
# El algoritmo genético evoluciona los pesos de la red que controla los motores.
# =============================================================================

import math
import pygame
import pymunk
import numpy as np

from config import (
    # Colores
    COLOR_TORSO, COLOR_PATA_ARRIBA, COLOR_PATA_ABAJO,
    COLOR_JOINT, COLOR_MEJOR, COLOR_MUERTO,
    # Anatomía
    TORSO_ANCHO, TORSO_ALTO, TORSO_MASA,
    MUSLO_ANCHO, MUSLO_ALTO, MUSLO_MASA,
    ESPINILLA_ANCHO, ESPINILLA_ALTO, ESPINILLA_MASA,
    # Motores
    MOTOR_FUERZA_MAX, MOTOR_RATE_MAX,
    # Física
    FRICCION_CUERPO,
    # Posición
    X_INICIO_CRIATURA, Y_SUELO,
    # Límites angulares
    LIMITE_MUSLO_RAD, LIMITE_ESPINILLA_MIN, LIMITE_ESPINILLA_MAX,
)
from neural_network import RedNeuronal


# Filtro de colisión: group=1 → las partes de la criatura NO colisionan entre sí
# pero SÍ colisionan con el suelo (que tiene group=0)
_FILTRO_CRIATURA = pymunk.ShapeFilter(group=1)


def _crear_segmento_cuerpo(
    espacio: pymunk.Space,
    masa: float,
    ancho: float,
    alto: float,
    pos: tuple[float, float],
) -> tuple[pymunk.Body, pymunk.Shape]:
    """
    Crea un Body dinámico con una Poly (rectángulo) y lo añade al espacio.
    Devuelve (body, shape).
    """
    inercia = pymunk.moment_for_box(masa, (ancho, alto))
    cuerpo  = pymunk.Body(masa, inercia)
    cuerpo.position = pos

    forma = pymunk.Poly.create_box(cuerpo, (ancho, alto))
    forma.friction   = FRICCION_CUERPO
    forma.elasticity = 0.0
    forma.filter     = _FILTRO_CRIATURA

    espacio.add(cuerpo, forma)
    return cuerpo, forma


def _crear_pivot(
    espacio: pymunk.Space,
    cuerpo_a: pymunk.Body,
    cuerpo_b: pymunk.Body,
    punto_mundo: tuple[float, float],
) -> pymunk.PivotJoint:
    """Crea un PivotJoint entre dos cuerpos en un punto del mundo."""
    joint = pymunk.PivotJoint(cuerpo_a, cuerpo_b, punto_mundo)
    joint.error_bias = pow(1.0 - 0.1, 60)   # amortiguación del joint
    espacio.add(joint)
    return joint


def _crear_motor(
    espacio: pymunk.Space,
    cuerpo_a: pymunk.Body,
    cuerpo_b: pymunk.Body,
) -> pymunk.SimpleMotor:
    """Crea un SimpleMotor entre dos cuerpos. Rate inicial = 0."""
    motor = pymunk.SimpleMotor(cuerpo_a, cuerpo_b, rate=0.0)
    motor.max_force = MOTOR_FUERZA_MAX
    espacio.add(motor)
    return motor


def _crear_limite_angular(
    espacio: pymunk.Space,
    cuerpo_a: pymunk.Body,
    cuerpo_b: pymunk.Body,
    min_rad: float,
    max_rad: float,
) -> pymunk.RotaryLimitJoint:
    """Crea un RotaryLimitJoint para limitar el giro relativo entre dos cuerpos."""
    limite = pymunk.RotaryLimitJoint(cuerpo_a, cuerpo_b, min_rad, max_rad)
    limite.error_bias = pow(1.0 - 0.3, 60)
    espacio.add(limite)
    return limite


# =============================================================================
# Clase Criatura
# =============================================================================

class Criatura:
    """
    Bípedo físico controlado por una red neuronal evolucionada.

    Anatomía (pymunk):
        [      TORSO      ]
            /           \\
        [MUSLO_IZQ]  [MUSLO_DER]
            |               |
        [ESPINILLA_IZQ] [ESPINILLA_DER]

    Fitness = distancia horizontal recorrida desde X_INICIO_CRIATURA.
    La simulación termina antes de DURACION_SIMULACION si:
        - El torso toca el suelo (posición Y cercana a Y_SUELO)
        - El torso se inclina demasiado (|ángulo| > 90°)
    """

    def __init__(
        self,
        espacio: pymunk.Space,
        genoma: list | None = None,
        color: tuple[int, int, int] | None = None,
    ):
        self.espacio  = espacio
        self.fitness  = 0.0
        self._viva    = True
        self.color    = color  # None = usar colores por defecto del config

        # Red neuronal — carga el genoma si se proporciona
        self.red = RedNeuronal(pesos=genoma)

        # Construir la anatomía física
        self._construir_cuerpo()

    # -------------------------------------------------------------------------
    # Construcción del cuerpo físico
    # -------------------------------------------------------------------------

    def _construir_cuerpo(self):
        """
        Crea todos los segmentos, articulaciones y motores en el espacio pymunk.
        Posición inicial: torso centrado en (X_INICIO, Y_SUELO - 120).
        """
        # --- Posiciones iniciales ---
        cx = float(X_INICIO_CRIATURA)
        # El torso está a 120 px sobre el suelo
        cy_torso = float(Y_SUELO) - 120.0

        # Puntos de anclaje de las caderas (borde inferior del torso)
        separacion_cadera = TORSO_ANCHO / 3.0
        cy_cadera         = cy_torso + TORSO_ALTO / 2.0

        # Muslos: su centro está MUSLO_ALTO/2 por debajo de la cadera
        cy_muslo    = cy_cadera + MUSLO_ALTO / 2.0
        cx_muslo_iz = cx - separacion_cadera
        cx_muslo_de = cx + separacion_cadera

        # Espinillas: su centro está ESPINILLA_ALTO/2 por debajo del muslo
        cy_espinilla_iz = cy_muslo + MUSLO_ALTO / 2.0 + ESPINILLA_ALTO / 2.0
        cy_espinilla_de = cy_muslo + MUSLO_ALTO / 2.0 + ESPINILLA_ALTO / 2.0

        # --- Crear segmentos ---
        self.torso, self._forma_torso = _crear_segmento_cuerpo(
            self.espacio, TORSO_MASA, TORSO_ANCHO, TORSO_ALTO, (cx, cy_torso)
        )
        self.muslo_iz, _ = _crear_segmento_cuerpo(
            self.espacio, MUSLO_MASA, MUSLO_ANCHO, MUSLO_ALTO,
            (cx_muslo_iz, cy_muslo)
        )
        self.muslo_de, _ = _crear_segmento_cuerpo(
            self.espacio, MUSLO_MASA, MUSLO_ANCHO, MUSLO_ALTO,
            (cx_muslo_de, cy_muslo)
        )
        self.espinilla_iz, _ = _crear_segmento_cuerpo(
            self.espacio, ESPINILLA_MASA, ESPINILLA_ANCHO, ESPINILLA_ALTO,
            (cx_muslo_iz, cy_espinilla_iz)
        )
        self.espinilla_de, _ = _crear_segmento_cuerpo(
            self.espacio, ESPINILLA_MASA, ESPINILLA_ANCHO, ESPINILLA_ALTO,
            (cx_muslo_de, cy_espinilla_de)
        )

        # --- Articulaciones (PivotJoint) ---
        # Cadera izquierda: torso ↔ muslo_iz
        punto_cadera_iz = (cx_muslo_iz, cy_cadera)
        _crear_pivot(self.espacio, self.torso, self.muslo_iz, punto_cadera_iz)

        # Cadera derecha: torso ↔ muslo_de
        punto_cadera_de = (cx_muslo_de, cy_cadera)
        _crear_pivot(self.espacio, self.torso, self.muslo_de, punto_cadera_de)

        # Rodilla izquierda: muslo_iz ↔ espinilla_iz
        punto_rodilla_iz = (cx_muslo_iz, cy_muslo + MUSLO_ALTO / 2.0)
        _crear_pivot(self.espacio, self.muslo_iz, self.espinilla_iz, punto_rodilla_iz)

        # Rodilla derecha: muslo_de ↔ espinilla_de
        punto_rodilla_de = (cx_muslo_de, cy_muslo + MUSLO_ALTO / 2.0)
        _crear_pivot(self.espacio, self.muslo_de, self.espinilla_de, punto_rodilla_de)

        # --- Límites angulares ---
        # Muslos: ±LIMITE_MUSLO_RAD respecto al torso
        _crear_limite_angular(
            self.espacio, self.torso, self.muslo_iz,
            -LIMITE_MUSLO_RAD, LIMITE_MUSLO_RAD
        )
        _crear_limite_angular(
            self.espacio, self.torso, self.muslo_de,
            -LIMITE_MUSLO_RAD, LIMITE_MUSLO_RAD
        )
        # Espinillas: solo hacia adelante (0° a 130°)
        _crear_limite_angular(
            self.espacio, self.muslo_iz, self.espinilla_iz,
            LIMITE_ESPINILLA_MIN, LIMITE_ESPINILLA_MAX
        )
        _crear_limite_angular(
            self.espacio, self.muslo_de, self.espinilla_de,
            LIMITE_ESPINILLA_MIN, LIMITE_ESPINILLA_MAX
        )

        # --- Motores ---
        self.motor_muslo_iz    = _crear_motor(self.espacio, self.torso,    self.muslo_iz)
        self.motor_espinilla_iz = _crear_motor(self.espacio, self.muslo_iz, self.espinilla_iz)
        self.motor_muslo_de    = _crear_motor(self.espacio, self.torso,    self.muslo_de)
        self.motor_espinilla_de = _crear_motor(self.espacio, self.muslo_de, self.espinilla_de)

    # -------------------------------------------------------------------------
    # Bucle de actualización
    # -------------------------------------------------------------------------

    def actualizar(self) -> None:
        """
        Un tick de simulación:
        1. Leer los 10 sensores del cuerpo.
        2. Pasar por la red neuronal → 4 salidas.
        3. Aplicar las salidas como velocidades de motor.
        4. Calcular el fitness actual.
        5. Comprobar si la criatura ha caído o se ha volcado.
        """
        if not self._viva:
            return

        # 1. Leer sensores
        entradas = self._leer_sensores()

        # 2. Inferencia de la red neuronal
        salidas = self.red.predecir(entradas)  # shape (4,), valores en [-1, 1]

        # 3. Aplicar velocidades a los motores
        self.motor_muslo_iz.rate     = float(salidas[0]) * MOTOR_RATE_MAX
        self.motor_espinilla_iz.rate = float(salidas[1]) * MOTOR_RATE_MAX
        self.motor_muslo_de.rate     = float(salidas[2]) * MOTOR_RATE_MAX
        self.motor_espinilla_de.rate = float(salidas[3]) * MOTOR_RATE_MAX

        # 4. Calcular fitness (distancia horizontal desde el inicio)
        self.fitness = self.torso.position.x - X_INICIO_CRIATURA

        # 5. Comprobar condiciones de muerte
        if self.torso.position.y > Y_SUELO - 10:
            # El torso ha tocado (casi) el suelo → caída
            self._viva = False
        elif abs(self.torso.angle) > math.pi / 2:
            # Inclinación excesiva (volcado)
            self._viva = False

    def _leer_sensores(self) -> np.ndarray:
        """
        Construye el vector de 10 entradas para la red neuronal.
        Todos los valores se limitan a [-1, 1] con np.clip.
        """
        t  = self.torso
        mi = self.muslo_iz
        me = self.muslo_de
        ei = self.espinilla_iz
        ed = self.espinilla_de

        entradas = [
            t.angle  / math.pi,                         # 1: inclinación torso
            t.angular_velocity / 10.0,                  # 2: vel. angular torso
            (mi.angle - t.angle)  / math.pi,            # 3: ángulo muslo izq relativo
            mi.angular_velocity   / 10.0,               # 4: vel. angular muslo izq
            (ei.angle - mi.angle) / math.pi,            # 5: ángulo espinilla izq relativo
            ei.angular_velocity   / 10.0,               # 6: vel. espinilla izq
            (me.angle - t.angle)  / math.pi,            # 7: ángulo muslo der relativo
            me.angular_velocity   / 10.0,               # 8: vel. angular muslo der
            (ed.angle - me.angle) / math.pi,            # 9: ángulo espinilla der relativo
            ed.angular_velocity   / 10.0,               # 10: vel. espinilla der
        ]
        return np.clip(entradas, -1.0, 1.0)

    # -------------------------------------------------------------------------
    # Estado
    # -------------------------------------------------------------------------

    def esta_viva(self) -> bool:
        """True si la criatura sigue en pie, False si ha caído o volcado."""
        return self._viva

    def obtener_genoma(self) -> list[np.ndarray]:
        """Devuelve los pesos de la red neuronal (genoma para el GA)."""
        return self.red.obtener_genoma()

    # -------------------------------------------------------------------------
    # Renderizado
    # -------------------------------------------------------------------------

    def dibujar(self, pantalla: pygame.Surface, offset_x: float = 0.0) -> None:
        """
        Dibuja la criatura en pantalla aplicando offset_x de cámara.
        - Cada segmento: rectángulo rotado con pygame.transform.rotate
        - Articulaciones: círculos color CONFIG.COLOR_JOINT
        - Muerta: todo en COLOR_MUERTO

        Args:
            pantalla:  superficie pygame donde dibujar
            offset_x:  desplazamiento horizontal de la cámara (píxeles)
        """
        muerta = not self._viva

        # Determinar colores base
        if muerta:
            color_torso      = COLOR_MUERTO
            color_muslo      = COLOR_MUERTO
            color_espinilla  = COLOR_MUERTO
        else:
            color_torso     = self.color if self.color else COLOR_TORSO
            color_muslo     = COLOR_PATA_ARRIBA
            color_espinilla = COLOR_PATA_ABAJO

        # Dibujar los segmentos
        self._dibujar_rect(pantalla, self.espinilla_iz, ESPINILLA_ANCHO, ESPINILLA_ALTO,
                           color_espinilla, offset_x)
        self._dibujar_rect(pantalla, self.espinilla_de, ESPINILLA_ANCHO, ESPINILLA_ALTO,
                           color_espinilla, offset_x)
        self._dibujar_rect(pantalla, self.muslo_iz, MUSLO_ANCHO, MUSLO_ALTO,
                           color_muslo, offset_x)
        self._dibujar_rect(pantalla, self.muslo_de, MUSLO_ANCHO, MUSLO_ALTO,
                           color_muslo, offset_x)
        self._dibujar_rect(pantalla, self.torso, TORSO_ANCHO, TORSO_ALTO,
                           color_torso, offset_x)

        # Dibujar articulaciones (joints) como círculos
        if not muerta:
            joints = [
                # (cuerpo_superior, cuerpo_inferior, altura relativa en superior)
                (self.torso,    self.muslo_iz,    TORSO_ALTO / 2.0),
                (self.torso,    self.muslo_de,    TORSO_ALTO / 2.0),
                (self.muslo_iz, self.espinilla_iz, MUSLO_ALTO / 2.0),
                (self.muslo_de, self.espinilla_de, MUSLO_ALTO / 2.0),
            ]
            for cuerpo_sup, cuerpo_inf, _ in joints:
                # El punto del joint es donde el cuerpo inferior sale del superior
                # (aproximado: posición del cuerpo inferior)
                px = int(cuerpo_inf.position.x + offset_x)
                py = int(cuerpo_inf.position.y - MUSLO_ALTO / 2.0)
                if 0 < px < pantalla.get_width():
                    pygame.draw.circle(pantalla, COLOR_JOINT, (px, py), 5)

    def _dibujar_rect(
        self,
        pantalla: pygame.Surface,
        cuerpo: pymunk.Body,
        ancho: int,
        alto: int,
        color: tuple,
        offset_x: float,
    ) -> None:
        """
        Dibuja un rectángulo rotado según el ángulo del body pymunk.
        Usa pygame.transform.rotate para la rotación.
        """
        # Posición en pantalla (aplicar offset de cámara)
        px = int(cuerpo.position.x + offset_x)
        py = int(cuerpo.position.y)

        # Culling: no dibujar si está completamente fuera de pantalla
        ancho_pantalla = pantalla.get_width()
        if px < -ancho or px > ancho_pantalla + ancho:
            return

        # Crear la superficie del rectángulo
        surf = pygame.Surface((ancho, alto), pygame.SRCALPHA)
        surf.fill(color)

        # Borde fino para definición visual
        pygame.draw.rect(surf, _oscurecer(color, 40), surf.get_rect(), 1)

        # Rotar (pymunk: ángulo positivo = sentido antihorario; pygame: horario)
        angulo_grados = -math.degrees(cuerpo.angle)
        surf_rot = pygame.transform.rotate(surf, angulo_grados)

        # Centrar en la posición del body
        rect = surf_rot.get_rect(center=(px, py))
        pantalla.blit(surf_rot, rect)


# -------------------------------------------------------------------------
# Utilidad de color
# -------------------------------------------------------------------------

def _oscurecer(color: tuple, cantidad: int) -> tuple:
    """Devuelve un color más oscuro en cantidad por canal."""
    return tuple(max(0, c - cantidad) for c in color[:3])
