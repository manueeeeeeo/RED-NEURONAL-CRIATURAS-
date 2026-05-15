# =============================================================================
# simulation.py — Envoltorio del espacio físico pymunk
# Gestiona la gravedad, el suelo y el ciclo de pasos de simulación.
# No dibuja nada: es pura física.
# =============================================================================

import pymunk
from config import GRAVEDAD, FRICCION_SUELO, Y_SUELO, ANCHO_VENTANA


class Simulacion:
    """
    Envuelve un pymunk.Space con configuración fija del mundo:
      - Gravedad global desde config.GRAVEDAD
      - Suelo estático horizontal en config.Y_SUELO con alta fricción

    Uso típico:
        sim = Simulacion()
        criatura = Criatura(sim.espacio, genoma)
        for _ in range(DURACION_SIMULACION):
            sim.paso(1/60)
            criatura.actualizar()
        sim.resetear()
    """

    def __init__(self):
        # Crear el espacio físico y configurar gravedad
        self.espacio = pymunk.Space()
        self.espacio.gravity = GRAVEDAD

        # Amortiguación global para evitar oscilaciones infinitas
        self.espacio.damping = 0.95

        # Construir el suelo estático
        self._crear_suelo()

    # -------------------------------------------------------------------------
    # Creación del suelo
    # -------------------------------------------------------------------------

    def _crear_suelo(self):
        """
        Crea un segmento estático (línea horizontal) en Y_SUELO.
        Se extiende muy a la izquierda y muy a la derecha para que la cámara
        pueda desplazarse sin que la criatura caiga al vacío.
        """
        cuerpo_suelo = pymunk.Body(body_type=pymunk.Body.STATIC)
        # El suelo va desde -5000 hasta 50000 px para dar margen de cámara
        self.suelo_seg = pymunk.Segment(
            cuerpo_suelo,
            (-5000, Y_SUELO),
            (50000, Y_SUELO),
            5  # grosor
        )
        self.suelo_seg.friction     = FRICCION_SUELO
        self.suelo_seg.elasticity   = 0.0   # sin rebote
        self.suelo_seg.filter       = pymunk.ShapeFilter(categories=0b01)

        self.espacio.add(cuerpo_suelo, self.suelo_seg)

    # -------------------------------------------------------------------------
    # Paso de simulación
    # -------------------------------------------------------------------------

    def paso(self, dt: float = 1 / 60):
        """
        Avanza la simulación física un instante de tiempo dt (en segundos).
        Se recomienda dt = 1/60 para 60 FPS.
        Subdivide el paso en varios sub-pasos para mayor estabilidad.
        """
        self.espacio.step(dt)

    # -------------------------------------------------------------------------
    # Resetear el espacio para la siguiente criatura
    # -------------------------------------------------------------------------

    def resetear(self):
        """
        Elimina todos los cuerpos y formas dinámicos del espacio
        (la criatura anterior) conservando el suelo estático.
        Prepara el espacio para añadir la siguiente criatura.
        """
        # Recopilar todos los cuerpos que NO son estáticos (son de la criatura)
        cuerpos_a_eliminar = [
            b for b in self.espacio.bodies
            if b.body_type == pymunk.Body.DYNAMIC
        ]

        for cuerpo in cuerpos_a_eliminar:
            # Eliminar todas las formas asociadas al cuerpo
            # Usamos espacio.shapes para encontrar las que pertenecen a este body
            formas_del_cuerpo = [s for s in list(self.espacio.shapes) if s.body is cuerpo]
            for forma in formas_del_cuerpo:
                self.espacio.remove(forma)
            self.espacio.remove(cuerpo)

        # Eliminar restricciones (joints, motores, etc.) huérfanas
        restricciones_a_eliminar = list(self.espacio.constraints)
        for r in restricciones_a_eliminar:
            self.espacio.remove(r)
