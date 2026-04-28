import pygame
import pymunk
import math
import pymunk.pygame_util

pygame.init()
pantalla = pygame.display.set_mode((800,600))
pygame.display.set_caption("Prueba articulaciones")
reloj = pygame.time.Clock()

# Espacio Físico
espacio = pymunk.Space()
espacio.gravity = (0,900)

# Suelo
suelo_cuerpo = pymunk.Body(body_type=pymunk.Body.STATIC)
suelo = pymunk.Segment(suelo_cuerpo, (0,500), (800, 500), 5)
suelo.friction = 1.0
espacio.add(suelo_cuerpo, suelo)

# Torso
# Es el cuerpo principal de la criatura
# Masa alta para que sea estable
torso_cuerplo = pymunk.Body(5, pymunk.moment_for_box(5,(60,30)))
torso_cuerplo.position = (400,400)
torso_forma = pymunk.Poly.create_box(torso_cuerplo, (60,30))
torso_forma.friction = 0.8
espacio.add(torso_cuerplo, torso_forma)

# Pata
# Un segmento más ligero, colgando del torso
pata_cuerpo = pymunk.Body(1, pymunk.moment_for_box(1,(10,50)))
pata_cuerpo.position = (400, 400)
pata_forma = pymunk.Poly.create_box(pata_cuerpo, (10,50))
pata_forma.friction = 1.0
espacio.add(pata_cuerpo, pata_forma)

# Articulación (PivotJoint)
#Une el torso con la para en un punto concreto
# El punto de unión está en el borde inferior del torso / superior de la pata
punto_union = (400,415)
joint = pymunk.PivotJoint(torso_cuerplo, pata_cuerpo, punto_union)
espacio.add(joint)

# Motor
# El motor hace girar la articulación a una velocidad angular.
# rate = velocidad en radianes/segundo (positivo = sentido horario)
# max_force = fuerza máxima que aplica el motor
motor = pymunk.SimpleMotor(torso_cuerplo, pata_cuerpo, rate=2.0)
motor.max_force = 50000
espacio.add(motor)

# LÍMITE DE ÁNGULO
# Opcional: limitar cuánto puede girar la articulación.
# Aquí lo dejamos libre para que veas el efecto completo.

opciones_dibujo = pymunk.pygame_util.DrawOptions(pantalla)
fuente = pygame.font.SysFont("consolas", 16)

# ── BUCLE ─────────────────────────────────────────────────────────────────────
corriendo = True
while corriendo:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            corriendo = False
        # Pulsa ESPACIO para invertir el motor (girar al otro lado)
        if evento.type == pygame.KEYDOWN and evento.key == pygame.K_SPACE:
            motor.rate = -motor.rate

    espacio.step(1 / 60)

    pantalla.fill((20, 20, 30))
    espacio.debug_draw(opciones_dibujo)

    # Mostrar info
    txt = fuente.render(f"Motor: {motor.rate:.1f} rad/s  |  ESPACIO = invertir", True, (200, 200, 200))
    pantalla.blit(txt, (10, 10))

    pygame.display.flip()
    reloj.tick(60)

pygame.quit()