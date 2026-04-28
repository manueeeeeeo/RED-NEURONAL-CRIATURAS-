import pygame
import pymunk
import pymunk.pygame_util # herramietnas para dibujar en pymunk con pygame

# -- 1. Iniciar PYGAME
pygame.init()
pantalla = pygame.display.set_mode((800,600))
pygame.display.set_caption("Prueba pymunk")
reloj = pygame.time.Clock()

# -- 2. CREAR EL ESPACIO FISICO
#En pymunk todo ocurre dentro de un "Space" (Espacio)
# El Space gestiona toda la gravedad y todos los objetos fisicos
espacio = pymunk.Space()
espacio.gravity = (0,900) # gravedad hacia abajo (eje Y positivo = abajo en pymunk)

# -- 3. CREAR EL SUELO
#Un objeto estático (mass=0) no se mueve aunque lo golpeen
# Se define con dos puntos (segmento línea)
cuerpo_suelo = pymunk.Body(body_type=pymunk.Body.STATIC)
suelo = pymunk.Segment(cuerpo_suelo, (0,550), (800, 550), 5)
#grosor = 5
suelo.friction = 1.0 #rozamiento a lto para que los objetos no resvalen
espacio.add(cuerpo_suelo, suelo)

# -- 4. CREAR UNA CAJA QUE CAE
#un body dinámico SI se mueve por las fuerzas
# Necesita masa e inercia (momento de incercia = resistencia a girar)
masa = 1
inercia = pymunk.moment_for_box(masa,(50, 50)) # pymunk calcula la inercia para una caja
cuerpo = pymunk.Body(masa, inercia)
cuerpo.position = (400,100) # posición inicial

# La Shape define la FORMA FÍSICA del objeto (para colisiones)
# Aquí es un rectángulo de 50x50
caja = pymunk.Poly.create_box(cuerpo, (50,50))
caja.friction = 0.8
espacio.add(cuerpo, caja)

# -- 5. HERRAMIENTA DE DIBUJO
# pygame_util.DrawOptions traduce los objetos pymunk a dibujos de pygame.
opciones_dibujo = pymunk.pygame_util.DrawOptions(pantalla)

# -- 6. BUCLE PRINCIPAL
corriendo = True
while corriendo:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            corriendo = False

    # PASO DE SIMULACIÓN: avanzar la física un instante de tiempo.
    # 1/60 segundos = un frame a 60 FPS.
    espacio.step(1 / 60)

    # Dibujar
    pantalla.fill((30, 30, 30))
    espacio.debug_draw(opciones_dibujo)   # dibuja todos los objetos del espacio
    pygame.display.flip()
    reloj.tick(60)
pygame.quit()