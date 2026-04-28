# Está clase nos va a ser servir para establecer
# todos los parametros del proyecto

# VENTANA #
ANCHO_VENTANA = 1200
ALTO_VENTANA = 700
FPS = 60
TITULO = "Red Neuronal - Criaturas"

# COLORES #
COLOR_FONDO        = (20, 20, 35)
COLOR_SUELO        = (60, 70, 80)
COLOR_TORSO        = (80, 140, 200)
COLOR_PATA_ARRIBA  = (60, 110, 170)   # muslo
COLOR_PATA_ABAJO   = (40,  80, 140)   # espinilla
COLOR_JOINT        = (220, 100, 50)   # punto de articulación
COLOR_MEJOR        = (50, 220, 120)   # color del mejor individuo
COLOR_MUERTO       = (50,  50,  65)
COLOR_HUD          = (220, 220, 240)
COLOR_ACENTO       = (50,  220, 120)

#FÍSICA #
GRAVEDAD = (0,1200) # cuanto pesa el mundo ( Y positivo, abajo)
FRICCION_SUELO = 2.0 # rozamiento del suelo ( más = más agarre )
FRICCION_CUERPO = 0.8 # rozamiento del cuerpo de la criatura
Y_SUELO = 580 # posición vertical del suelo en la pantall

# ANATOMÍA DE LA CRIATURA #
# El cuerpo tiene esta forma:
#
#     [   TORSO   ]          ← rectángulo central
#       /       \
#  [MUSLO]   [MUSLO]        ← pata izquierda y derecha (parte 1)
#     |           |
# [ESPINILLA] [ESPINILLA]   ← parte inferior de cada pata (parte 2)
#
TORSO_ANCHO   = 60    # anchura del torso en píxeles
TORSO_ALTO    = 25    # altura del torso
TORSO_MASA    = 8     # masa del torso (más masa = más estable)

MUSLO_ANCHO   = 12    # anchura del muslo
MUSLO_ALTO    = 40    # longitud del muslo
MUSLO_MASA    = 1.5

ESPINILLA_ANCHO = 10  # anchura de la espinilla
ESPINILLA_ALTO  = 38  # longitud de la espinilla
ESPINILLA_MASA  = 1.0