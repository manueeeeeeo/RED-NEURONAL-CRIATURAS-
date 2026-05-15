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

# MOTORES #
MOTOR_FUERZA_MAX  = 800000   # fuerza máxima que aplican los motores
MOTOR_RATE_MAX    = 8.0      # velocidad máxima del motor (rad/s)

# POSICIÓN INICIAL DE LA CRIATURA #
X_INICIO_CRIATURA = 200      # posición X de inicio (lejos del borde izq)

# LÍMITES ANGULARES DE LAS ARTICULACIONES #
LIMITE_MUSLO_RAD      = 1.22   # ±70° en radianes (math.pi * 70/180)
LIMITE_ESPINILLA_MIN  = 0.0    # 0° (espinilla no va hacia arriba)
LIMITE_ESPINILLA_MAX  = 2.27   # 130° en radianes

# RED NEURONAL #
ENTRADAS      = 10       # sensores de la criatura
CAPAS_OCULTAS = [16, 8]  # capas ocultas
SALIDAS       = 4        # un motor por articulación

# ALGORITMO GENÉTICO #
TAMANO_POBLACION      = 20     # número de criaturas por generación

TASA_MUTACION         = 0.18   # probabilidad inicial de mutar cada peso
TASA_MUTACION_MIN     = 0.04   # mínimo de tasa de mutación
TASA_MUTACION_DECAY   = 0.97   # factor de reducción por generación exitosa
MAGNITUD_MUTACION     = 0.40   # magnitud inicial de perturbación gaussiana
MAGNITUD_MUTACION_MIN = 0.06   # magnitud mínima

ELITISMO              = 3      # top-N que pasan sin cambios
PORCENTAJE_PADRES     = 0.50   # porcentaje de población que puede ser padre
PORCENTAJE_ALEATORIOS = 0.10   # porcentaje de individuos aleatorios por gen

# SIMULACIÓN #
DURACION_SIMULACION = 1200     # frames máximos por criatura (a 60 FPS = 20 s)
MAX_GENERACIONES    = 0        # 0 = sin límite

# VISUALIZACIÓN #
GRAFICA_GENS    = 60           # generaciones visibles en la gráfica
MOSTRAR_GRAFICA = True