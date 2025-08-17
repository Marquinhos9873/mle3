import mlflow
from datetime import datetime
import os


class PREPROCESADOR():
    pass

class PROCESO():
    pass

class GUARDADOMLFLOW():
    pass

class 

''' ----------------------------------------------------------------------- '''

def crear_experimento_mlflow(ruta_contador="contador_experimentos.txt"):
    if os.path.exists(ruta_contador):
        with open(ruta_contador, "r") as f:
            ultimo_numero = int(f.read().strip())
    else:
        ultimo_numero = 0
    nuevo_numero = ultimo_numero + 1
    with open(ruta_contador, "w") as f:
        f.write(str(nuevo_numero))
  
    fecha_actual = datetime.now().strftime("%d/%m/%y")
    nombre_experimento = f"Experimento - {nuevo_numero}, {fecha_actual}"
    mlflow.create_experiment(nombre_experimento)

    return exp_name

def decomposicion_series(data):
    decomposition = seasonal_decompose(
    air_passengers['#Passengers'], 
    model='aditive', 
    period=12
    )
     