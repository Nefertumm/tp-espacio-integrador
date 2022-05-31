import pandas as pd

def desarmarFecha(fecha: str):
    fechaDesarmada = tuple(fecha.split("-"))
    return fechaDesarmada

def imputarPeso(row):
    if row["peso"] < 0:
        return row["peso"] * -1
    return row["peso"]

def imputarMontoOperacion(row):
    if row["monto_operacion"] < 0:
        return row["monto_operacion"] * -1
    return row["monto_operacion"]

def loadDatabase():
    db = pd.read_csv('Exportaciones.csv')
    
    # Columna "peso"
    # Eliminación inicial -> Por condiciones de la introducción
    db = db.loc[(db["peso"] <= 300)]
    
    # Imputamos peso negativo simplemente multiplicando por -1
    db["peso"] = db.apply(imputarPeso, axis=1)
    
    # Columna "monto_operacion"
    # Eliminación inicial -> Por condiciones de la introducción
    db = db.loc[(db["monto_operacion"] <= 15000)]
    # Eliminación con montos iguales a 0
    db = db.loc[(db["monto_operacion"] != 0)]
    
    # Imputamos monto_operacion negativo multiplicando por -1
    db["monto_operacion"] = db.apply(imputarMontoOperacion, axis=1)
    
    # Columna "tiempo_envío"
    # Se eliminarán registros donde el tiempo sea negativo o 0.
    db = db.loc[db["tiempo_envío"] > 0]
    
    # Columna "pais_destino"
    db = db.dropna(subset=['pais_destino'])
    
    # Columna "origen_especifico"
    db = db.dropna(subset=['origen_especifico'])
    
    # Columna "provincia"
    db = db.dropna(subset=['provincia'])
    
    # Boleteamos envio_gratis
    db = db.drop(columns = ["envio_gratis"])
    
    print(db)
   
loadDatabase()