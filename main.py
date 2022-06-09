import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import rcParams
import numpy as np
import math

def imputarPeso(row):
    if row['peso'] < 0:
        return row['peso'] * -1
    return row['peso']

def imputarMontoOperacion(row):
    if row['monto_operacion'] < 0:
        return row['monto_operacion'] * -1
    return row['monto_operacion']

def calcularDesvioYMediaColumna(column: str, db):
    columnList = db[column].tolist()
    media = sum(columnList) / len(columnList)
    aux = 0
    for t in columnList:
        aux += (t - media) ** 2
    return ((aux / (len(columnList))) ** 0.5, media)

def min_max(lista: list):
    maxi = max(lista)
    mini = min(lista)
    return (mini, maxi)

def loadDatabase():
    db = pd.read_csv('Exportaciones.csv')
    
    # Columna "peso"
    # Eliminación inicial -> Por condiciones de la introducción
    db = db.loc[(db['peso'] <= 300)]
    
    # Imputamos peso negativo simplemente multiplicando por -1
    db['peso'] = db.apply(imputarPeso, axis=1)
    
    # Columna "monto_operacion"
    # Eliminación inicial -> Por condiciones de la introducción
    db = db.loc[(db['monto_operacion'] <= 15000)]
    # Eliminación con montos iguales a 0
    db = db.loc[(db['monto_operacion'] != 0)]
    
    # Imputamos monto_operacion negativo multiplicando por -1
    db['monto_operacion'] = db.apply(imputarMontoOperacion, axis=1)
    
    # Columna "tiempo_envío"
    # Se eliminarán registros donde el tiempo sea negativo o 0.
    db = db.loc[db['tiempo_envío'] > 0]
    
    # Columna "pais_destino"
    db = db.dropna(subset=['pais_destino'])
    
    # Columna "origen_especifico"
    db = db.dropna(subset=['origen_especifico'])
    
    # Columna "provincia"
    db = db.dropna(subset=['provincia'])
    
    # Boleteamos envio_gratis
    db = db.drop(columns = ['envio_gratis'])
    
    # Convertimos fecha_creacion a datetime de pandas para trabajar mejor.
    db['fecha_creacion'] = pd.to_datetime(db['fecha_creacion'])
    
    return db

def normalizar_tiempo_envio(db):
    desvio, media = calcularDesvioYMediaColumna("tiempo_envío", db)
    db = db.assign(tiempo_envío_norm = lambda x: (x.tiempo_envío - media) / desvio )
    return db

def randomColorList(amount: int):
    return ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(amount)]

def pie_chart_paises(db):
    paisDict = {}
    paisDict = defaultdict(lambda:0, paisDict)
    for x in db.index:
        paisDict[db['pais_destino'][x]] += 1
        
    to_remove = []
    paisDictOr = paisDict.copy()
    paisDict['Otros'] = 0
    for idx, x in paisDictOr.items():
        if x <= 40:
            paisDict['Otros'] += x
            to_remove.append(idx)
    for t in to_remove:
        paisDict.pop(t)

    pctgs = {}
    aux = 0
    for idx, x in paisDict.items():
        aux += x

    for idx, x in paisDict.items():
        pctgs[idx] = np.round(x * 100 / aux, 2)

    rcParams['figure.dpi'] = 150
    rcParams['figure.figsize'] = (6, 9)
    fig1, ax1 = plt.subplots()
    colors = randomColorList(len(paisDict))
    paisDict = sorted(paisDict.items(), key=lambda k_v: k_v[1], reverse=True)
    l = ax1.pie([x[1] for x in paisDict], startangle=90, colors=colors)
    for label, t in zip([x[0] for x in paisDict], l[1]):
        x, y = t.get_position()
        angle = int(math.degrees(math.atan2(y, x)))
        ha = "left"

        if x<0:
            angle -= 180
            ha = "right"

        plt.annotate(label+ ": " + str(pctgs[label]) + "%", xy=(x,y), rotation=angle, ha=ha, va="center", rotation_mode="anchor", size=8)

    ax1.set_title('Gráfico de torta de los países destinos')
    ax1.axis('equal')

    plt.savefig('pie.png', bbox_inches='tight')

def tiempo_envio_pais_destino(db):
    paisDict = {}
    paisDict = defaultdict(list)
    for x in db.index:
        paisDict[db['pais_destino'][x]].append(db['tiempo_envío'][x])
    
    to_remove = []
    for idx, x in paisDict.items():
        if len(x) <= 40:
            to_remove.append(idx)
    for x in to_remove:
        paisDict.pop(x)
    return paisDict

def media_tiempo_de_envio_por_pais(paisDict: defaultdict):
    mediaPaises = {}
    for pais, l in paisDict.items():
        mediaPaises[pais] = sum(l) / len(l)
    return mediaPaises

def findOrigen(paisDict, provincia, localidad):
    for ind, y in enumerate(paisDict[provincia]):
        if y[0] == localidad:
            return ind
    return -1

def provincia_origen_especifico(db):
    paisDict = {}
    paisDict = defaultdict(list)
    total = 0
    for x in db.index:
        provincia = db['provincia'][x]
        localidad = db['origen_especifico'][x]
        index = findOrigen(paisDict, provincia, localidad)
        if index != -1:
            paisDict[provincia][index][1] += 1
        else:
            paisDict[provincia].append([localidad, 1])
        total += 1
    return (paisDict, total)

def cantPorProvincia(paisDict):
    listaProvincia = []
    for idx, x in paisDict.items():
        aux = 0
        for y in x:
            aux += y[1]
        listaProvincia.append((idx, aux))
    return listaProvincia

db = loadDatabase()

desvio, media = calcularDesvioYMediaColumna("monto_operacion", db)
print(f'Monto operacion: Media {media} - Desvio: {desvio}')

db = normalizar_tiempo_envio(db)

intervalo_fecha = db.loc[(db['fecha_creacion'].dt.month == 12) | (db['fecha_creacion'].dt.month == 1) | (db['fecha_creacion'].dt.month == 2)]
listIntervalo = intervalo_fecha['monto_operacion'].tolist()
print(min_max(listIntervalo))

anio_2017 = db.loc[(db['fecha_creacion'].dt.year == 2017)]
list2017 = anio_2017['monto_operacion'].tolist()
print(f'2017: {min_max(list2017)}')

anio_2018 = db.loc[(db['fecha_creacion'].dt.year == 2018)]
list2018 = anio_2018['monto_operacion'].tolist()
print(f'2018: {min_max(list2018)}')

anio_2019 = db.loc[(db['fecha_creacion'].dt.year == 2019)]
list2019 = anio_2019['monto_operacion'].tolist()
print(f'2019: {min_max(list2019)}')

# print(db.info())
desvioT, mediaT = calcularDesvioYMediaColumna("tiempo_envío", db)
print(f'Tiempo de envío: Media {mediaT} - Desvio: {desvioT}')

paisDict = tiempo_envio_pais_destino(db)
mediaPaises = media_tiempo_de_envio_por_pais(paisDict)

# for idx, x in mediaPaises.items():
#     print(f'{idx}: {x} \n')

for idx, x in paisDict.items():
    mini, maxi = min_max(x)
    print(f'{idx}: Min: {mini} - Max: {maxi}')
    
paisDict2, totalDict = provincia_origen_especifico(db)
listaProvincia = cantPorProvincia(paisDict2)
for x in listaProvincia:
    print(f'{x[0]}: {x[1]}')

# db.to_csv('Exportaciones_d.csv')