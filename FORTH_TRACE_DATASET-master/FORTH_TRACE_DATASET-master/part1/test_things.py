# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 20:20:45 2025

@author: Fábio Fernandes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.pyplot as plt


    
def recopilacionVectores():
    personaABuscar = input("¿De qué persona deseas la información?")
    dispositivoABuscar = int(input("¿De qué dispositivo deseas la información? (1-5) "))
        
        
    pX_1 = np.array(pd.read_csv(f"part{personaABuscar}dev{dispositivoABuscar}.csv"))
        
    # Extraer vectores
    acelerometroX = pX_1[:, 1]
    acelerometroY = pX_1[:, 2]
    acelerometroZ = pX_1[:, 3]
    giroscopioX   = pX_1[:, 4]
    giroscopioY   = pX_1[:, 5]
    giroscopioZ   = pX_1[:, 6]
    magnetometroX = pX_1[:, 7]
    magnetometroY = pX_1[:, 8]
    magnetometroZ = pX_1[:, 9]
        
    distanciaAcelerometro = np.sqrt(acelerometroX*2 + acelerometroY*2 + acelerometroZ*2)
    distanciaGiroscopio = np.sqrt(giroscopioX*2 + giroscopioY*2 + giroscopioZ*2)
    distanciaMagnetometro = np.sqrt(magnetometroX*2 + magnetometroY*2 + magnetometroZ*2)
        
        
    aceleromX_act1 = pX_1[pX_1[:,11]==1, 1]
    aceleromY_act1 = pX_1[pX_1[:,11]==1, 2]
    aceleromZ_act1 = pX_1[pX_1[:,11]==1, 3]
    distanciaAcelerometro_act1 = np.sqrt(aceleromX_act1*2 + aceleromY_act12 + aceleromZ_act1*2)
    
    aceleromX_act2 = pX_1[pX_1[:,11]==2, 4]
    aceleromY_act2 = pX_1[pX_1[:,11]==2, 5]
    aceleromZ_act2 = pX_1[pX_1[:,11]==2, 6]
    distanciaAcelerometro_act2 = np.sqrt(aceleromX_act2*2 + aceleromY_act22 + aceleromZ_act2*2)
        
    act2 = pX_1[pX_1[:,11]==2, 11]
    act3 = pX_1[pX_1[:,11]==3, 11]
    act4 = pX_1[pX_1[:,11]==4, 11]
    act5 = pX_1[pX_1[:,11]==5, 11]
    act6 = pX_1[pX_1[:,11]==6, 11]
    act7 = pX_1[pX_1[:,11]==7, 11]
    act8 = pX_1[pX_1[:,11]==8, 11]
    act9 = pX_1[pX_1[:,11]==9, 11]
    act10 = pX_1[pX_1[:,11]==10, 11]
    act11 = pX_1[pX_1[:,11]==11, 11]
    act12 = pX_1[pX_1[:,11]==12, 11]
    act13 = pX_1[pX_1[:,11]==13, 11]
    act14 = pX_1[pX_1[:,11]==14, 11]
    act15 = pX_1[pX_1[:,11]==15, 11]
    act16 = pX_1[pX_1[:,11]==16, 11]
    
    act2 = pX_1[pX_1[:,11]==2, 11]
    
    plt.boxplot([distanciaAcelerometro_act1,distanciaAcelerometro_act2])
    plt.xlabel('Actividades')
    plt.ylabel('Eje Y')
    plt.show()
        
    sns.boxplot(data=[distanciaAcelerometro_act1,distanciaAcelerometro_act2])
    plt.show
recopilacionVectores()