# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:57:38 2020

@author: enriq
"""

import pandas as pd
import numpy as np
import os 
import sys
import seaborn as sns

test = pd.read_csv(r"./data/Base1_test.csv")

dateparse = lambda x: pd.to_datetime(x)

# Carga datos
base_1 = pd.read_csv(r"./data/Base1_train.csv", parse_dates = ["MES_COTIZACION"], date_parser = dateparse) # Base cotizaciones
base_2 = pd.read_csv(r"./data/Base2.csv", sep = ";", parse_dates = ["MES_COTIZACION"], date_parser = dateparse) # Información sociodemográfica + digital
base_3 = pd.read_csv(r"./data/Base3.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base productos BBVA
base_4 = pd.read_csv(r"./data/Base4.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base de saldos en el Sistema Financiero
base_5 = pd.read_csv(r"./data/Base5.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base de consumos con tarjeta

# Exploratory Data Analysis
## Base_1

#Cantidad de clientes unicos / longitud del df
len(base_1.COD_CLIENTE.unique()) / base_1.shape[0]
# PLT no son únicos los clientes

## Base_2
#Cantidad de clientes unicos / longitud del df
len(base_2.COD_CLIENTE.unique())/base_2.shape[0]
# PLT no son únicos los clientes

## Base_3
#Cantidad de clientes unicos / longitud del df
len(base_3.COD_CLIENTE.unique())/base_3.shape[0]
# PLT son únicos los clientes

## Base_4
#Cantidad de clientes unicos / longitud del df
len(base_4.COD_CLIENTE.unique())/base_4.shape[0]
# PLT no son únicos los clientes

## Base_5
# Cantidad de clientes unicos / longitud del df
len(base_5.COD_CLIENTE.unique())/base_5.shape[0]
# PLT son únicos los clientes

# En ningún df hay codigos de clientes nulos
base_1.COD_CLIENTE.isnull().sum()
base_2.COD_CLIENTE.isnull().sum()
base_3.COD_CLIENTE.isnull().sum()
base_4.COD_CLIENTE.isnull().sum()
base_5.COD_CLIENTE.isnull().sum()


def joinColumns(df1, df2):
    df = df1.merge(df2, on = "COD_CLIENTE", how = "left")
    df = df.loc[(df["MES_COTIZACION_y"] <= df["MES_COTIZACION_x"]) |
                (df["MES_COTIZACION_y"].isnull()), :]
    df = df.sort_values("MES_COTIZACION_y", ascending = False)
    df = df.drop_duplicates(["COD_CLIENTE", "COD_SOL"], keep = "first")
    df = df.drop("MES_COTIZACION_y", axis = 1)
    df = df.rename(columns = {"MES_COTIZACION_x": "MES_COTIZACION"})
    return df

train = joinColumns(base_1, base_2)

def joinColumns2(df1, df2):

    df = df1.merge(df2, on = "COD_CLIENTE", how = "left", indicator = True)
    
    dfx = df.loc[df["MES_COTIZACION_y"] <= df["MES_COTIZACION_x"], :]
    dfy = df[df['_merge'] == 'left_only']
    dfz = df.loc[df["MES_COTIZACION_y"] > df["MES_COTIZACION_x"], :]
    for col in df2.columns:
        if col != "COD_CLIENTE":
            dfz[col] = np.nan

    df = pd.concat([dfx, dfy, dfz]).sort_values("MES_DATA", ascending = False)    
    df = df.drop_duplicates(["COD_CLIENTE", "COD_SOL"], keep = "first")
    df = df.drop("MES_COTIZACION_y", axis = 1)
    df = df.rename(columns = {"MES_COTIZACION_x": "MES_COTIZACION"})

    return df

train = joinColumns2(train, base_3)

# CODIGO JULIO

train = joinColumns(train, base_4)

# CODIGO URI

train = joinColumns(train, base_5)

# AQUÍ CÓDIGO QUIQUE

a = train.corr()
for col in a.columns:
    if abs(a[col]) >= 0.9:
        