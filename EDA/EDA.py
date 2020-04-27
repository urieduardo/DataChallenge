# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:57:38 2020

@author: enriq
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import sys
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer

test = pd.read_csv(r"./data/Base1_test.csv")

dateparse = lambda x: pd.to_datetime(x)

# Carga datos
base_1 = pd.read_csv(r"./data/Base1_train.csv", parse_dates = ["MES_COTIZACION"], date_parser = dateparse) # Base cotizaciones
base_2 = pd.read_csv(r"./data/Base2.csv", sep = ";", parse_dates = ["MES_COTIZACION"], date_parser = dateparse) # Información sociodemográfica + digital
base_3 = pd.read_csv(r"./data/Base3.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base productos BBVA
base_4 = pd.read_csv(r"./data/Base4.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base de saldos en el Sistema Financiero
base_5 = pd.read_csv(r"./data/Base5.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base de consumos con tarjeta

# casting de predictores categóricos
base_4["ST_CREDITO"] = base_4["ST_CREDITO"].astype("category")
base_5[["CD_DIVISA", "TP_TARJETA"]] = base_5[["CD_DIVISA", "TP_TARJETA"]].astype("category")

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
    dfy = df.loc[df['_merge'] == 'left_only', :]
    dfz = df.loc[df["MES_COTIZACION_y"] > df["MES_COTIZACION_x"], df.columns[:len(df1.columns)]]   
    # print(dfx.dtypes)
    # print(dfy.dtypes)
    # print(dfz.dtypes)
    df = pd.concat([dfx, dfy, dfz], sort = False).drop("_merge", axis = 1)
    df = df.sort_values("MES_DATA", ascending = False)    
    df = df.drop_duplicates(["COD_CLIENTE", "COD_SOL"], keep = "first")
    df = df.drop("MES_COTIZACION_y", axis = 1)
    df = df.rename(columns = {"MES_COTIZACION_x": "MES_COTIZACION"})

    return df

train = joinColumns2(train, base_3)

# CODIGO JULIO

def joinColumns3(df1, df2):

    df = df1.merge(df2, on = "COD_CLIENTE", how = "left", indicator = True)
    dfx = df.loc[df["MES_COTIZACION_y"] <= df["MES_COTIZACION_x"], :]
    dfy = df.loc[df['_merge'] == 'left_only', :]
    dfz = df.loc[df["MES_COTIZACION_y"] > df["MES_COTIZACION_x"], df.columns[:len(df1.columns)]]
    df = pd.concat([dfx, dfy, dfz], sort = False).drop("_merge", axis = 1)   

    df.columns = df1.columns.to_list() + df.columns[len(df1.columns):].to_list()
    
    # separación entre predictores numéricos y categóricos
    nume = df2.columns[df2.dtypes == "int64"]
    cate = df.columns[df2.dtypes == "category"]
    df_num = df[df.columns.difference(cate)]
    df_cate = df[df.columns.difference(nume)]
    
    df_gr_m_num = df_num.groupby(df1.columns.to_list(), as_index = False).mean()
    df_gr_m_cate = df_cate.groupby(df1.columns.to_list(), as_index = False).count()

    df_gr_m = df.groupby(df1.columns.to_list(), as_index = False).mean()

    df1 = df1.merge(df_gr_m, on = df1.columns.to_list(), how = "left")
    
    return df1

    # df = df.drop(["MES_COTIZACION_y", "MES_DATA_y"], axis = 1)
    # df = df.rename(columns = {"MES_COTIZACION_x" : "MES_COTIZACION", "MES_DATA_x" : "MES_DATA"})

train  = joinColumns3(train, base_4)

# CODIGO URI

def joinColumns3(df1, df2):

   df = df1.merge(df2, on = "COD_CLIENTE", how = "left", indicator = True)
   
   dfx = df.loc[df["MES_COTIZACION_y"] <= df["MES_COTIZACION_x"], :]
   dfy = df[df['_merge'] == 'left_only']
   dfz = df.loc[df["MES_COTIZACION_y"] > df["MES_COTIZACION_x"], :]
   for col in df2.columns:
       if col != "COD_CLIENTE":
           dfz[col] = np.nan

   df = pd.concat([dfx, dfy, dfz]).sort_values("MES_DATA", ascending = False)  
   df = df.drop_duplicates(["COD_CLIENTE", "COD_SOL"], keep = "first")
   df = df.drop(["MES_COTIZACION_y", "MES_DATA_y", "_merge"], axis = 1)
   df = df.rename(columns = {"MES_COTIZACION_x": "MES_COTIZACION", "MES_DATA_x" : "MES_DATA"})

   return df

train = joinColumns(train, base_5)

# AQUÍ CÓDIGO QUIQUE

# Dividiendo columnas por tipo
bool_cols = [
"FLG_DESEMBOLSO",
"USO_BI_M0",
"USO_BI_M1",
"USO_BI_M2",
"USO_BM_M0",
"USO_BM_M1",
"USO_BM_M2"
]

# for col in bool_cols:
#     train[col] = train[col].astype(bool)

numeric_cols = train.dtypes[train.dtypes == np.float64].index.to_list() +\
    train.dtypes[train.dtypes == np.int64].index.to_list()

cat_cols = train.dtypes[train.dtypes == "O"].index[2:].to_list()

# Análisis Columnas numéricas

# Eliminando una de cada dos columnas numéricas correlacionadas
# (Para facilitar análisis)

corrmat = train[numeric_cols].corr()
droped_corr_cols = []

for col in corrmat.columns:
    if col not in droped_corr_cols:
        for colname in a.index:
            if colname in droped_corr_cols:
                pass
            else:
                print(colname + " se correlaciona mucho con " + col +
                      " por lo tanto nos deshacemos de ella por aportar\n la misma información")
                droped_corr_cols.append(colname)
                corrmat = corrmat.drop(colname, axis = 1)

# Graficando histogramas de columnas numéricas

for col in numeric_cols:
    train[col].plot.hist(title = col)
    s = train.describe()[col].to_string() + \
        "\nMissing Values: " + str(train.isnull().sum()[col]) + \
        "\nMissing Values %: " + str(round(train.isnull().sum()[col]/len(train),4))
    plt.figtext(1, 0.5, s)
    plt.show()

## Análisis Columnas categóricas

for col in cat_cols:
    train[col].value_counts().plot.bar(title = col, rot = 0)
    # s = train.describe()[col].to_string() + \
    #     "\nMissing Values: " + str(train.isnull().sum()[col]) + \
    #     "\nMissing Values %: " + str(round(train.isnull().sum()[col]/len(train),4))
    # plt.figtext(1, 0.5, s)
    plt.show()

imp = SimpleImputer(strategy="most_frequent")
imp.fit_transform(df)

# Chi Cuadrada

for col in cat_cols:
    label_encoder = LabelEncoder()
    train[col] = label_encoder.fit_transform(train[col])

label = 'FLG_DESEMBOLSO'

X = train[cat_cols + bool_cols].drop(label,axis=1)
y = train[label]

chi_scores = chi2(X,y)

p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)

p_values.plot.bar()