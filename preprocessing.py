#Importación de librerías
import pandas as pd
import dask as dd
import io
import numpy as np

from sklearn.preprocessing import LabelEncoder

#Load the required data
def read_data(mode):#name_file is the name of the file loaded, example: 'Data_Clientes.csv'
  #Cargar data
  if mode is 'test':
    from google.colab import files
    uploaded = files.upload()

    #Obtiene el nombre del archivo cargado
    for fn in uploaded.keys():
      filename=format(fn)
    
    print(filename)
    data = pd.read_csv(io.BytesIO(uploaded[filename]))#Lee el archivo cargado
    
  if mode is 'prueba':
    filename='/content/Data_Clientes.csv'
    data=pd.read_csv('/content/Data_Clientes.csv')

  return data

#Reemplaza los valores nan de la columna 'column_nan' por 0
def fill_nan_to_0(dataframe,list_to_fill):
  #Reemplaza los nan de la lista con 0
  for column in list_to_fill:
    dataframe[column]=dataframe[column].fillna(0)
  
  return dataframe


#Reemplaza los valores nan de la columna indicada por la mediana
def fill_by_median(dataframe, list_to_fill):
  for column in list_to_fill:
    dataframe[column] = dataframe[column].fillna(dataframe[column].median())

  return dataframe

#Reemplaza los valores nan de la columna indicada con la frase ingresada
def fill_by_words(dataframe, list_to_fill):
  wordcolumns=list(list_to_fill.keys())

  for column in wordcolumns:
    dataframe[column] = dataframe[column].fillna(list_to_fill[column])

  return dataframe

#Función de limpieza principal para eliminar valores NaN del dataframe
def fill_dataframe(dataframe, to_0, to_median, to_words):
  dataframe=fill_nan_to_0(dataframe,list_to_fill=to_0)

  dataframe=fill_by_median(dataframe,list_to_fill=to_median)

  dataframe=fill_by_words(dataframe,list_to_fill=to_words)

  return dataframe


#Convierte a data frame el arreglo en el cual todos los valores en 'array' de 'column' son convertidos a 'value'
def do_isin(dataframe, column, array, value):
  dataframe[column]  = pd.Series(np.where(dataframe[column].isin(array),value,dataframe[column]))

  return dataframe

#Crea un 'newcolumn' en el dataframe con la comparacion numerica realizada con los valores en 'array' en la respectiva 'column'
def do_inrange(dataframe, column, newcolumn, array):
  dataframe[newcolumn] = pd.Series(np.where((dataframe[column] > array[0]) & (dataframe[column] < array[1]), True, False))

  return dataframe

#Función principal de análisis
def do_analysis(dataframe):
  dataframe=do_isin(dataframe,'seg_un',[0,3],0)
  dataframe=do_isin(dataframe,'grp_riesgociiu',['grupo_2','grupo_3','grupo_9','grupo_8','grupo_1'],'grupo_11')

  dataframe=do_inrange(dataframe,'edad','joven',[18,30])
  dataframe=do_inrange(dataframe,'edad','adulto',[31,45])

  return dataframe


#Codifica la data que presente clasificacion según el peso asignado por un objeto entrenado tipo 'LabelEncoder'
def encoding_data(dataframe, features):

  for column in features:
    #print(column)
    ds_clase=LabelEncoder()
    ds_clase.fit(dataframe[str(column)])
    dataframe[str(column)]=ds_clase.transform(dataframe[str(column)])
  
  return dataframe


def asign_types(dataframe, list_types):
  features=list(list_types.keys())

  for column in features:
    dataframe[column] = dataframe[column].astype(list_types[column])

  return dataframe