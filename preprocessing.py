import numpy as np
import pandas as pd
import pickle


def get_garbage_features(nombre_archivo):
    """
    Crea una lista de Features a partir de un archivo txt que contiene features artificiales.

    Parámetros:
    - nombre_archivo: Ruta del archivo txt que contiene las features (str).

    Retorna:
    - lista_resultado: Lista de features procesadas (list).

    Imprime:
    - Mensajes de error si ocurren problemas durante la lectura o procesamiento del archivo.
    """
    # Lista para almacenar los resultados
    lista_resultado = []

    try:
        # Abrir el archivo en modo lectura
        with open(nombre_archivo, 'r') as archivo:
            # Leer cada línea del archivo
            for linea in archivo:
                # Eliminar los espacios en blanco al inicio y al final de la línea
                linea = linea.strip()

                # Verificar si la línea no está vacía
                if linea:
                    # Convertir la línea a entero y agregar a la lista
                    numero = int(linea)
                    feature = f'Feature_{numero}'
                    lista_resultado.append(feature)

    except FileNotFoundError:
        print(f"El archivo '{nombre_archivo}' no fue encontrado.")
    except ValueError as ve:
        print(f"Error al convertir a entero: {ve}")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

    return lista_resultado



def clean_data(data, garbage_features):
    """
    Elimina columnas especificadas de data.

    Parámetros:
    - data (pd.DataFrame): El DataFrame original.
    - garbage_features iteralbe: Lista de nombres de columnas a ser eliminadas.

    Retorna:
    - pd.DataFrame: El DataFrame actualizado después de eliminar las columnas.
    """
    data = data.drop(garbage_features, axis=1)
    return data

def read_pickle(file):
    with open(file, 'rb') as archivo:
        # Cargar la lista desde el archivo
        mi_lista_recuperada = pickle.load(archivo)
        #print(mi_lista_recuperada)
        return mi_lista_recuperada

