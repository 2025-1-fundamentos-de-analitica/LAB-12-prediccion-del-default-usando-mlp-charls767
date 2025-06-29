import zipfile
import pickle
import gzip
import json
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


def clean_data(df):
    # Creamos una copia
    df = df.copy()
    # Quitamos la columna ID
    df = df.drop('ID', axis=1)
    # Renombramos la columna 
    df = df.rename(columns={'default payment next month': 'default'})
    # Eliminamos registros con datos faltantes
    df = df.dropna()
    # Eliminar filas donde el registro es NA
    df = df[(df['EDUCATION'] != 0 ) & (df['MARRIAGE'] != 0)]
    # Para la columna EDUCATION, valores > 4 indican niveles superiores de educación, agrupe estos valores en la categoría "others".
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4

    return df

def model():
    # Transformacion de variables categoricas
    categories = ['SEX', 'EDUCATION', 'MARRIAGE']  
    # Transformacion de variables numericas
    numerics = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5","PAY_AMT6"
    ]

    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categories),
            ('scaler', StandardScaler(), numerics)
        ],
        remainder='passthrough'  # deja las columnas no categóricas como están
    )

    # Creamos el Select K Best
    selectkbest = SelectKBest(score_func=f_classif)

    # Crear el pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ("selectkbest", selectkbest),
        ('pca', PCA()),  # Si no se le pasa argumentos usa todas las componentes 
        ('classifier', MLPClassifier(max_iter=15000, random_state=42))
    ])

    return pipeline

def hyperparameters(model, n_splits, x_train, y_train, scoring):
    # Búsqueda de parámetros con validación cruzada
    estimator = GridSearchCV(
        estimator=model,
        param_grid = {
            'pca__n_components': [None],
            'selectkbest__k': [20],
            'classifier__hidden_layer_sizes': [(50, 30, 40, 60)],
            'classifier__alpha': [0.28],
            'classifier__learning_rate_init': [0.001],
        },
        cv=n_splits,
        refit=True,
        scoring=scoring
    )
    # Entrenamos
    estimator.fit(x_train, y_train)

    return estimator

def metrics(model, x_train, y_train, x_test, y_test):
    # Realizamos las predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Creamos los diccionarios con las metricas
    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': (precision_score(y_train, y_train_pred, average='binary')),
        'balanced_accuracy':(balanced_accuracy_score(y_train, y_train_pred)),
        'recall': (recall_score(y_train, y_train_pred, average='binary')),
        'f1_score': (f1_score(y_train, y_train_pred, average='binary'))
    }

    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': (precision_score(y_test, y_test_pred, average='binary')),
        'balanced_accuracy':(balanced_accuracy_score(y_test, y_test_pred)),
        'recall': (recall_score(y_test, y_test_pred, average='binary')),
        'f1_score': (f1_score(y_test, y_test_pred, average='binary'))
    }

    return train_metrics, test_metrics

def matrix(model, x_train, y_train, x_test, y_test):
    # Realizamos las predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculamos los valores de la matriz de confusion
    cm_train = confusion_matrix(y_train, y_train_pred)
    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()

    cm_test = confusion_matrix(y_test, y_test_pred)
    tn_test, fp_test, fn_test, tp_test = cm_test.ravel()


    # Creamos los diccionarios
    train_matrix = {
        'type': 'cm_matrix',
        'dataset': 'train', 
        'true_0': {
            'predicted_0': int(tn_train),
            'predicted_1': int(fp_train)
        },
        'true_1': {
            'predicted_0': int(fn_train),
            'predicted_1': int(tp_train)
        }
    }

    test_matrix = {
        'type': 'cm_matrix',
        'dataset': 'test', 
        'true_0': {
            'predicted_0': int(tn_test),
            'predicted_1': int(fp_test)
        },
        'true_1': {
            'predicted_0': int(fn_test),
            'predicted_1': int(tp_test)
        }
    }

    return train_matrix, test_matrix

def save_model(model):
    # Crear las carpetas si no existen
    os.makedirs('files/models', exist_ok=True)

    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)

def save_metrics(metrics):
    # Crear las carpetas si no existen
    os.makedirs('files/output', exist_ok=True)

    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            json_line = json.dumps(metric)
            f.write(json_line + "\n")

############################################################################################

# Nombres de los archivos
file_Test = 'files/input/test_data.csv.zip'
file_Train = 'files/input/train_data.csv.zip'

# Leemos los archivos zip
with zipfile.ZipFile(file_Test, 'r') as zip:
    # Leemos cada archivo csv
    with zip.open('test_default_of_credit_card_clients.csv') as f:
        # Creamos un dataframe con el archivo
        df_Test = pd.read_csv(f)

# Leemos los archivos zip
with zipfile.ZipFile(file_Train, 'r') as zip:
    # Leemos cada archivo csv
    with zip.open('train_default_of_credit_card_clients.csv') as f:
        # Creamos un dataframe con el archivo
        df_Train = pd.read_csv(f)

# Realizamos la limpieza de los dataset
df_Test = clean_data(df_Test)
df_Train = clean_data(df_Train)

#  Dividimos los datasets en x_train, y_train, x_test, y_test.
x_train, y_train = df_Train.drop('default', axis=1), df_Train['default']
x_test, y_test = df_Test.drop('default', axis=1), df_Test['default']

# Creamos el modelo
model_pipeline = model()

# Optimizamos los parametros
model_pipeline = hyperparameters(model_pipeline, 10, x_train, y_train, 'balanced_accuracy')

# Guardamos el modelo
save_model(model_pipeline)

# Calculamos las metricas para el conjunto de entrenamiento y prueba
train_metrics, test_metrics = metrics(model_pipeline, x_train, y_train, x_test, y_test)

# Calculamos las metricas de la matriz de confusion para el conjunto de entrenamiento y prueba
train_matrix, test_matrix = matrix(model_pipeline, x_train, y_train, x_test, y_test)

# Guardamos todas las metricas calculadas
save_metrics([train_metrics, test_metrics, train_matrix, test_matrix])