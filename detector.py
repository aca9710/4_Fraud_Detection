#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 00:27:04 2023

@author: Arturo Castillo Alpizar
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay


#Modelo usado
#En estadística, la regresión logística es un tipo de análisis de regresión 
#utilizado para predecir el resultado de una variable categórica
# (una variable que puede adoptar un número limitado de categorías) en función 
# de las variables independientes o predictoras. Es útil para modelar la 
# probabilidad de un evento ocurriendo en función de otros factores. 
reg = linear_model.LogisticRegression(max_iter= 10000)  #LogisticRegressionCV(max_iter= 10000)


#Importando datos
df = pd.read_csv('card_transdata.csv')
#df.columns
#sns.heatmap(df.corr())


#Obteniendo matrices
arrx = df[df.columns[:-1]].to_numpy()
arry = df[df.columns[-1]].to_numpy()

#Generando los valores de entrenamiento y de pruebas
x_train, x_test, y_train, y_test = train_test_split(arrx, arry, train_size=0.9)

#Entrenando el modelo
reg.fit(x_train, y_train)

#Calculando prediccion
y_pred = reg.predict(x_test)


#La métrica de recall, también conocida como el ratio de verdaderos positivos,
# es utilizada para saber cuantos valores positivos son correctamente 
# clasificados.
print("Recall", reg.score(x_test, y_test))

#Calcular la precisión media (AP) a partir de las puntuaciones de las 
#predicciones
print("AP: {0:0.2f}".format(average_precision_score(y_test, y_pred)))



#AUC-ROC Curve
#En Machine Learning, la medición del rendimiento es una tarea esencial. 
#Entonces, cuando se trata de un problema de clasificación, podemos contar 
#con una curva AUC-ROC. Esta es una de las métricas de evaluación más 
#importante para verificar el rendimiento de cualquier modelo de clasificación.
#roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                   estimator_name='example estimator')
display.plot()
plt.show()


#Matríz de confusión (F1-SCORE)
#En el caso específico de un clasificador binario podemos interpretar estos 
#números como el recuento de positivos verdaderos (aciertos), positivos 
#falsos (errores), negativos verdaderos (errores), y negativos falsos
# (aciertos).
print("f1_score: {0:0.2f}".format(f1_score(y_test, y_pred)))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#ACCURACY
#La métrica accuracy representa el porcentaje total de valores correctamente 
#clasificados, tanto positivos como negativos. Es recomendable utilizar 
#esta métrica en problemas en los que los datos están balanceados, es decir, 
#que haya misma cantidad de valores de cada etiqueta (en este caso mismo 
# número de 1s y 0s).
print("accuracy_score:: {0:0.2f}".format(accuracy_score(y_test, y_pred)))

#Recall (Exhaustividad)
#La métrica de exhaustividad nos va a informar sobre la cantidad que el 
#modelo de machine learning es capaz de identificar. 
precision, recall, _ = precision_recall_curve(y_test, y_pred)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.show()

print(classification_report(y_test, y_pred))





