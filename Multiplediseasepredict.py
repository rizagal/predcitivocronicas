# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:12:12 2023

@author: Admin
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd


#Loading the saved models

diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav','rb'))


#Sidebar for navigation

with st.sidebar:
    
    selected = option_menu('Sistema de predicción de enfermedades múltiples',
                           ['Predicción de diabetes',
                            'Prediccion de enfermedades cardiacas1',
                           'Deteccion de Datos Anomalos'],
                           icons = ['activity','heart'],
                           default_index = 0)
    
    
#Diabetes Prediction Page
if(selected == 'Predicción de diabetes'):
    
    #Page title
    st.title('Predicción de diabetes mediante ML')
    
    Pregnancies = st.text_input('Número de embarazos')
    Glucose = st.text_input('Nivel de glucosa --- (Menos de 100 mg/dL (5,6 mmol/L ) se considera normal)')
    BloodPressure = st.text_input('Valor de presión arterial --- (Se considera normal presión sistólica de menos de 120 y una presión diastólica de menos de 80)')
    SkinThickness = st.text_input('Valor de espesor de piel --- (En la mayoría de las partes del cuerpo la epidermis tiene un espesor de sólo 0,1 mm aproximadamente en total, más delgada en la piel que rodea los ojos (0,05mm) y más gruesa (entre 1 y 5mm) en las plantas de los pies.)')
    Insulin = st.text_input('Nivel de insulina --- (Los valores normales de insulina en sangre se encuentran entre 5-25 unidades por mililitro (U/ml). Cuando este parámetro es mayor a 30 U/ml en ayunas, se plantea una insulinorresistencia.)')
    BMI = st.text_input('Valor de IMC --- (Si su IMC es entre 18.5 y 24.9, se encuentra dentro del rango de peso normal o saludable. Si su IMC es entre 25.0 y 29.9, se encuentra dentro del rango de sobrepeso. Si su IMC es 30.0 o superior, se encuentra dentro del rango de obesidad.)')
    DiabetesPedigreeFunction = st.text_input('Valor de la función Generaciones de la diabetes')
    Age = st.text_input('Edad de la Persona')
    
    
    #Code for prediction
    diab_diagnosis = ''
    
    #Creating a button for prediction
    
    if st.button('Resultado de la prueba de diabetes'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0]==1):
            diab_diagnosis = 'La persona es propensa a diabétes'
            
        else:
            diab_diagnosis = 'La persona no es propensa a diabétes'
            
            
    st.success(diab_diagnosis)
    
    
    
            
#Heart Disease Prediction Page
if(selected == 'Prediccion de enfermedades cardiacas'):
    
    #Page title
    st.title('Prediccion de enfermedades cardiacas usando ML')
    
    age = st.number_input('Edad de la Persona')
    sex = st.number_input('Sexo de la Persona')
    cp = st.number_input('Tipos de dolor en el pecho')
    trestbps = st.number_input('Presión arterial en reposo')
    chol = st.number_input('Colesterol sérico en mg/dl')
    fbs = st.number_input('Glucemia en ayunas > 120 mg/dl')
    restecg = st.number_input('Resultados electrocardiográficos en reposo')
    thalach = st.number_input('Frecuencia cardíaca máxima alcanzada')
    exang = st.number_input('Angina inducida por ejercicio')
    oldpeak = st.number_input('Depresión del ST inducida por el ejercicio.')
    slope = st.number_input('Pendiente del segmento ST del ejercicio máximo')
    ca = st.number_input('Vasos principales coloreados por fluoroscopia.')
    thal = st.number_input('tal: 0 = normal; 1 = defecto solucionado; 2 = defecto reversible')
    
    
    #Code for prediction
    heart_diagnosis = ''
    
    #Creating a button for prediction
    
    if st.button('Resultado de la prueba cardíaca'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        if (heart_prediction[0]==1):
            heart_diagnosis = 'La persona es propensa a una enfermedad cardíaca.'
            
        else:
            heart_diagnosis = 'La persona no es propensa a ninguna enfermedad cardíaca.'
            
            
    st.success(heart_diagnosis)
    
    
if(selected == 'Deteccion de Datos Anomalos'):
    
    #Page title
    st.title('Deteccion de Datos Anomalos - Con Bosques de Aislamiento (Iforests)')
    
# Cargar datos
carros = np.loadtxt("carros_usados.csv", delimiter=",")

data=pd.read_csv("carros_usados.csv")
st.write(data.head())

resultados = np.zeros((3, carros.size//2))

# Bosques de Aislamiento con diferente contaminación
c = [0.05, 0.1] 
for i in range(len(c)):
    modelo = IsolationForest(contamination=c[i]).fit(carros)
    resultados[i] = modelo.predict(carros)
    
# Graficar datos anómalos 
plt.set_cmap("jet")
fig = plt.figure(figsize=(13, 4))


for i in range(len(c)):    
    ax = fig.add_subplot(1, 3, i+1)
    ax.scatter(carros[resultados[i]==-1][:, 0], 
               carros[resultados[i]==-1][:, 1], 
               c="skyblue", marker="s", s=500)
    ax.scatter(carros[:, 0], 
               carros[:, 1], 
               c=range(carros.size//2), marker="x",
               s=500, alpha=0.6)
    ax.set_title("Contaminación: %0.2f" % c[i], size=18, color="purple")
    ax.set_ylabel("Precio ($)", size=10)
    ax.set_xlabel("Kms recorridos", size=14)

st.pyplot(fig)

        
