# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:12:12 2023

@author: Admin
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu


#Loading the saved models

diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav','rb'))


#Sidebar for navigation

with st.sidebar:
    
    selected = option_menu('Sistema de predicción de enfermedades múltiples',
                           ['Predicción de diabetes',
                            'Prediccion de enfermedades cardiacas'],
                           icons = ['activity','heart'],
                           default_index = 0)
    
    
#Diabetes Prediction Page
if(selected == 'Predicción de diabetes'):
    
    #Page title
    st.title('Predicción de diabetes mediante ML')
    
    Pregnancies = st.text_input('Número de embarazos')
    Glucose = st.text_input('Nivel de glucosa (Menos de 100 mg/dL (5,6 mmol/L ) se considera normal)')
    BloodPressure = st.text_input('Valor de presión arterial')
    SkinThickness = st.text_input('Valor de espesor de piel')
    Insulin = st.text_input('Nivel de insulina')
    BMI = st.text_input('valor de IMC')
    DiabetesPedigreeFunction = st.text_input('Valor de la función Generaciones de la diabetes')
    Age = st.text_input('Edad de la Persona')
    
    
    #Code for prediction
    diab_diagnosis = ''
    
    #Creating a button for prediction
    
    if st.button('Resultado de la prueba de diabetes'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0]==1):
            diab_diagnosis = 'La persona es diabética'
            
        else:
            diab_diagnosis = 'La persona no es diabética'
            
            
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
            heart_diagnosis = 'La persona sufre una enfermedad cardíaca.'
            
        else:
            heart_diagnosis = 'La persona no padece ninguna enfermedad cardíaca.'
            
            
    st.success(heart_diagnosis)
    
    
    

    
# #Parkinsons Prediction Page
# if(selected == 'Predicción del Parkinson'):
    
#     #Page title
#     st.title('Predicción del Parkinson Usando ML')
    

#     fo = st.text_input('MDVP:Fo(Hz)')
#     fhi = st.text_input('MDVP:Fhi(Hz)')
#     flo = st.text_input('MDVP:Flo(Hz)')
#     Jitter_percent = st.text_input('MDVP:Jitter(%)')
#     Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
#     RAP = st.text_input('MDVP:RAP')
#     PPQ = st.text_input('MDVP:PPQ')
#     DDP = st.text_input('Jitter:DDP')
#     Shimmer = st.text_input('MDVP:Shimmer')
#     Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
#     APQ3 = st.text_input('Shimmer:APQ3')
#     APQ5 = st.text_input('Shimmer:APQ5')
#     APQ = st.text_input('MDVP:APQ')
#     DDA = st.text_input('Shimmer:DDA')
#     NHR = st.text_input('NHR')
#     HNR = st.text_input('HNR')
#     RPDE = st.text_input('RPDE')
#     DFA = st.text_input('DFA')
#     spread1 = st.text_input('spread1')
#     spread2 = st.text_input('spread2')
#     D2 = st.text_input('D2')
#     PPE = st.text_input('PPE')
        
        
#     #Code for prediction
#     parkinsons_diagnosis = ''
        
#     #Creating a button for prediction
        
#     if st.button('Resultado de la prueba de Parkinson'):
#             parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
            
#             if (parkinsons_prediction[0]==1):
#                 parkinsons_diagnosis = 'La persona sufre la enfermedad de Parkinson.'
                
#             else:
#                 parkinsons_diagnosis = 'La persona no padece la enfermedad de Parkinson.'
                
                
#     st.success(parkinsons_diagnosis)
        
        
