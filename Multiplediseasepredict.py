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
import plotly.express as px  # pip install plotly-express

#Loading the saved modelsg

diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav','rb'))

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Sistema de predicción de enfermedades múltiples", page_icon=":bar_chart:", layout="wide")


#Sidebar for navigation

with st.sidebar:
    
    selected = option_menu('Sistema de predicción de enfermedades múltiples',
                           ['Predicción de diabetes',
                            'Prediccion de enfermedades cardiacas',
                            'Deteccion de Datos Anomalos',
                            'Visualizar Datos en Tabla'],
                           icons = ['activity','heart'],
                           default_index = 0)
    
    
#Diabetes Prediction Page
if(selected == 'Predicción de diabetes'):
    
    #Page title
    st.title('Predicción de diabetes mediante ML')
    
    Pregnancies = st.text_input('Número de embarazos')
    Glucose = st.text_input('Nivel de glucosa')
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
    
    
    

    
#Parkinsons Prediction Page
if(selected == 'Deteccion de Datos Anomalos'):
    
    #Page title
    st.title('Deteccion de Datos Anomalos - Con Bosques de Aislamiento (Iforests)')
    
    # Cargar datos
    carros = np.loadtxt("carros_usados.csv", delimiter=",")

    # data= pickle.load(open('diabetes_model.sav','rb'))
    # st.write(data.head())

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





#Parkinsons Prediction Page
if(selected == 'Visualizar Datos en Tabla'):
    
    #Page title
    st.title('Visualizar Datos en Tabla')   

    # ---- READ EXCEL ----
  
    df = pd.read_csv("oportunidadstreamlit.csv")
    
    st.sidebar.header("Favor Filtrar:")
    ips = st.sidebar.multiselect(
    "Seleccione IPS:",
    options=df["NOMBREIPS"].unique(),
    default=df["NOMBREIPS"].unique()
    )

    mes = st.sidebar.multiselect(
    "Seleccione Mes:",
    options=df["MES"].unique(),
    default=df["MES"].unique()
    )


    df_selection = df.query(
    "NOMBREIPS == @ips & MES == @mes"
    )

    # st.dataframe(df_selection)

    st.markdown("""---""")

    # TOP KPI's
    total_sales = int(df_selection["NOMBREIPS"].count())
    # average_rating = round(df_selection["Rating"].mean(), 1)
    # star_rating = ":star:" * int(round(average_rating, 0))
    # average_sale_by_transaction = round(df_selection["Total"].mean(), 2)

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Total Registros:")
        st.subheader(f"{total_sales:,}")
    # with middle_column:
    #     st.subheader("Average Rating:")
    #     st.subheader(f"{average_rating} {star_rating}")
    # with right_column:
    #     st.subheader("Average Sales Per Transaction:")
    #     st.subheader(f"US $ {average_sale_by_transaction}")
    

    # SALES BY PRODUCT LINE [BAR CHART]
    sales_by_product_line = df_selection.groupby(by=["SERVICIO"])[["OPORTUNIDAD"]].mean().sort_values(by="OPORTUNIDAD")
    fig_product_sales = px.bar(
        sales_by_product_line,
        x="OPORTUNIDAD",
        y=sales_by_product_line.index,
        orientation="h",
        title="<b>Oportunidad por Servicio</b>",
        color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
        template="plotly_white",
    )
    fig_product_sales.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

   

    col1,col2=st.columns(2)

    with col1:
      st.dataframe(df_selection)
    with col2:
       fig_product_sales       
