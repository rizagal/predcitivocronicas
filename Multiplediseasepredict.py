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
import webbrowser
import streamlit.components.v1 as components

# Este modelo lo genere en google colab en la cuenta de facildiez@gmail.com el archivo se llama Entrenar Modelo.ipynb, para crearlo me guie con: https://www.youtube.com/watch?v=lK0aVny0Rsw
riesgocardio_model = pickle.load(open('model_datosderiesgo.pkl','rb'))

diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav','rb'))

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Resultado Indicadores y Sistema de predicción de enfermedades", page_icon=":bar_chart:", layout="wide")

#remove default theme
theme_plotly = None # None or streamlit

 
# CSS Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

st.sidebar.image("SIGES17.png",caption="")


def format_func(option):
    return CHOICES[option]    

#Sidebar for navigation

with st.sidebar:
    
    selected = option_menu('Resultado Indicadores y Sistema de Predicción de Enfermedades',
                           ['Indicadores de Calidad',
                            'Prediccion de enfermedades cardiacas',
                            'Predicción de diabetes',
                            'Modelo Construido Riesgo Cardiovascular.',
                            'Deteccion de Datos Anomalos',
                            'Cuerpo Humano Interactivo'],
                           icons = ['activity','heart','house','book','pen','person'],
                           default_index = 0)
   


#Prediccion con modelo construido con info de pacientes propios
if(selected == 'Modelo Construido Riesgo Cardiovascular'):
    
    #Page title
    st.title('Modelo Construido Riesgo Cardiovascular')   

    # Para que funcione el selectbox se necesita de la funcion que esta arriba def format_func(option):
    # return CHOICES[option]    
    CHOICES = {1: "Hombre", 2: "Mujer"}
    option = st.selectbox("Seleccione Sexo", options=list(CHOICES.keys()), format_func=format_func)
    # st.write(option)
    genero = option

    Edad = st.text_input('Edad')

    CHOICES = {1: "Si", 2: "No"}
    option1 = st.selectbox("Realiza Actividad Fisica", options=list(CHOICES.keys()), format_func=format_func)
    Fisica = option1
     
    CHOICES = {1: "Si", 2: "No"}
    option2 = st.selectbox("Come Verduras y Frutas", options=list(CHOICES.keys()), format_func=format_func)
    Verdufrutas = option2

    CHOICES = {2: "No", 1: "Si"}
    option3 = st.selectbox("Toma Medicamentos para la Hipertension Regularmente", options=list(CHOICES.keys()), format_func=format_func)
    Medihiper = option3

    CHOICES = {2: "No", 1: "Si"}
    option4 = st.selectbox("Algunos Valores de Glucosa Altos", options=list(CHOICES.keys()), format_func=format_func)
    Glucoaltos = option4

    CHOICES = {2: "No", 1: "Si"}
    option5 = st.selectbox("Diagnsotico Con Diabetes Algun Familiar", options=list(CHOICES.keys()), format_func=format_func)
    Diagdiabfamili = option5
    #Code for prediction
    diab_diagnosis = ''
    
    #Creating a button for prediction    
    if st.button('Resultado de Riesgo Cardio Vascular Modelo Creado'):
        diab_prediction = riesgocardio_model.predict([[genero, Edad, Fisica, Verdufrutas, Medihiper, Glucoaltos, Diagdiabfamili]])       
        
        if (diab_prediction!=1):
            diab_diagnosis = diab_prediction    
            
    st.success(diab_diagnosis)
    


    
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
    st.title('Deteccion de Datos Anomalos en Oportunidad de Consulta Medica- Con Bosques de Aislamiento (Iforests)')
    
    # Cargar datos
    carros = np.loadtxt("deteccion_anomalos.csv", skiprows=1, usecols=(1, 2), delimiter=",")

    resultados = np.zeros((3, carros.size//2))

    # Bosques de Aislamiento con diferente contaminación
    c = [0.05, 0.04] 
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
                c="skyblue", marker="s", s=900)
        ax.scatter(carros[:, 0], 
                carros[:, 1], 
                c=range(carros.size//2), marker="x",
                s=900, alpha=0.1)
        ax.set_title("Contaminación: %0.2f" % c[i], size=12, color="purple")
        ax.set_ylabel("Oportunidad", size=10)
        ax.set_xlabel("Meses", size=10)

    st.pyplot(fig)





#Parkinsons Prediction Page
if(selected == 'Indicadores de Calidad'):
    
    #Page title
    st.title('Resultado Indicadores de Calidad')   

    # ---- READ EXCEL ----
  
    df = pd.read_csv("oportunidadstreamlit.csv")
    
    st.header("Favor Filtrar:")
    ips = st.selectbox(
    "Seleccione IPS:",
    options=df["NOMBREIPS"].unique(),
    help="Seleccione Sede",  
    )

    mes = st.multiselect(
    "Seleccione Mes:",
    options=df["MES"].unique(),
    default=df["MES"].unique()
)


    servicio = st.multiselect(
    "Seleccione Servicio:",
    options=df["SERVICIO"].unique(),
    default=df["SERVICIO"].unique()
)

    df_selection = df.query(
    "NOMBREIPS == @ips & MES == @mes & SERVICIO == @servicio"
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
        # st.subheader("Total Registros:")
        # st.subheader(f"{total_sales:,}")
        st.metric(label="Total Registros:",value=f"{total_sales:,.0f}")
    # with middle_column:
    #     st.subheader("Average Rating:")
    #     st.subheader(f"{average_rating} {star_rating}")
    # with right_column:
    #     st.subheader("Average Sales Per Transaction:")
    #     st.subheader(f"US $ {average_sale_by_transaction}")
    

    # [BAR CHART]
    sales_by_product_line = df_selection.groupby(by=["SERVICIO"])[["OPORTUNIDAD"]].mean().sort_values(by="OPORTUNIDAD")
    fig_product_sales = px.bar(
        sales_by_product_line,
        x="OPORTUNIDAD",
        y=sales_by_product_line.index,
        orientation="h",
        title="<b>Consolidado Oportunidad por Servicio Durante los Meses Seleccionados</b>",
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

if(selected == 'Cuerpo Humano Interactivo'):
    components.iframe("https://www.google.com/?hl=es", height=900, width=500)


