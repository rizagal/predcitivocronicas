# -*- coding: utf-8 -*-
"""
Creado a partir del 1 de Junio 2024

@author: RZ

https://icons.getbootstrap.com/
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
import plotly.express as px  # pip install plotly-express
from streamlit.components.v1 import html
from pygwalker.api.streamlit import StreamlitRenderer
import streamlit.components.v1 as components
from streamlit_pdf_reader import pdf_reader

# Este modelo lo genere en google colab en la cuenta de facildiez@gmail.com el archivo se llama Entrenar Modelo.ipynb, para crearlo me guie con: https://www.youtube.com/watch?v=lK0aVny0Rsw
#riesgocardio_model = pickle.load(open('model_datosderiesgo.pkl','rb'))

diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav','rb'))

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Resultado Indicadores y Sistema de Predicción de Enfermedades", page_icon=":bar_chart:", layout="wide")

#remove default theme
theme_plotly = None # None or streamlit

st.markdown("""
<style>

.block-container
{
    padding-top: 0rem;
    padding-bottom: 0rem;
    margin-top: 0rem;
}

</style>
""", unsafe_allow_html=True)
 
# CSS Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

st.sidebar.image("SIGES17.png",caption="")


def format_func(option):
    return CHOICES[option]    

#Sidebar for navigation

with st.sidebar:
    
    selected = option_menu('Indicadores de Calidad y Sistema de Predicción de Enfermedades',
                           ['Consulta Resultado Indicadores de Oportunidad',
                            'Consulta Resultado Indicadores Atencion al Usuario',
                            'Importancia de los Indicadores',
                            'Planeacion Integral',
                            'Visualizacion de Servicios Habilitados por IPS - REPS',
                            'Visualizacion Poblacion Contratada',
                            'Predicción de Diabetes',
                            'Modelo Construido Riesgo Cardiovascular',                            
                            'Prediccion de Enfermedades Cardiacas',
                            'Deteccion de Datos Anomalos'
                            
                            ],
                           icons = ['activity','pen','pen','pen','building','universal-access','book','heart','clipboard','person'],
                           default_index = 0)
   


#Prediccion con modelo construido con info de pacientes propios
if(selected == 'Modelo Construido Riesgo Cardiovascular'):
    
    #Page title
    st.title('Modelo Construido con Arbol de Decision (sklearn) Riesgo Cardiovascular')   

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
if(selected == 'Predicción de Diabetes'):
    
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
if(selected == 'Prediccion de Enfermedades Cardiacas'):
    
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


def color_negative_red(s):

    is_max = s == s.max()
    return ['background-color: #e88868' if v else '' for v in is_max]

def color_negative_redusuario(s):
    is_min = s == s.min()
    return ['background-color: #e88868' if v else '' for v in is_min]


#Parkinsons Prediction Page
if(selected == 'Consulta Resultado Indicadores de Oportunidad'):
    
    #Page title
    st.title('Resultado Indicadores de Calidad Red de Salud de Ladera E.S.E')   

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
        text="OPORTUNIDAD",        
        color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
        template="plotly_white",
    )
    fig_product_sales.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )
    fig_product_sales.update_traces(textposition="outside")
    fig_product_sales.update_layout(title_text='Consolidado Oportunidad por Servicio Durante los Meses Seleccionados', title_x=0.1)    
    
    col1,col2=st.columns(2)

    with col1:
       st.dataframe(df_selection.style.apply(color_negative_red, subset=['OPORTUNIDAD']).format({"OPORTUNIDAD": "{:.3}"}),hide_index=True,height=450,use_container_width=True,column_order=("SERVICIO","MES","OPORTUNIDAD","NOMBREIPS"))
   
        
    with col2:
       
        fig = px.line(df_selection, x='MES', y='OPORTUNIDAD', color='SERVICIO', text="OPORTUNIDAD", markers=True)
        fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
        )
        fig.update_traces(textposition="top center")        
        fig.update_layout(title_text='Oportunidad de la IPS por Meses y Servicios Seleccionados', title_x=0.1)   
        fig    
       



    col1,col2=st.columns(2)
    with col1:     
       fig_product_sales

    with col2:
 
        fig = px.bar(df, x='OPORTUNIDAD', y='NOMBREIPS', color='SERVICIO', text="OPORTUNIDAD", orientation="h")
        fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
        )
        fig.update_traces(textposition="inside")
        fig.update_layout(title_text='Oportunidad de Todas las IPS y Servicios', title_x=0.3)        
        fig    


def open_page(url):
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)


if(selected == 'Consulta Resultado Indicadores Atencion al Usuario'):
    
    #Page title
    st.title('Resultado Indicadores de Atencion al Usuario')   

    # ---- READ EXCEL ----
  
    df = pd.read_csv("atencionusuariostreamlit.csv")
    
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


    df_selection = df.query(
    "NOMBREIPS == @ips & MES == @mes"
    )

    st.markdown("""---""")

    # TOP KPI's
    total_sales = int(df_selection["NOMBREIPS"].count())

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        # st.subheader("Total Registros:")
        # st.subheader(f"{total_sales:,}")
        st.metric(label="Total Registros:",value=f"{total_sales:,.0f}")

    col1,col2=st.columns(2)

    with col1:
       st.dataframe(df_selection.style.apply(color_negative_redusuario, subset=['PORCENTAJE']),hide_index=True,height=450,use_container_width=True,column_order=("EVALUACION","MES","PORCENTAJE","NOMBREIPS"))
   
        
    with col2:
        st.markdown("""---""")
     


if(selected == 'Importancia de los Indicadores'):
    
    # open_page('http://ideabien-001-site2.atempurl.com/')
    # components.iframe("https://informa-51763.web.app/index3dcardesml.html", height=500)
    # with st.expander("Acerca de #30DaysOfStreamlit"):
    #     st.markdown('''
    #     **#30DaysOfStreamlit** es un desafío diseñado para ayudarlo a comenzar a crear aplicaciones Streamlit.
        
    #     En particular, podrás:
    #     - Configure un entorno de desarrollo para construir aplicaciones Streamlit
    #     - Construir tu primer aplicación Streamlit
    #     - Aprender acerca de todos los sorprendentes componentes para usar en tu aplicación Streamlit
    #     ''')
    with open(f'contenido1.md', 'r') as f:
        st.markdown(f.read())
        st.image(f'razonindicadores.png')
        with st.expander("💡 Video Tutorial de Indicadores"):
            with st.spinner("Cargando video"):
                st.video("indicadores.mp4", format="video/mp4", start_time=0)

if(selected == 'Planeacion Integral'):
    #Page title
   st.title('Planeacion Integral e Indicadores en Salud')
   source1='./Planeacion Integral e Indicadores en Salud.pdf'
   pdf_reader(source1)


if(selected == 'Visualizacion de Servicios Habilitados por IPS - REPS'):
     #Page title
    st.title('Visualizacion de Servicios Habilitados por IPS - REPS') 
    df = pd.read_csv("reps2024.csv")
    pyg_app = StreamlitRenderer(df,spec="bikes_chart.json")
    pyg_app.explorer()

if(selected == 'Visualizacion Poblacion Contratada'):
    components.iframe("https://app.powerbi.com/view?r=eyJrIjoiY2ZkZTNhYmYtZWQ2ZC00NzQxLWE3MTctZjA4MGUxOTI4N2MxIiwidCI6ImE4MjRmZDZkLTVkYzItNDdjMC1iNTQ2LTU5MWZmZGJmYmFlNiJ9", height=1000)

