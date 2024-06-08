# Que es un Indicador

Para empezar un Indicador es una expresion matematica que permite evaluar el comportamiento de una variable o caracteristica objeto de estudio.

## **Cuales son los tipos de Indicadores que podemos usar?**
- Razon. 
- Proporcion.
- Promedio
- Indice
- Tasa

## **Crear un nuevo entorno conda**
Una vez que haz instalado conda, comencemos creando un entorno para gestionar todas las dependencias de la librería Python.

Para crear un nuevo entorno con Python 3.9, ejecute lo siguiente:
```bash
conda create -n stenv python=3.9
```

donde `create -n stenv` va a crear un entorno conda llamado `stenv` y `python=3.9` va a especificar que se utilice la version 3.9 de Python en el entorno conda.

## **Activar el entorno conda**

Para utilizar el entorno conda llamado `stenv` que acabamos de crear, ejecute lo siguiente en su terminal:

```bash
conda activate stenv
```

## **Instalar la librería Streamlit**

Es tiempo de instalar la librería `streamlit`:
```bash
pip install streamlit
```

## **Ejecutar la aplicación demo de Streamlit**
Para ejecutar la aplicación demo (Figura 1) ingrese:
```bash
streamlit hello
```
