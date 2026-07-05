import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Configuración de la interfaz de la página web
st.set_page_config(page_title="Riesgo Actuarial IA", layout="centered")

# --- ENCABEZADO OBLIGATORIO (INFORMACIÓN DEL ESTUDIANTE) ---
st.title("Evaluación de Riesgo Actuarial Inteligente")
st.markdown("### **Asignatura:** IS-701 - Inteligencia Artificial (Campus Comayagua)")
st.markdown("### **Estudiante:** Edwin Eduardo Guzmán Ramos")
st.markdown("### **Número de Cuenta:** 20211930058")
st.write("Introduzca los datos del cliente para segmentar su nivel de riesgo médico mediante K-Means.")
st.write("---") 

# --- CONFIGURACIÓN DE RUTAS ---
# Apuntamos a la carpeta 'cargos_medicos' y su subestructura de modelos
MODEL_DIR = Path("models")
KMEANS_PATH = MODEL_DIR / "kmeans_riesgo_actuarial.pkl"

# Intentar acoplar el archivo de metadata con o sin extensión explícita
META_PATH_JSON = MODEL_DIR / "model_metadata.json"
META_PATH_SIN = MODEL_DIR / "model_metadata"
META_PATH = META_PATH_JSON if META_PATH_JSON.exists() else META_PATH_SIN

@st.cache_resource
def cargar_artefactos_ia():
    if not KMEANS_PATH.exists():
        st.error(f"No se encontró el archivo del modelo en '{KMEANS_PATH}'. Verifique las carpetas en su GitHub.")
        st.stop()
        
    # Cargar el pipeline completo de scikit-learn
    modelo_kmeans = joblib.load(KMEANS_PATH)
    
    # Cargar el mapeo de riesgo del archivo de metadatos
    mapa_riesgo = {"0": "Bajo", "1": "Medio", "2": "Alto"} # Respaldo por defecto
    if META_PATH.exists():
        try:
            with open(META_PATH, "w" if not META_PATH.exists() else "r", encoding="utf-8") as f:
                meta_data = json.load(f)
                mapa_riesgo = meta_data.get("mapa_riesgo", mapa_riesgo)
        except Exception:
            pass
            
    return modelo_kmeans, mapa_riesgo

# Inicializar los modelos de Machine Learning
modelo, diccionario_riesgo = cargar_artefactos_ia()

# --- FORMULARIO DE ENTRADA DE DATOS DEL CLIENTE ---
st.subheader("Datos Demográficos y Factores de Riesgo")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Edad del Asegurado:", min_value=18, max_value=100, value=35, step=1)
    sex = st.selectbox("Sexo Biológico:", options=["Male", "Female"])
    children = st.slider("Cantidad de Hijos / Dependientes:", min_value=0, max_value=5, value=0, step=1)

with col2:
    bmi = st.slider("Índice de Masa Corporal (BMI):", min_value=15.0, max_value=60.0, value=25.0, step=0.1)
    smoker = st.selectbox("¿Es Fumador activo?:", options=["No", "Yes"])
    region = st.selectbox("Región de Residencia:", options=["Southeast", "Southwest", "Northwest", "Northeast"])

st.write("---")
st.subheader("Información Financiera")
charges = st.number_input("Cargos Médicos Históricos Anuales ($):", min_value=100.0, max_value=100000.0, value=10000.0, step=500.0)

# --- PROCESAMIENTO Y PREDICCIÓN EN TIEMPO REAL ---
if st.button("Evaluar Nivel de Riesgo Actuarial"):
    # Construcción exacta del DataFrame estructurado tal como se entrenó en Colab
    cliente_df = pd.DataFrame([{
        "age": age,
        "sex": sex.lower(),
        "bmi": bmi,
        "children": children,
        "smoker": smoker.lower(),
        "region": region.lower(),
        "charges": charges
    }])
    
    with st.spinner("Procesando transformaciones matemáticas y asignando Clúster..."):
        try:
            # Predecir el índice del clúster
            cluster_asignado = int(modelo.predict(cliente_df)[0])
            
            # Obtener la etiqueta de riesgo mapeada desde los metadatos de clase
            label_riesgo = diccionario_riesgo.get(str(cluster_asignado), "No determinado")
            
            # Textos explicativos para el dictamen del cliente
            explicaciones = {
                "Bajo": "Cliente agrupado con perfiles de menor costo médico promedio y hábitos saludables.",
                "Medio": "Cliente agrupado con perfiles de costo y factores de riesgo intermedios.",
                "Alto": "Cliente agrupado con perfiles de alto costo médico y/o factores de riesgo críticos (ej. fumador)."
            }
            dictamen = explicaciones.get(label_riesgo, "Análisis completado de forma exitosa.")
            
            st.write("---")
            st.subheader("Resultados del Análisis Estadístico")
            
            # Formatear el color del contenedor según el nivel de riesgo
            if label_riesgo == "Bajo":
                st.success(f"**Clasificación de Riesgo:** Nivel **{label_riesgo}** (Clúster {cluster_asignado})")
            elif label_riesgo == "Medio":
                st.warning(f"**Clasificación de Riesgo:** Nivel **{label_riesgo}** (Clúster {cluster_asignado})")
            else:
                st.error(f"**Clasificación de Riesgo:** Nivel **{label_riesgo}** (Clúster {cluster_asignado})")
                
            st.info(f"**Dictamen Técnico:** {dictamen}")
            
        except Exception as e:
            st.error(f"Ocurrió un error al procesar la predicción: {str(e)}")
