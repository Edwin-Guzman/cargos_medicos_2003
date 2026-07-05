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

# --- CARGA DEL MODELO (Tal como lo pidió la ingeniera) ---
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load("models/kmeans_riesgo_actuarial.pkl")
    except Exception as e:
        st.error(f"No se pudo cargar el modelo. Verifique que el archivo esté en 'models/kmeans_riesgo_actuarial.pkl'. Error: {e}")
        st.stop()

modelo = cargar_modelo()

# Mapeo estándar de los 3 clústeres entrenados en clase
# Nota: Si en tu Colab el clúster 0 era el de cargos más bajos, el orden es Bajo, Medio, Alto.
# Este diccionario traduce el número que da el modelo a texto comprensible.
mapa_riesgo = {
    0: "Bajo",
    1: "Medio",
    2: "Alto"
}

# --- FORMULARIO INTERACTIVO (Complementando el diccionario estático de la ingeniera) ---
st.subheader("Datos Demográficos y Factores de Riesgo")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Edad del Asegurado:", min_value=18, max_value=100, value=45, step=1)
    sex = st.selectbox("Sexo Biológico:", options=["Male", "Female"])
    children = st.slider("Cantidad de Hijos / Dependientes:", min_value=0, max_value=5, value=2, step=1)

with col2:
    bmi = st.slider("Índice de Masa Corporal (BMI):", min_value=15.0, max_value=60.0, value=31.2, step=0.1)
    smoker = st.selectbox("¿Es Fumador activo?:", options=["Yes", "No"])
    region = st.selectbox("Región de Residencia:", options=["Southeast", "Southwest", "Northwest", "Northeast"])

st.write("---")
st.subheader("Información Financiera")
charges = st.number_input("Cargos Médicos Históricos Anuales ($):", min_value=100.0, max_value=100000.0, value=28000.0, step=500.0)

# --- BOTÓN DE EVALUACIÓN ---
if st.button("Evaluar Nivel de Riesgo Actuarial"):
    
    # Construcción exacta del DataFrame estructurado tal como lo dejó la ingeniera
    cliente = pd.DataFrame([{
        "age": age,
        "sex": sex.lower(),
        "bmi": bmi,
        "children": children,
        "smoker": smoker.lower(),
        "region": region.lower(),
        "charges": charges
    }])
    
    with st.spinner("Calculando asignación de Clúster..."):
        try:
            # Predicción idéntica al código de la ingeniera:
            cluster = modelo.predict(cliente)[0]
            
            # Obtener el nivel de riesgo asociado al número de clúster
            riesgo_detectado = mapa_riesgo.get(cluster, "No determinado")
            
            st.write("---")
            st.subheader("Resultados del Análisis")
            
            # Formatear visualmente el resultado según el clúster
            if riesgo_detectado == "Bajo":
                st.success(f"**Resultado:** Clúster asignado: **[{cluster}]** - Riesgo **{riesgo_detectado}**")
            elif riesgo_detectado == "Medio":
                st.warning(f"**Resultado:** Clúster asignado: **[{cluster}]** - Riesgo **{riesgo_detectado}**")
            else:
                st.error(f"**Resultado:** Clúster asignado: **[{cluster}]** - Riesgo **{riesgo_detectado}**")
                
            st.info("El modelo ha procesado las variables demográficas y financieras mapeándolas de forma exitosa en el espacio vectorial entrenado.")
            
        except Exception as e:
            st.error(f"Ocurrió un error en la predicción: {str(e)}")
