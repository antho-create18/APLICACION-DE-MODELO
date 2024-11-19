import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Análisis de Calidad de Productos", layout="wide")

# Configuración de estilo para gráficos
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "figure.figsize": (8, 6),
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "axes.edgecolor": "#4f4f4f",
    "axes.linewidth": 1.2
})

# Título principal
st.title("🎯 Sistema de Predicción de Calidad de Productos")
st.markdown("---")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")
    test_size = st.slider("Proporción del conjunto de prueba", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("Semilla aleatoria", 0, 100, 42)
    solver = st.selectbox("Método de optimización", 
                          ['liblinear', 'lbfgs', 'newton-cg', 'sag'],
                          help="Algoritmo para optimizar el modelo")

# Función para cargar datos
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    return df

# Carga de datos
st.header("1. 📊 Carga de Datos")
uploaded_file = st.file_uploader("Cargar archivo Excel", type="xlsx")

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())
    st.write("Estadísticas descriptivas:")
    st.dataframe(df.describe())

    # Selección de columnas para el modelo
    st.header("2. 🎯 Selección de Variables")
    col1, col2, col3 = st.columns(3)
    with col1:
        X1 = st.selectbox("Variable X1", df.columns)
    with col2:
        X2 = st.selectbox("Variable X2", df.columns)
    with col3:
        Y = st.selectbox("Variable Objetivo", df.columns)
    
    X = df[[X1, X2]].values
    y = df[Y].values

    # Preprocesamiento
    if st.button("📊 Preprocesamiento"):
        st.subheader("Limpieza y Escalado de Datos")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # División de los datos
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
        
        # Guardar en session_state
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['scaler'] = scaler
        st.session_state['X_scaled'] = X_scaled
        st.session_state['y'] = y
        
        st.success("Preprocesamiento completado: Escalado y división de datos.")
    
    # Entrenamiento
    if 'X_train' in st.session_state and st.button("⚙️ Entrenamiento"):
        st.subheader("Entrenamiento del Modelo de Regresión Logística")
        model = LogisticRegression(solver=solver)
        model.fit(st.session_state['X_train'], st.session_state['y_train'])
        
        # Guardar el modelo en session_state
        st.session_state['model'] = model
        st.success("Entrenamiento completado.")
    
    # Evaluación
    if 'model' in st.session_state and st.button("📈 Evaluación"):
        st.subheader("Evaluación del Modelo")
        y_pred = st.session_state['model'].predict(st.session_state['X_test'])
        y_prob = st.session_state['model'].predict_proba(st.session_state['X_test'])

        # Métricas de rendimiento
        col1, col2 = st.columns(2)

        # Gráfico de matriz de confusión
        with col1:
            st.subheader("Matriz de Confusión")
            cm = confusion_matrix(st.session_state['y_test'], y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='cividis', cbar=False, ax=ax, annot_kws={"size": 14})
            ax.set_title("Matriz de Confusión", pad=15)
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Valor Real")
            st.pyplot(fig)

        # Gráfico de curva ROC
        with col2:
            st.subheader("Curva ROC")
            lb = LabelBinarizer()
            y_test_binarized = lb.fit_transform(st.session_state['y_test'])
            n_classes = y_test_binarized.shape[1]
            fig, ax = plt.subplots()
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'Clase {lb.classes_[i]} (AUC = {roc_auc:.2f})', lw=2)
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_title("Curva ROC", pad=15)
            ax.set_xlabel("Falsos Positivos")
            ax.set_ylabel("Verdaderos Positivos")
            ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
            st.pyplot(fig)

        # Tres gráficos de comparación
        col3, col4, col5 = st.columns(3)

        # Gráfico 1: Modelo inicial
        with col3:
            st.subheader("Gráfico 1: Modelo Inicial")
            fig1, ax1 = plt.subplots()
            sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=st.session_state['y'], palette='viridis', ax=ax1, s=100)
            ax1.set_title("Modelo Inicial", pad=15)
            ax1.set_xlabel(X1)
            ax1.set_ylabel(X2)
            st.pyplot(fig1)

        # Gráfico 2: Modelo mejorado (con escalado)
        with col4:
            st.subheader("Gráfico 2: Modelo Mejorado")
            X_scaled = st.session_state['X_scaled']
            y = st.session_state['y']
            fig2, ax2 = plt.subplots()
            ax2.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], color='#4caf50', label='Clase 0', s=60, alpha=0.7)
            ax2.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], color='#e91e63', label='Clase 1', s=60, alpha=0.7)
            ax2.set_title("Modelo Mejorado con Escalado", pad=15)
            ax2.set_xlabel("Componente Escalada 1")
            ax2.set_ylabel("Componente Escalada 2")
            ax2.legend()
            st.pyplot(fig2)

        # Gráfico 3: Modelo con regiones de decisión
        with col5:
            st.subheader("Gráfico 3: Modelo con Regiones de Decisión")
            x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
            y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
            h = (x_max - x_min) / 100
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            Z = st.session_state['model'].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            fig3, ax3 = plt.subplots()
            ax3.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.6)
            ax3.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], color='#2196f3', label='Clase 0', edgecolor='k', s=70)
            ax3.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], color='#f44336', label='Clase 1', edgecolor='k', s=70)
            ax3.set_title("Modelo Mejorado con Regiones de Decisión", pad=15)
            ax3.set_xlabel("Componente Escalada 1")
            ax3.set_ylabel("Componente Escalada 2")
            ax3.legend(loc="upper left")
            st.pyplot(fig3)

    # Predicción interactiva
    if 'model' in st.session_state:
        st.subheader("🔍 Predicción Interactiva")
        new_x1 = st.number_input(f"Valor para {X1}", float(X[:, 0].min()), float(X[:, 0].max()))
        new_x2 = st.number_input(f"Valor para {X2}", float(X[:, 1].min()), float(X[:, 1].max()))
        if st.button("Realizar Predicción"):
            new_data = np.array([[new_x1, new_x2]])
            new_data_scaled = st.session_state['scaler'].transform(new_data)
            prediction = st.session_state['model'].predict(new_data_scaled)
            probability = st.session_state['model'].predict_proba(new_data_scaled).max()
            result = "DEFECTUOSO ❌" if prediction[0] == 1 else "NO DEFECTUOSO ✅"
            st.markdown(f"""
            ### Resultado de la Predicción:
            - **Estado Predicho**: {result}
            - **Confianza del Modelo**: {probability:.2%}
            - Un producto con {new_x1} unidades y tiempo de entrega {new_x2} minutos probablemente esté en este estado.
            """)
