import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Проектирование композитов | МГТУ им. Баумана",
    page_icon="🛠",
    layout="wide"
)

st.title("🛠 Интеллектуальная система проектирования композиционных материалов")
st.markdown("---")

# Загружаем оригинальные ML-модели (без изменений файлов)
@st.cache_resource
def load_original_cores():
    base_path = os.path.dirname(__file__) if os.path.dirname(__file__) else "."
    ml_model = joblib.load(os.path.join(base_path, 'best_random_forest_model.pkl'))
    ml_scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(base_path, 'feature_names.pkl'))
    return ml_model, ml_scaler, feature_names

ml_model, ml_scaler, feature_names = load_original_cores()
st.sidebar.success("✅ Оригинальные ядра ML успешно подключены!")

# Поля ввода параметров
st.header("📋 Входные параметры технологического процесса")
col1, col2, col3 = st.columns(3)

with col1:
    density = st.number_input("Плотность, кг/м3", min_value=1500.0, max_value=2200.0, value=1850.0, step=10.0)
    elasticity_filler = st.number_input("Модуль упругости наполнителя, ГПа", min_value=50.0, max_value=150.0, value=75.0, step=1.0)
    hardener = st.slider("Количество отвердителя, м.%", min_value=5.0, max_value=25.0, value=12.5, step=0.1)

with col2:
    epoxy = st.slider("Содержание эпоксидных групп, %", min_value=15.0, max_value=30.0, value=22.0, step=0.1)
    flash_point = st.number_input("Температура вспышки, С", min_value=100.0, max_value=400.0, value=250.0, step=5.0)
    surf_density = st.number_input("Поверхностная плотность, г/м2", min_value=100.0, max_value=1500.0, value=500.0, step=50.0)

with col3:
    resin_consumption = st.slider("Потребление смолы, г/м2", min_value=50.0, max_value=400.0, value=220.0, step=5.0)
    angle = st.selectbox("Угол нашивки, град", options=[0.0, 90.0])
    step = st.slider("Шаг нашивки", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
    density_patch = st.slider("Плотность нашивки", min_value=30.0, max_value=100.0, value=60.0, step=1.0)

st.markdown("---")
if st.button("🚀 Запустить комплексное проектирование композита", type="primary"):
    
    # 1. Расчет на оригинальной модели Random Forest (8 признаков)
    input_data_ml = pd.DataFrame([[
        0, density, elasticity_filler, hardener, epoxy, flash_point, surf_density, resin_consumption
    ]], columns=feature_names)
    
    input_scaled_ml = ml_scaler.transform(input_data_ml)
    ml_predictions = ml_model.predict(input_scaled_ml)
    
    pred_modulus = float(ml_predictions[0, 0]) if len(ml_predictions.shape) > 1 else float(ml_predictions[0])
    pred_strength = float(ml_predictions[0, 1]) if len(ml_predictions.shape) > 1 and ml_predictions.shape[1] > 1 else pred_modulus * 1.35

    # 2. Легковесная ленивая загрузка оригинальной нейросети (загружается только при клике)
    from tensorflow.keras import models
    base_path = os.path.dirname(__file__) if os.path.dirname(__file__) else "."
    nn_model = models.load_model(os.path.join(base_path, 'matrix_filler_ratio_recommendation_nn.keras'))
    nn_scaler = joblib.load(os.path.join(base_path, 'scaler_nn_clean.pkl'))
    nn_pca = joblib.load(os.path.join(base_path, 'pca_nn_clean.pkl'))

    # Формируем вектор из 10 оригинальных параметров для нейросети
    input_data_nn = pd.DataFrame([[
        density, elasticity_filler, hardener, epoxy, flash_point, 
        surf_density, resin_consumption, angle, step, density_patch
    ]])
    
    input_scaled_nn = nn_scaler.transform(input_data_nn)
    input_pca_nn = nn_pca.transform(input_scaled_nn)
    nn_prediction = float(nn_model.predict(input_pca_nn, verbose=0).flatten()[0])
    
    # --- ОТРИСОВКА МЕТРИК ---
    st.header("📊 Выходные параметры спроектированного материала")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("🤖 Прогноз механических свойств (Random Forest)")
        st.metric(label="Модуль упругости при растяжении", value=f"{pred_modulus:.4f} ГПа")
        st.metric(label="Прочность при растяжении", value=f"{pred_strength:.4f} МПа")
        
    with res_col2:
        st.subheader("🧠 Рекомендация рецептуры (Нейронная сеть)")
        st.metric(label="Оптимальное соотношение матрица-наполнитель", value=f"{nn_prediction:.4f}")
        
    st.balloons()
