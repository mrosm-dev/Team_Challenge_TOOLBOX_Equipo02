# 🧰 Caja de Herramientas para Regresión

Este repositorio contiene una toolbox en Python orientada al análisis exploratorio de datos (EDA) y la selección de variables en problemas de regresión, combinando:

- Tipificación automática de variables
- Tests estadísticos
- Selección de features
- Visualizaciones interpretables

El proyecto incluye dos versiones de la misma toolbox:
- V1 → Implementación fiel al enunciado original
- V2 → Reinterpretación personal, más robusta y reutilizable


## 🎯 Objetivo del proyecto

Construir una base reutilizable para las primeras fases de cualquier pipeline de Machine Learning en regresión:

- Entender rápidamente la naturaleza de las variables
- Detectar relaciones relevantes con la variable objetivo
- Reducir dimensionalidad de forma justificada
- Facilitar la interpretación previa al modelado

Todo ello aplicado sobre datasets reales, no ejemplos sintéticos.


## 📂 Estructura del repositorio

├── data/

│_________AutoInsurance.csv

├── Team_Challenge_ToolBox.ipynb    → Enunciado original de las funciones

├── toolbox_ML.py                   → Versión V1 (enunciado)

├── toolbox_ML_v2.py                → Versión V2 (refactor y mejoras)

├── notebook_demo.ipynb             → Notebook de demostración y comparación

└── README.md


## 🔄 Filosofía V1 vs V2

V1 — Enfoque académico
- Cumple estrictamente el enunciado
- Salidas simples (listas)
- Validaciones mínimas
- Lógica funcional clara

V2 — Enfoque profesional
La V2 reinterpreta el problema priorizando:
- Robustez (validaciones explícitas y errores informativos)
- Reutilización (tipificación de variables compartida entre funciones)
- Salidas ricas (DataFrames con métricas como correlación y p-value)
- Visualización explicativa (gráficos con contexto estadístico integrado)


## 🧠 Funcionalidades principales

1. Descripción y tipificación de variables
- Tipo de dato
- Valores nulos
- Cardinalidad absoluta y relativa
- Clasificación sugerida:
  - Categórica binaria
  - Categórica nominal
  - Numérica discreta
  - Numérica continua
  - Bajo interés

2. Selección de variables numéricas (regresión)
- Correlación de Pearson
- Umbral mínimo configurable
- Test de significación estadística
- Resultados estructurados y trazables

3. Selección de variables categóricas
- Mann–Whitney U para variables binarias
- ANOVA para variables categóricas multiclase
- Filtrado automático por p-value

4. Visualización guiada por estadística
- Scatter plots para variables numéricas
- KDE / histogramas condicionados para variables categóricas
- Métricas estadísticas integradas en los gráficos


## 📊 Dataset de ejemplo

Se utiliza un dataset real de clientes de seguros de automóvil (Kaggle), con una variable objetivo continua:

Customer Lifetime Value (CLV)

El dataset incluye variables:
- Numéricas
- Categóricas
- Temporales