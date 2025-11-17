# ⏱️ DSRP-MLE3: Clasificación Supervisada
![release](https://img.shields.io/badge/release-v1.0.0-blue.svg)
![python](https://img.shields.io/badge/Python-3.11.1-blue?logo=python)
![editor](https://img.shields.io/badge/Editor-JupyterLab-F37626?logo=jupyter)
![framework](https://img.shields.io/badge/Framework-Scikit--learn-orange?logo=scikit-learn)
![xgboost](https://img.shields.io/badge/Library-XGBoost-EB5E00?logo=xgboost)
![pandas](https://img.shields.io/badge/Library-Pandas-150458?logo=pandas)
![mlflow](https://img.shields.io/badge/Tool-MLflow-0194E2?logo=mlflow)
![dagshub](https://img.shields.io/badge/Cloud-DagsHub-FF6F20?logo=github)
![feast](https://img.shields.io/badge/FeatureStore-Feast-00A67E?logo=feast)
![catboost](https://img.shields.io/badge/Library-CatBoost-FFCC00?logo=catboost)
![lightgbm](https://img.shields.io/badge/Library-LightGBM-4CAF50?logo=leaf)
![gbc](https://img.shields.io/badge/Model-GradientBoostingClassifier-00B8D9)
![dtc](https://img.shields.io/badge/Model-DecisionTreeClassifier-795548)
![rfc](https://img.shields.io/badge/Model-RandomForestClassifier-2E7D32)
![loguru](https://img.shields.io/badge/Logger-Loguru-0A0F0B)
![evidently](https://img.shields.io/badge/Monitoring-Evidently-4B44CE)

---

## 📑 Índice

🧠 1. Problema y Objetivo del Proyecto
    - Contexto del problema
    - Objetivo del trabajo
    
✍️ 2. Descripción de los datasets

🌊 3. Project Flowchart
    - Diagrama de procesos

✉️ 4. Model Card:
    >>> Supervised Classification
    
🐈‍⬛  5. Estrategia de Git

📊 6. Anexos

💥 7. Resultados y Conclusiones

---

## 🧠 1. Problema y Objetivo del Proyecto
###     Contexto del problema:
### El presente muestra las problemáticas de clasificar y predecir, siendo primero las de clasificar a encuestados
### En este proyecto se presentan 4 datasets, pero dos tipos de problemas a resolver. En los primeros dos se ejecuta una solución de time series "Series de tiempo" (Serie de Tiempo.ipynb) mediante un ensamble como se pudo apreciar en las clases, se usó [Prophet, ARIMA, XGBOOST, ETS] y se procede con los demás puntos del código. Por otro lado, en la segunda parte del presente repositorio podemos visualizar la ejecución de una solución de clasificación supervisada para resolver las características de la gravedad del estrés en encuestados mediante el uso de los algoritmos como: BaggingClassifier(), RandomTreeClassifier(), VotingClassifier(), entre otros. Esto se muestra en el notebook "Ensambles.ipynb". Ambos datasets se encuentran disponibles dentro de la carpeta notebooks en la sección superior.

###    Objetivo del trabajo: 
###    En este proyecto, se trabaja con los conjuntos de datos Stress_Dataset.csv, StressLevelDataset.csv, Microsoft_Stock.csv, ma_lga_12345.csv (data.csv). Con los objetivos de desarrollar modelos que puedan ayudar a o predigan los valores a futuro en los datasets Microsoft_Stock.csv y ma_lga_12345.csv usando modelos de series de tiempo, para registrar valores de cierre de mercado para la acción de la empresa Microsoft ® y consecutivamente el mismo método para el precio (columna "MA") de ma_lga_12345.csv. En el caso de los otros datasets restantes se optó por aplicar una estrategia para poder clasificar y supervisar a alumnos/encuestados sobre sus niveles de estrés actual mediante una clasificación por algoritmos supervisados, donde se busca clasificar mediante etiquetas descritas en la última columna de estos datasets (Stress_Dataset.csv y StressLevelDataset.csv).
### Se buscó aplicar las estrategias de MLOps mediante las herramientas de Feast, Mlflow y Dagshub para su reproducibilidad y futuro trabajo en conjunto sobre modelos ya entrenados/guardados en estos ambientes. El propósito es practicar el uso de estas herramientas para agilizar y afianzar los procesos de desarrollo en un entorno de trabajo en conjunto.
### Para poder evaluar los procedimientos a realizar se aplicarán las métricas que se vieron de teoría; de clasificación como accuracy (proporción de aciertos), recall (capacidad de detectar positivos), F1-score (balance entre precisión y recall), classification report (resumen por clase) y confusion matrix (errores y aciertos por clase), que sirven para evaluar el desempeño de modelos supervisados; y métricas de clustering como Silhouette Score (cohesión y separación de clusters, valores cercanos a 1 son mejores), Davies-Bouldin Index (similaridad entre clusters, valores bajos son mejores), Calinski-Harabasz Index (separación inter vs intra clusters, valores altos son mejores), Adjusted Rand Index y Normalized Mutual Information (comparan clusters con etiquetas reales, siendo 1 la máxima concordancia), que permiten medir la calidad del agrupamiento en escenarios no supervisados.

---

## ✍️ 2. Descripción de los datasets

### 📌 Dataset: Student Stress Monitoring – StressLevelDataset.csv
    
 | Ítem                      | Detalle                                                                                                                                            |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fuente**                | [Página de Kaggle](https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets) Autor: Md Sultanul Islam Ovi             |
| **Licencia**              | CC BY-NC-SA 4.0 – uso académico con atribución                                                                                                     |
| **Filas utilizadas**      | All                                                                                                                                           |
| **Variable objetivo**     | `stress_level` – Nivel de estrés del estudiante (Low, Medium, High)                                                                                |
| **Familias de variables** | **Demográficas**: age, gender <br> **Hábitos**: sleep\_hours, study\_hours, physical\_activity <br> **Académicas**: exam\_pressure, academic\_load |
| **Valores faltantes**     | Ninguno reportado                                                                                                                                  |
| **Unidades**              | Horas, categorías y escalas ordinales                                                                                                              |

| Variable            | Descripción                                              | Tipo de dato | Familia de Variable |
| ------------------- | -------------------------------------------------------- | ------------ | ------------------- |
| `stress_level`      | Nivel de estrés del estudiante (`Low`, `Medium`, `High`) | object       | Etiqueta/objetivo   |
| `age`               | Edad del estudiante                                      | int          | Demográfica         |
| `gender`            | Género (`Male`, `Female`)                                | object       | Demográfica         |
| `sleep_hours`       | Horas de sueño promedio                                  | float        | Hábito              |
| `study_hours`       | Horas dedicadas al estudio                               | float        | Hábito              |
| `academic_load`     | Carga académica percibida                                | int          | Académica           |
| `exam_pressure`     | Nivel de presión por exámenes                            | int          | Académica           |
| `physical_activity` | Frecuencia de actividad física semanal                   | int          | Hábito              |

..... (8 of 21 columns)

### 📌 Dataset: Student Stress Monitoring – Stress_Dataset.csv

| Ítem                      | Detalle                                                                                                                                |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Fuente**                | [Página de Kaggle](https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets) Autor: Md Sultanul Islam Ovi |
| **Licencia**              | CC BY-NC-SA 4.0                                                                                                                        |
| **Filas utilizadas**      | All                                                                                                                                  |
| **Variable objetivo**     | `stress_category` – Clasificación del estrés (Academic, Health, Social)                                                     |
| **Familias de variables** | **Factores de estrés**: academic\_stress, financial\_stress, health\_stress, social\_support                                           |
| **Valores faltantes**     | Ninguno reportado                                                                                                                      |
| **Unidades**              | Escalas ordinales y categóricas                                                                                                        |

| Variable           | Descripción                                              | Tipo de dato | Familia de Variable |
| ------------------ | -------------------------------------------------------- | ------------ | ------------------- |
| `stress_category`  | Tipo principal de estrés (`Academic`, `Financial`, etc.) | object       | Etiqueta/objetivo   |
| `academic_stress`  | Grado de presión académica                               | int          | Factor académico    |
| `health_stress`    | Grado de estrés relacionado con salud                    | int          | Factor de salud     |
| `social_support`   | Nivel de apoyo social                                    | int          | Factor social       |

..... (4 of 26 columns)


## 🌊 3. Project Flowchart
### - Diagrama de procesos

>>> Clasificación
<img width="1473" height="791" alt="image" src="https://github.com/user-attachments/assets/14926916-1fac-4368-95aa-8efe4a951ef9" />



## Supervised Classification
## 🔹 Detalles
- **Autor:** Marco P  
- **Fecha de release:** 25, July 2025  
- **Versión:** 1.0  
- **Model Type:** Clasificación de etiquetas

---

## 🔹 Uso previsto
- **Finalidad principal:** Este modelo fue creado con el objetivo de **apoyar la predicción del tipo de estrés sufrido por estudiantes** (Estré, Eustress, Distress) basándose en tanto teoría y variables de mayor peso que otras, como contextuales.  
- **Usuarios esperados:** Está orientado a ser utilizado por **analistas de conducta, psicologos y estudiantes de ciencia médicas o aprendizaje automático apra clasificacion de encuestados**.  
- **Fuera de Alcance y Consideraciones Éticas:** Este modelo **no está diseñado para reemplazar el criterio médico profesional**, por lo que **no debe emplearse como único instrumento de diagnóstico clínico o de orientación psicológica**.  

---

## 🔹 Datos de Entrenamiento
- **Dataset utilizado:**  Student Stress Monitoring – StressLevelDataset.csv, Stressdataset.csv
- **Fuente:** [Student Stress Monitoring Datasets](https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets?select=Stress_Dataset.csv)  
- **Tamaño del dataset:**   
- **Preprocesamiento aplicado:** Aplicación de librería procesamientos.py, Normalización, eliminación de valores faltantes, división train/test  

---

## 🔹 Datos de Evaluación
- **Split de datos:** 80% entrenamiento / 20% prueba  
- **Balance de clases:**  
- **Conjunto de evaluación:**   

---

## 🔹 Métricas de Evaluación
- **Accuracy:**   
- **Recall:** 
- **Precision:** 
- **F1-score:** 
- **Confusion Matrix:** disponible en los experimentos registrados en MLflow  

---

## 🔹 Consideraciones Éticas
- Este modelo **puede estar sesgado** si se aplica a poblaciones con características distintas a las del dataset de entrenamiento.  
- **No sustituye diagnóstico médico**, solo es un apoyo analítico.  
- Uso indebido podría generar interpretaciones erróneas con impacto en la salud estudiantil.  

---





## 🐈‍⬛  5. Estrategia de Git
### Ramas: 
### main: Contiene el código de producción, estable
### add-ons: Agregados varios que se van desarrollando
### develop: Ejecución

## 📊 6. Anexos
## https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets?select=Stress_Dataset.csv
## https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets?select=StressLevelDataset.csv

## 💥 7. Resultados y Conclusiones








## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mle2 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mle2   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mle2 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

>>>>>>> c618bf4 (Primer commit del proyecto mle2)




