import mlflow
from datetime import datetime
import os

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier, BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler





class process_funcion(metrhod= 'DBSCAN')


#procesar con dbscan






def experiment_definition(X_train, X_test, y_train, y_test, model=None, input_value='mean'):

    if model is None:
        model = input("Que modelo desea aplicar?\n"
                             "(1) Random Forest\n"
                             "(2) Bagging\n"
                             "(3) Voting\n"
                             "(4) XGBoost\n"
                             "(5) LGBM\n"
                             "(6) Catboost\n")
    
    models = {
        "1": ("Random Forest", RandomForestClassifier()),
        "2": ("Bagging", BaggingClassifier()),
        "3": ("Voting", VotingClassifier(estimators=[
            ('rf', RandomForestClassifier()),
            ('bag', BaggingClassifier()),
            ('knb', KMeans())
        ], voting='soft')),
        "4": ("XGBoost", XGBClassifier(max_depth=5, n_estimators=100)),
        "5": ("LGBM", LGBMClassifier()),
        "6": ("CatBoost", CatBoostClassifier(verbose=0))
    }

    if model not in models:
        print("Opción inválida.")
        return

    run_name, algorithm = models[model]

    with mlflow.start_run(run_name=run_name):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=f"{input_value}")),
            (f"{run_name}", algorithm)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        print(f"{run_name} - Accuracy: {acc:.4f} | F1: {f1:.4f}")

def config_uri():
    uri = input("Introduce la URL del servidor MLflow: ")
    mlflow.set_tracking_uri(uri)
    print(f"Tracking URI configurado en: {uri}")
config_uri()

def crear_experimento_mlflow(ruta_contador="contador_experimentos.txt"):
    if os.path.exists(ruta_contador):
        with open(ruta_contador, "r") as f:
            ultimo_numero = int(f.read().strip())
    else:
        ultimo_numero = 0
    nuevo_numero = ultimo_numero + 1
    with open(ruta_contador, "w") as f:
        f.write(str(nuevo_numero))
  
    fecha_actual = datetime.now().strftime("%d/%m/%y")
    nombre_experimento = f"Experimento - {nuevo_numero}, {fecha_actual}"
    mlflow.create_experiment(nombre_experimento)

    return exp_name()


