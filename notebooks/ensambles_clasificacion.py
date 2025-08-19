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




'''def proceso_epsilon():
    valor_muestreo_neighbor = input('Valor de muestreo para el modelo de vecinos')
    minimun_samples = valor_muestreo_neighbor
    neighbors = NearestNeighbors(n_neighbors=minimun_samples)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    
    # Ordenar las distancias al k-ésimo vecino
    distances = np.sort(distances[:, min_samples-1])
    plt.plot(distances)
    plt.ylabel("Distancia al {}-ésimo vecino".format(min_samples))
    plt.xlabel("Puntos ordenados")
    plt.show()

    kneighbors_eps = input('Valor aproximado para el epsilon en base al grafico:')
    dbscan = DBSCAN(eps=kneighbors_eps, min_samples=minimun_samples)
    labels = dbscan.fit_predict(X_scaled)
    

proceso_epsilon()
#procesar con dbscan, 18 clusters(numero de columnas)'''

#deberia usar calses para llamar a las funciones y solo depender de ingresar los argumentos?, en ves de usar clase pense usar funciones y llamarlas directamente


def experiment_definition(X_train, X_test, y_train, y_test, model=None, input_value="mean"):
    if model is None:
        model = input(
            "Que modelo desea aplicar?\n"
            "(1) Random Forest\n"
            "(2) Bagging\n"
            "(3) Voting\n"
            "(4) XGBoost\n"
            "(5) LGBM\n"
            "(6) Catboost\n"
        )

    models = {
        "1": ("Random Forest", RandomForestClassifier()),
        "2": ("Bagging", BaggingClassifier()),
        "3": ("Voting", VotingClassifier(estimators=[
            ("rf", RandomForestClassifier()),
            ("bag", BaggingClassifier()),
        ], voting="soft")),
        "4": ("XGBoost", XGBClassifier(max_depth=5, n_estimators=100)),
        "5": ("LGBM", LGBMClassifier()),
        "6": ("CatBoost", CatBoostClassifier(verbose=0)),
    }

    if model not in models:
        print("Opción inválida.")
        return

    run_name, algorithm = models[model]

    with mlflow.start_run(run_name=run_name):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=input_value)),
            (run_name, algorithm),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_params({"model": run_name, "imputer": input_value})
        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        print(f"{run_name} - Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return pipeline
    
'''def config_uri():
    uri = input("Introduce la URL del servidor MLflow: ")
    mlflow.set_tracking_uri(uri)
    print(f"Tracking URI configurado en: {uri}")



Aca quiero poner la ruta de contador de experimentos en el mismo repo, ejemplo con os

def crear_experimento_mlflow(nombre_experimento: str = None, ruta_contador="contador_experimentos.txt"):
    if os.path.exists(ruta_contador):
        with open(ruta_contador, "r") as f:
            ultimo_numero = int(f.read().strip())
    else:
        ultimo_numero = 0

    nuevo_numero = ultimo_numero + 1
    with open(ruta_contador, "w") as f:
        f.write(str(nuevo_numero))

    fecha_actual = datetime.now().strftime("%d/%m/%y")
    if nombre_experimento is None:
        nombre_experimento = f"Experimento - {nuevo_numero}, {fecha_actual}"

    try:
        exp_id = mlflow.create_experiment(nombre_experimento)
    except Exception:
        exp_id = mlflow.get_experiment_by_name(nombre_experimento).experiment_id

    print(f"Usando experimento '{nombre_experimento}' con ID {exp_id}")
    return exp_id'''


