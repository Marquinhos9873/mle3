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
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.preprocessing import StandardScaler
from feast import (
    Entity,
    FeatureStore,
    FeatureService,
    FeatureView,
    Field,
    FileSource
)
from loguru








# to_train_stresslevel = FeatureProcessor(datos = train_stresslevel, name_pipeline = 'stresslevel_pipeline')
# to_train_stresslevel.run(columnas_promedio = ('','','',''))


class FeatureProcessor:
    def __init__(self, datos: pd.DataFrame, name_pipeline: str):
        self.datos = datos
        self.name_pipeline = name_pipeline
        self.feature_table = None

    def scale(self, columnas: tuple[str, ...], components: int = 3) -> pd.DataFrame:
        pca = PCA(n_components = components)
        pipe = Pipeline(
            steps=[
                ("std_scaling", StandardScaler()),
                ("pca", PCA(n_components=components))
            ]
        )
        variance_ratio = pca.explained_variance_ratio
        print(f"Variance ratio: {variance_ratio}")
        #agregar aqui y devolver como variance_ratio en el return
        return pd.DataFrame(
            pipe.fit_transform(self.datos[columnas]),
            columns=[f"Pipe_feature{i+1}" for i in range(n_components)]
        )


#Una ves que se creen los pca_features agregarse al dataset final


      def run(self, columnas_promedio: tuple[str, ...], num_columnas: int) -> pd.DataFrame:
        #tengo un problema con el pyproject vim/nano
        #logger.info(f"Inicializando pipeline {self.name_pipeline}")
        numerics = self.scale()
        media_stress = (self.datos[list(columnas_promedio)].mean(axis=1))
        media_df = pd.DataFrame({"stress_exposure_mean": media_stress})
          
        modeling_dataset = pd.concat([numerics, stress_mean], axis=1)
        # Dataset Previo el pipeline
        pipe = Pipeline(
            steps=[
                ("feature_selection", VarianceThreshold()),
                ("scaling_robust", RobustScaler())
            ]
        )
        self.feature_table =  pd.DataFrame(
            pipe.fit_transform(modeling_dataset),
            columns=modeling_dataset.columns
        )
          

        return self.feature_table

    def write_feature_table(self, filepath: str) -> None:
        """Escribimos la feature table final para modelamiento
        """

        if not self.feature_table.empty: # -> True o False
            self.feature_table.to_parquet(f"{filepath}.parquet", index=False)
            self.feature_table.to_csv(f"{filepath}.csv", index=False)
        else:
            raise Exception("Ejecutar el comando .run()")            
        
        
        
class Metricsdeploy(variance_ratio):
    def pcavarianza(self, variance_ratio):
        return print(f'La varianza que se explica despues del PCA:{variance_ratio}')   
    def clusteringmetrics(self, X_scaled, labels):
        silhouette = silhouette_score(X_scaled, labels)
        dbi = davies_bouldin_score(X_scaled, labels)
        chi = calinski_harabasz_score(X_scaled, labels)
        ari = adjusted_rand_score(y, labels) # compara clusters con etiquetas reales
        nmi = normalized_mutual_info_score(y, labels)
        
        Sil = print(f"Silhouette Score: {silhouette:.3f}")
        Dbouldin = print(f"Davies-Bouldin Index: {dbi:.3f}")
        Charab = print(f"Calinski-Harabasz Index: {chi:.3f}")
        ADI = print(f"Adjusted Rand Index {ari:.3f}")
        NM = print(f"Normalized Mutual Info: {nmi:.3f}")
        return Sil, Dbouldin, Charab, ADI, NM 
    
    
    

class GuardadoFeature:




    


class Procesoexperimento:
    def proceso_epsilon(n_muestreo: int , ):
        #Numero de muestreo = valor definido para ver donde esta el alza en la varianza
        neighbors = NearestNeighbors(n_neighbors=n_muestreo)
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

        return
        
    

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
    








class ProcesoMLFLOW:
    '''def config_uri(self, uri: str):
    uri = input("Introduce la URL del servidor MLflow: ")
    mlflow.set_tracking_uri(uri)
    tracking = print(f"Tracking URI configurado en: {uri}")
    return tracking


Aca quiero poner la ruta de contador de experimentos en el mismo repo, ejemplo con os

def crear_experimento_mlflow(nombre_experimento: str, ruta_contador="contador_experimentos.txt"):
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


    