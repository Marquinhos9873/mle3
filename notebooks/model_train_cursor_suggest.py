import datetime

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    DecisionTreeClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

from reescalador import scaler_process


# ---------------------------------------------------------------------------
# Modelos y espacios de hiperparámetros por defecto
# ---------------------------------------------------------------------------

DEFAULT_MODELS = {
    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=1,
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, verbose=1),
    "XGBClassifier": XGBClassifier(
        tree_method="auto",
        objective="binary:logistic",
        eval_metric="logloss",
        verbosity=1,
        n_jobs=-1,
    ),
    "CatBoosting Classifier": CatBoostClassifier(
        task_type="GPU",
        devices="0",
        verbose=False,
        early_stopping_rounds=50,
    ),
}


DEFAULT_PARAMS = {
    "Decision Tree": {
        "criterion": ["gini", "log_loss"],
        "max_depth": [None, 10, 12, 16],
        "min_samples_split": [2, 3, 4, 5],
    },
    "Random Forest": {
        "n_estimators": [110, 115, 125, 130],
        "criterion": ["gini", "log_loss", "entropy"],
        "max_depth": [None, 5, 7, 8],
    },
    "Gradient Boosting": {
        "n_estimators": [110, 115, 125, 130],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.0045, 0.01, 0.05, 0.10],
        "min_samples_split": [2, 3, 4, 5],
    },
    "XGBClassifier": {
        "learning_rate": [0.0045, 0.01, 0.05, 0.10],
        "max_depth": [5, 6, 7],
        "n_estimators": [64, 128, 256],
        "tree_method": ["auto", "approx"],
    },
    "CatBoosting Classifier": {
        "iterations": [200, 400, 800],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "depth": [3, 4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"],
    },
}


# ---------------------------------------------------------------------------
# Función genérica para evaluar modelos + búsqueda de hiperparámetros
# ---------------------------------------------------------------------------

def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models=None,
    params=None,
    opt_strategy: str = None,
):
    """
    Entrena varios modelos, realiza búsqueda de hiperparámetros sencilla
    (GridSearchCV o RandomizedSearchCV), calcula métricas y las registra
    en MLflow.

    Devuelve:
        report: dict {nombre_modelo: accuracy_test}
        best_models: dict {nombre_modelo: estimator_entrenado}
    """

    if models is None:
        models = DEFAULT_MODELS
    if params is None:
        params = DEFAULT_PARAMS

    report = {}
    best_models = {}

    for name, model in models.items():
        logger.info(f"---- Procesando modelo: {name} ----")
        param_grid = params.get(name, {})

        search = None
        if param_grid:
            if opt_strategy == "grid_search":
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring="accuracy",
                    n_jobs=-1,
                )
            elif opt_strategy == "random_search":
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    cv=3,
                    scoring="accuracy",
                    n_iter=10,
                    n_jobs=-1,
                    random_state=42,
                )

        if search is not None:
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = search.best_score_
            logger.info(f"Mejores hiperparámetros ({name}): {best_params}")
            logger.info(f"Mejor score CV ({name}): {best_cv_score:.4f}")
        else:
            # Sin búsqueda de hiperparámetros
            best_estimator = model.fit(X_train, y_train)
            best_params = {}
            best_cv_score = None

        y_pred = best_estimator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        logger.info(f"Accuracy de test ({name}): {acc:.4f}")
        report[name] = acc
        best_models[name] = best_estimator

        # Logging básico a MLflow (asumiendo que ya hay un run abierto)
        mlflow.log_metric(f"{name}_test_accuracy", float(acc))
        if best_cv_score is not None:
            mlflow.log_metric(f"{name}_cv_accuracy", float(best_cv_score))
        if best_params:
            mlflow.log_params({f"{name}__{k}": v for k, v in best_params.items()})

        # Reporte de clasificación y matriz de confusión como artefactos
        cls_report_str = classification_report(y_test, y_pred, digits=4)
        logger.info(f"Classification report ({name}):\n{cls_report_str}")

        # Guardar el classification report como txt
        report_path = f"classification_report_{name.replace(' ', '_')}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(cls_report_str)
        mlflow.log_artifact(report_path)

        # Matriz de confusión
        confmat = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confmat)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax)
        ax.set_title(f"Confusion matrix - {name}")
        fig_path = f"confusion_matrix_{name.replace(' ', '_')}.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(fig_path)

    return report, best_models


# ---------------------------------------------------------------------------
# Clase principal de procesamiento
# ---------------------------------------------------------------------------


class Processing:
    """
    Clase orquestadora que:
      - Configura MLflow (tracking URI y experimento)
      - Divide en train/test
      - Ejecuta un proceso de reescalado con PCA (para análisis)
      - Llama a evaluate_models y registra resultados
    """

    def __init__(
        self,
        name_exp: str = "Default_try_exp",
        dagshub_repo_url: str | None = None,
        opt_strategy: str = "grid_search",
    ):
        self.name_exp = name_exp
        self.dagshub_repo_url = dagshub_repo_url
        self.opt_strategy = opt_strategy
        self.current_date_experiment = datetime.date.today().strftime("%Y%m%d")
        self.current_time_experiment = datetime.datetime.now().strftime("%H%M%S")

    def _setup_mlflow(self):
        if self.dagshub_repo_url:
            logger.info(f"Usando tracking URI: {self.dagshub_repo_url}")
            mlflow.set_tracking_uri(self.dagshub_repo_url)

        experiment_name = f"{self.name_exp}_DSRP_mle3"
        logger.info(f"Usando experimento: {experiment_name}")
        mlflow.set_experiment(experiment_name)

    def starting_process(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        models: dict | None = None,
        params: dict | None = None,
    ):
        """
        Punto de entrada principal.

        Parámetros:
            X, y: datos de entrada
            models, params: opcionalmente puedes pasar tus propios modelos
                            y espacios de hiperparámetros.
        """
        logger.info("Inicializando procesamiento de modelo...")
        self._setup_mlflow()

        run_name = f"{self.current_date_experiment}_{self.current_time_experiment}"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Shape X_train: {X_train.shape} | X_test: {X_test.shape}")
        logger.info(f"Shape y_train: {y_train.shape} | y_test: {y_test.shape}")

        # Proceso de reescalado + PCA solo para análisis y logging
        scaler_proc = scaler_process(
            data=pd.DataFrame(X_train),
            scale_method=StandardScaler(),
            pipeline_name="StandardScaler",
        )
        scaler_pipeline, metadata = scaler_proc.run_scaler()
        logger.info("Proceso de reescalado + PCA ejecutado.")
        logger.info(
            f"Varianza explicada por componentes PCA: {scaler_proc.explained_variance}"
        )

        # Log simple del número de componentes y varianza explicada
        if scaler_proc.explained_variance is not None:
            for i, var in enumerate(scaler_proc.explained_variance, start=1):
                mlflow.log_metric(f"PCA_component_{i}_var_explained", float(var))

        mlflow.autolog(log_models=True)
        logger.info("MLflow autolog activado.")

        with mlflow.start_run(run_name=run_name):
            report, best_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models or DEFAULT_MODELS,
                params=params or DEFAULT_PARAMS,
                opt_strategy=self.opt_strategy,
            )

            # Seleccionar mejor modelo
            best_model_name = max(report, key=report.get)
            best_model_score = report[best_model_name]

            logger.info(
                f"Mejor modelo: {best_model_name} "
                f"con accuracy de test = {best_model_score:.4f}"
            )

            mlflow.log_metric("best_model_test_accuracy", float(best_model_score))
            mlflow.log_param("best_model_name", best_model_name)

        return report, best_model_name, best_model_score, best_models


# Ejemplo de uso (puedes adaptarlo en tu notebook):
#
# from model_train import Processing, DEFAULT_MODELS, DEFAULT_PARAMS
# proc = Processing(
#     name_exp="MiExperimento",
#     dagshub_repo_url="https://TU_TRACKING_URI_O_NONE",
#     opt_strategy="grid_search",
# )
# report, best_name, best_score, best_models = proc.starting_process(
#     X=mis_features,
#     y=mi_target,
#     models=DEFAULT_MODELS,
#     params=DEFAULT_PARAMS,
# )