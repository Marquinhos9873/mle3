import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (  StackingClassifier,
                                BaggingClassifier, 
                                RandomForestClassifier, 
                                VotingClassifier )


class Processing:
    
    def __init__(self, pipeline_process_name, algoritmo_process, hyperparams, name_exp = None, dagshub_repo_url = None):
        self.logger.info("Inicializando procesamiento...")
        self.pipeline_process_name = pipeline_process_name
        self.algoritmo_process = algoritmo_process
        self.hyperparams = hyperparams
        self.name_exp = name_exp or "Default_try_exp"
        self.dagshub_repo_url = dagshub_repo_url
        self.current_date_experiment = datetime.date.today().strftime("%Y%m%d")
        self.current_time_experiment = datetime.datetime.now().strftime("%H%M%S")
        
    
    def run_process(self, X_train, y_train, X_test, y_test):

        self.logger.info("Inicializando proceso del modelo...")
        self.logger.info(f"Tracking: {self.dagshub_repo_url}")
        self.logger.info(f'Experimento seteado como: {name_exp}_DSRP_mle3')
        mlflow.set_tracking_uri(f'{self.dagshub_repo_url}')
        mlflow.set_experiment(f"{name_exp}_DSRP_mle3")
        mlflow.create_experiment(f"{name_exp}_DSRP_mle3")
        
        
        run_name = f"{self.pipeline_process_name}_{self.current_date_experiment}_{self.current_time_experiment}"
        model = self.algoritmo_process(**self.hyperparams)
        pipeline = Pipeline(steps=[("model", model)])


        mlflow.autolog(log_models=True)
        with mlflow.start_run(run_name=run_name):
          



            




        
        pass