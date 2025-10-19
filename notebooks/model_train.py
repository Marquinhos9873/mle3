import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (  GradientBoostingClassifier,
                                StackingClassifier,
                                BaggingClassifier,
                                DecisionTreeClassifier,
                                RandomForestClassifier, 
                                VotingClassifier,
                                AdaBoost        )
import hyperparams


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



    def evaluate_models(X_train, y_train,X_test,y_test,models,param, ):
        try:
            report = {}
    
            for i in range(len(list(models))):
                model = list(models.values())[i]
                para=param[list(models.keys())[i]]
    
                #gs = GridSearchCV(model,para,cv=3)
                #gs.fit(X_train,y_train)
    
                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)
    
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
    
                report[list(models.keys())[i]] = test_model_score

        return report
    
    def run_process(self, X_train, y_train, X_test, y_test):

        self.logger.info("Inicializando proceso del modelo...")
        self.logger.info(f"Tracking: {self.dagshub_repo_url}")
        self.logger.info(f'Experimento seteado como: {name_exp}_DSRP_mle3')
        mlflow.set_tracking_uri(f'{self.dagshub_repo_url}')
        mlflow.set_experiment(f"{name_exp}_DSRP_mle3")
        mlflow.create_experiment(f"{name_exp}_DSRP_mle3")




        models = {
            "LogisticRegression": LogisticRegression(**params),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Classifier": KNeighborsClassifier(**params),
            "Decision Tree Classifier": DecisionTreeClassifier(**params),
            "Random Forest Classifier": RandomForestClassifier(**params),
            "XGBClassifier": XGBClassifier(**params), 
            "CatBoosting Regressor": CatBoostClassifier(**params),
            "AdaBoost Classifier": AdaBoostClassifier(**params),
            "Gradient Boosting": GradientBoostingClassifier(**params),
            "LGBM": LGBMClassifier(**params)
            
        }
        
        
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
        logger.info("Reporte de Clasificación")
        print(classification_report(y_true = y_true, y_pred = y_pred, digits = 4, output_dict = True))
        report_dataframe_artifact = pd.DataFrame(report)
        
        
        run_name = f"{self.pipeline_process_name}_{self.current_date_experiment}_{self.current_time_experiment}"
        model = self.algoritmo_process(**self.hyperparams)
       


        mlflow.autolog(log_models=True)
        with mlflow.start_run(run_name=run_name):
          



            




        
        return 
        pass