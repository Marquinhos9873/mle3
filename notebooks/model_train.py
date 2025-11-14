import pandas as pd
import numpy as np
import datetime
import mlflow
from loguru import logger
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import (classification_report,
                             accuracy_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (  GradientBoostingClassifier,
                                DecisionTreeClassifier,
                                RandomForestClassifier, 
                                AdaBoostClassifier)


from interpret import show



import hyperparams as hparams
import reescalador as rescaler
import interpretabilidad as inter

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


        

    def run_process(self, X_train, y_train, X_test, y_test, params):

        self.logger.info("Inicializando proceso del modelo...")
        self.logger.info(f"Tracking: {self.dagshub_repo_url}")
        self.logger.info(f'Experimento seteado como: {name_exp}_DSRP_mle3')
        mlflow.set_tracking_uri(f'{self.dagshub_repo_url}')
        mlflow.set_experiment(f"{name_exp}_DSRP_mle3")
        mlflow.create_experiment(f"{name_exp}_DSRP_mle3")
        
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        
        model_train.evaluate_models(self.X_train, self.y_train, self.X_test, self.y_test)
        main_pipeline = Pipeline(
            steps = [(      ),
                     (       ),
                     (       ),
                     (       ),
                       
            ]
        )



       
        mlflow.autolog(log_models=True)
        with mlflow.start_run(run_name=run_name):




























    

    def evaluate_models(X_train, y_train, X_test, y_test, models):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]
                


                ### Integracion de hyperparametros estrategia
                #gs = GridSearchCV(model,para,cv=3)
                #gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report
        
        except Exception as e:
            print(f"Error en evaluate_models: {e}")
            return None
        
        
            

    def define_models(self, params):
          
        models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

        params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                                     'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
        
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
        
        
        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

        best_model = models[best_model_name]


        logger.info("Reporte de Clasificación")
        cls_report = classification_report(y_true = y_true, y_pred = y_pred, digits = 4, output_dict = True)
        print(cls_report)
        report_dataframe_artifact = pd.DataFrame(report)
        
        
        run_name = f"{self.pipeline_process_name}_{self.current_date_experiment}_{self.current_time_experiment}"
        model = self.algoritmo_process(**self.hyperparams)
       
    


          



            




        
        return 
        pass