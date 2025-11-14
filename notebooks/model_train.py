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
    
    def __init__(self, algoritmo_process, hyperparams, name_exp = None, dagshub_repo_url = None):
        self.logger.info("Inicializando procesamiento...")
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
        
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
                  
      
        mlflow.autolog(log_models=True)
        with mlflow.start_run(run_name=run_name):
        







    

    def evaluate_models(X_train, y_train, X_test, y_test, models):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]
                


                ###Integracion de hyperparametros estrategia
                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

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
                "XGBClassifier": XGBClassifier(device ='cuda', verbosity = '1', ),
                "CatBoosting Classifier": CatBoostClassifier(task_type='GPU', devices='0', early_stopping_rounds = 50),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

        params={
                "Decision Tree": {
                    'criterion': ['gini','log_loss'],
                    'max_depth': [0, 10, 12, 16],
                    
                
                },

                "Random Forest":{
                    
                },

                "Gradient Boosting":{
                    
                },
                
                "XGBClassifier":{
                    'learning_rate':[0.0045 , 0.01, 0.05, 0.10],
                    'max_depth': [5, 6, 7],
                    'n_estimators': [256, 128, 64, 12],
                    'tree_method': ['auto', 'approx']
                },

                "CatBoosting Classifier":{
                    
                },

                "AdaBoost Classifier":{
                    
                }
                
            }

        ### 'learning_rate': np.logspace(-3, -0.4, 15) , log distribución de valores, probar


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
        return None
    


    ''' CUDA toolkit (2.3gb aaaa) 
    def interpretability():
        if model == "XGBoostClassifier":

            explainer = shap.explainers.GPUTree(model, X)
            shap_values = explainer(X)
            barplot_shap = shap.plots.bar(shap_values)
            waterfall_plot_shap = shap.plots.waterfall(shap_values[0])

        return barplot_shap, waterfall_plot_shap
        
        mlflow.log_article(barplot_shap)
        mlflow.log_article(waterfall_plot_shap)
    '''