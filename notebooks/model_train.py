import pandas as pd
import numpy as np
import datetime
import mlflow
from loguru import logger
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import (classification_report,
                            confusion_matrix,
                            ConfusionMatrixDisplay,
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
import interpretabilidad as inter

class Processing:
    
    def __init__(self, algoritmo_process, hiperparams, name_exp = None, dagshub_repo_url = None):
        self.logger.info("Inicializando procesamiento...")
        self.algoritmo_process = algoritmo_process
        self.hiperparams = hiperparams
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
        run_name = f"{self.current_date_experiment}_{self.current_time_experiment}"
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
                  
      
        mlflow.autolog(log_models=True)
        with mlflow.start_run(run_name=run_name):
        pipeline_process = Pipeline([
            ("scaler", self.scale_method),
            (   ,  ),
            (   ,  )
            ])
       







    

    
        
            

    def define_models(self):
          
        models = {
                "Random Forest": RandomForestClassifier( verbose = 1, **params),
                "Decision Tree": DecisionTreeClassifier(**params),
                "Gradient Boosting": GradientBoostingClassifier( verbose = 1, **params),
                "XGBClassifier": XGBClassifier(device ='cuda', verbosity = '1', **params),
                "CatBoosting Classifier": CatBoostClassifier(task_type='GPU', devices='0', early_stopping_rounds = 50)
            }

        params={
                "Decision Tree": {
                    'criterion': ['gini','log_loss'],
                    'max_depth': [0, 10, 12, 16],
                    'min_samples_split':[2, 3, 4, 5],
                                   
                },

                "Random Forest":{
                    'n_estimators' : [110 , 115, 125, 130],
                    'criterion' : ['gini' ,'log_loss', 'entropy'],
                    'max_depth' : [None, 5, 7 , 8],

                },

                "Gradient Boosting":{
                    'n_estimators' : [110 , 115, 125, 130],
                    'max_depth' : [None, 5, 7 , 8],
                    'learning_rate':[0.0045 , 0.01, 0.05, 0.10],
                    'min_samples_split':[2, 3, 4, 5],
                },
                
                "XGBClassifier":{
                    'learning_rate':[0.0045 , 0.01, 0.05, 0.10],
                    'max_depth': [5, 6, 7],
                    'n_estimators': [256, 128, 64, 12],
                    'tree_method': ['auto', 'approx']
                },

                "CatBoosting Classifier":{
                    'iterations': [200, 400, 800],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1],
                    'depth': [3, 4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS']
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
        mlflow.log_artifact(confmat)
        mlflow.log_artifact(disp)
        mlflow.log_artifact(best_model)
        mlflow.log_metrics(f'{best_model_name}': best_model_score)


        logger.info("Reporte de Clasificación en proceso")
        cls_report = classification_report(y_true = y_train, y_pred = y_pred, digits = 4, output_dict = True)
        print(cls_report)
        confmat = confusion_matrix(y_true = y_true, y_pred = y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix = confmat)
        disp.plot()
        plt.show()
        


        return models, params, best_model_name, best_model_score
    
    
    def evaluate_models(X_train, y_train, X_test, y_test, models, params, opt_strategy):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]
                


                ###Integracion de soft hp tuning para reporte de resultados

                gs_opt = AIOptimizer(opt_strategy="grid_search", search_space=para, algorithm=model)
                rs_opt = AIOptimizer(opt_strategy="random_search", search_space=para, algorithm=model)
                gs_opt.optimize()
                rs_opt.optimize()
                grid_model = gs_opt.best_estimator_
                random_model = rs_opt.best_estimator_
                grid_model.fit(X_train,y_train)
                random_model.fit(X_train,y_train)
                
                

                grid_model = model.set_params(**gs.best_params_)
                random_model = model.set_params(**rs.best_params_)
                grid_model.fit(X_train,y_train)
                random_model.fit(X_train,y_train)

                y_train_pred = grid_model.predict(X_train)
                y_train_pred_random = random_model.predict(X_train)
                y_test_pred = grid_model.predict(X_test)
                y_test_pred_random = random_model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            
        except Exception as e:
            print(f"Error en evaluate_models: {e}")
            return None

        return report   


    











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