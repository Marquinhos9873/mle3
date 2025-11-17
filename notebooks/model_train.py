import pandas as pd
import numpy as np
import datetime
import mlflow
import shap
import interpret 
import lime
from loguru import logger

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, RegressionPreset, ClassificationPreset
from evidently.metrics import *



from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import (   GridSearchCV,
                                        RandomizedSearchCV,
                                        train_test_split)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report,
                            confusion_matrix,
                            ConfusionMatrixDisplay,
                            accuracy_score)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from interpret import show
import reescalador as rsc

import interpretabilidad as inter


class Processing:
    
    def __init__(self, pipeline_name, algoritmo_process, hiperparams, name_exp = None, dagshub_repo_url = None, set_model = None, set_params = None   ):
        self.logger.info("Inicializando procesamiento...")
        self.set_model = set_model
        self.hiperparams = hiperparams
        self.name_exp = name_exp or "Default_try_exp"
        self.dagshub_repo_url = dagshub_repo_url
        self.current_date_experiment = datetime.date.today().strftime("%Y%m%d")
        self.current_time_experiment = datetime.datetime.now().strftime("%H%M%S")
        self.pipeline_name = pipeline_name

        

    def starting_process(self, X, y):

        self.logger.info("Inicializando proceso del modelo...")
        self.logger.info(f"Tracking: {self.dagshub_repo_url}")
        self.logger.info(f'Experimento seteado como: {name_exp}_DSRP_mle3')
        mlflow.set_tracking_uri(f'{self.dagshub_repo_url}')
        mlflow.set_experiment(f"{name_exp}_DSRP_mle3")
        mlflow.create_experiment(f"{name_exp}_DSRP_mle3")
        run_name = f"{self.current_date_experiment}_{self.current_time_experiment}"
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Datos divididos en train y test")
        logger.info(f"Datos de train: {X_train.shape}")
        logger.info(f"Datos de test: {X_test.shape}")
        logger.info(f"Datos de train: {y_train.shape}")
        logger.info(f"Datos de test: {y_test.shape}")
        
        explained_variance_process = rsc.scaler_process(data = X_train, scale_method = StandardScaler(), pipeline_name = "StandardScaler")          
        explained_variance_process.run_scaler()
        logger.info("Explained variance process started")
        logger.info(f"Explained variance: {explained_variance_process.explained_variance}")
        logger.info(f"Explained variance ratio: {explained_variance_process.explained_variance_ratio}")
        logger.info(f"Explained variance process: {explained_variance_process.explained_variance_process}")
        
      
        mlflow.autolog(log_models=True)
        logger.info("Mlflow started")
        with mlflow.start_run(run_name=run_name):
            self.pipeline_name = Pipeline(steps=[
            ( "scaler", StandardScaler()),
            ( "Modelfitting", set_model(**set_params).fit(X_train, y_train)),
            ( "Predicting", set_model.predict(X_test)),
            ("Permutation_importance", ),
            ])
       



        
            

    def define_models(self, models, params):



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
        
        best_model_indicator = {'{best_model_name}': best_model_score}
        
        mlflow.log_artifact(best_model_indicator)


        logger.info("Reporte de Clasificación en proceso")
        cls_report = classification_report(y_true = y_test, y_pred = y_pred, digits = 4, output_dict = True)
        print(cls_report)
        confmat = confusion_matrix(y_true = y_test, y_pred = y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix = confmat)
        disp.plot()
        plt.show()
        


        return best_model_name, best_model_score


    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
    def evaluate_models(X_train, y_train, X_test, y_test, models, params, opt_strategy):
        try:
            
            if models not in list(models.values()):
                raise ValueError(f"Model {model} not found in models")
            else:
                pass



            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]
                


                ###Integracion de soft hp tuning para reporte de resultados

                gs_opt = AIOptimizer(opt_strategy="grid_search", search_space=para, algorithm=model)
                rs_opt = AIOptimizer(opt_strategy="random_search", search_space=para, algorithm=model)
                gs_opt.optimize()
                rs_opt.optimize()
                
                

                

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


    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

          
    models = {
        "Random Forest": RandomForestClassifier(verbose=1, **params["Random Forest"]),
        "Decision Tree": DecisionTreeClassifier(**params["Decision Tree"]),
        "Gradient Boosting": GradientBoostingClassifier(verbose=1, **params["Gradient Boosting"]),
        "XGBClassifier": XGBClassifier(device='cuda', verbosity=1, **params["XGBClassifier"]),
        "CatBoosting Classifier": CatBoostClassifier(task_type='GPU', devices='0', early_stopping_rounds=100, **params["CatBoosting Classifier"])
    }
    
    params={"Decision Tree": {'criterion': ['gini','log_loss'],
                                'max_depth': [0, 10, 12, 16],
                                'min_samples_split':[2, 3, 4, 5]},
    
            "Random Forest":{'n_estimators' : [110 , 115, 125, 130],
                                'criterion' : ['gini' ,'log_loss', 'entropy'],
                                'max_depth' : [None, 5, 7 , 8]},
    
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
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class Monitoring:
    def __init__(self):

        pass
    def evidently_local_monitoring():
        CMAPPING = ColumnMapping()
        CMAPPING.target = "target"




        classification_performance_report = Report(metrics=[ClassificationPreset()])
        classification_performance_report.run(reference_data=X_train, current_data=X_test)
        classification_performance_report
        mlflow.log_artifact(classification_performance_report)
    pass    

class Interprete:
    def __init__(self):


        pass

    def interpret_model(self):


        pass

    def shap_model(self, X_test):
        shap_explainer = shap.Explainer(best_model_name)
        shap_values = shap_explainer.shap_values(X_test)
        shap_sum = shap.summary_plot(shap_values=shap_values, features=X_test, feature_names=FEATURES)
        shap_sum
        mlflow.log_artifact(shap_sum)

        pass

    def lime_model(self):



        pass





