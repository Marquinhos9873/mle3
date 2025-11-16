
from hyperopt import (fmin,
                      tpe,
                      space_eval,
                      Trials,
                      STATUS_OK,
                      hp)
from hpsklearn import (HyperoptGridSearchCV,
                      HyperoptRandomizedSearchCV, 
                      HyperoptTune,
                      HyperoptEstimator
                      )
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import optuna



tpe_search_space = {
        "loss": hp.choice("loss", ["log_loss", "exponential"]) ,
        "learning_rate": hp.normal("learning_rate", 0.1,0.01 ),
        "n_estimators": hp.quniform("n_estimators", 10, 100, 10),
        "min_samples_split": hp.quniform("min_samples_split", 2, 10 ,1),
        "max_depth": hp.quniform("max_depth", 3, 20, 1)
    }
    



#-----------------------------------------------------------------------------------------
class HyperparamTuning:
 
    def __init__(self, opt_strategy: str, search_space, algorithm ) -> None:
        self.strategy = opt_strategy
        self.search_space = search_space
        self.algorithm = algorithm



    def optuna_tunning(self, X_train, y_train, X_test, y_test, models, params):
        params = {

        "loss": trial.suggest_categorical("loss", ["log_loss", "exponential"]),
        "max_depth": trial.suggest_int("max_depth", 5, 20)
        
        }

        classifier = GradientBoostingClassifier(**params)
        classifier.fit(X_train_clf, y_train_clf)
        predictions = classifier.predict(X_test_clf)
        _accuracy = accuracy_score(y_test_clf, predictions)


  
    def objective(params):
        # params -> accuracy -> {mas alto posible}
    
        adj_params = {
             "loss": params["loss"] ,
            "learning_rate": params["learning_rate"],
            "n_estimators": int(params["n_estimators"]),
            "min_samples_split": int(params["min_samples_split"]),
            "max_depth": int(params["max_depth"])
        }
        classifier = GradientBoostingClassifier(**adj_params)
        classifier.fit(X_train_clf, y_train_clf)
        predictions = classifier.predict(X_test_clf)
        _accuracy = accuracy_score(y_test_clf, predictions)
        
        return {
            "loss": 1 - _accuracy,
            "status": STATUS_OK
        }



    
 '''   
 
        with mlflow.start_run(run_name="tpe_hyperopt") as run:
     
    
        trials = Trials()
        best   = fmin(
            fn=objective,
            space=tpe_search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials,
        )
    
        best_params = space_eval(tpe_search_space, best)
        best_params = {
                 "loss": best_params["loss"] ,
                "learning_rate": best_params["learning_rate"],
                "n_estimators": int(best_params["n_estimators"]),
                "min_samples_split": int(best_params["min_samples_split"]),
                "max_depth": int(best_params["max_depth"])
            }
        
        classifier = GradientBoostingClassifier(**best_params)
        classifier.fit(X_train_clf, y_train_clf)
        predictions = classifier.predict(X_test_clf)
        logger.info(f"Best Model accuracy {accuracy_score(y_test_clf, predictions)}")
    
        mlflow.log_metric("accuracy", accuracy_score(y_test_clf, predictions))
        mlflow.log_params(best_params)
        
'''

#-----------------------------------------------------------------------------------------



class AIOptimizer:

    def __init__(self, opt_strategy: str, search_space, algorithm ) -> None:
        self.strategy = opt_strategy
        self.search_space = search_space
        self.algorithm = algorithm

    def optimize(self):

        if self.strategy == "grid_search":
            gs_classifier = GridSearchCV(
                estimator=self.algorithm, 
                param_grid=self.search_space, 
                cv=3,
                scoring="accuracy"
            )
            gs_classifier.fit(X_train_clf, y_train_clf)
            logger.info(f"Best Score {gs_classifier.best_score_}")
            logger.info(f"Best Params {gs_classifier.best_params_}")
            
            grid_search_artifac = {
            'Used model' : f'{self.algorithm}',
            'Best score' : f'{gs_classifier.best_score_}',
            'Best Params' : f'{gs_classifier.best_params_}'
             }
            
            mlflow.log_artifac(grid_search_artifac)
            
            return (
                gs_classifier.best_estimator_, 
                gs_classifier.best_params_, 
                gs_classifier.best_score_
            )
        elif self.strategy == "random_search":
            rs_classifier = RandomizedSearchCV(
                estimator=self.algorithm, 
                param_distributions=self.search_space, 
                cv=3,
                scoring="accuracy",
                n_iter=5
            )
            rs_classifier.fit(X_train_clf, y_train_clf)
            logger.info(f"Best Score {rs_classifier.best_score_}")
            logger.info(f"Best Params {rs_classifier.best_params_}")

            random_search_artifac = {
            'Used model' : f'{self.algorithm}',
            'Best score' : f'{gs_classifier.best_score_}',
            'Best Params' : f'{gs_classifier.best_params_}'
             }
            
            mlflow.log_artifac(random_search_artifac)
            
            return (
                rs_classifier.best_estimator_, 
                rs_classifier.best_params_, 
                rs_classifier.best_score_
            )






class xgbopt:
    def init(self, model, params, GPU_use : int):
        self.model = model
        self.params = params
        self.GPU_use = GPU_use

    def tunning(self, trial):
        params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        }
        if GPU_use == 1:
        model = xgb.XGBClassifier(
            **params,
            device = "cuda"
        )
        elif GPU_use == 0:
        model = xgb.XGBClassifier(
            **params,
        )
        else:
        except Exception as e:
            print(f"Error en evaluate_models: {e}")
            return None

        
        score = cross_val_score(model, X, y, cv=3, scoring="roc_auc").mean()
        mlflow.log_artifac("cross_val_score XBGCcuda", score)
        return score        
            
   def set_opt_study()     
        study = optuna.create_study(study_name="xgboost_study_cuda", direction="maximize")
        study.optimize(objective_gpu, n_trials=10, show_progress_bar=True, n_jobs=-1)
        
        # Retrieve the best parameter values
        best_params = study.best_params
        print(f"\nBest parameters: {best_params}")


























        