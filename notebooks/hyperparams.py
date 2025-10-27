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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna


def bymodel_search_space(model):
    if model == "Random Forest":
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    elif model == "XGBoost":
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    elif model == "LGBM":
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    elif model == "CatBoost":
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    
    elif model == "QuadraticDiscriminantAnalysis":
        return {
            "reg_param": [0.1, 0.5, 1.0]
        }
    elif model == "SVC_linear":
        return {
            "C": [0.025, 0.1, 0.5]
        }
    elif model == "SVC_rfb":
        return {
            "gamma": [2, 5, 10]
        }
    elif model == "Gaussian_Process_Classifier":
        return {
            "alpha": [0.1, 0.5, 1.0]
        }
    elif model == "MLP_Classifier":
        return {
            "alpha": [0.1, 0.5, 1.0]
        }
    elif model == "GaussianNB":
        return {
            "alpha": [0.1, 0.5, 1.0]
        }
    elif model == "QuadraticDiscriminantAnalysis":
        return {
            "reg_param": [0.1, 0.5, 1.0]
        }














class HyperparamTuning:
 
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
            return (
                rs_classifier.best_estimator_, 
                rs_classifier.best_params_, 
                rs_classifier.best_score_
            )

#-----------------------------------------------------------------------------------------
with mlflow.start_run(run_name="gridsearch") as run:
    
    # Espacio de busqueda
    gridsearch_params = {
        "loss": ("log_loss", "exponential"),
        "learning_rate": [0.1,  0.5],
        "n_estimators": [10, 100]
    }
    
    optimizer = AIOptimizer(
        opt_strategy="grid_search",
        search_space=gridsearch_params,
        algorithm=classifier
    )
    _, params, score = optimizer.optimize()

    mlflow.log_metric("accuracy", score)
    mlflow.log_params(params)

#-----------------------------------------------------------------------------------------
    
    tpe_search_space = {
        "loss": hp.choice("loss", ["log_loss", "exponential"]) ,
        "learning_rate": hp.normal("learning_rate", 0.1,0.01 ),
        "n_estimators": hp.quniform("n_estimators", 10, 100, 10),
        "min_samples_split": hp.quniform("min_samples_split", 2, 10 ,1),
        "max_depth": hp.quniform("max_depth", 3, 20, 1)
    }
    
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
        
    
    #-----------------------------------------------------------------------------------------
    
    
    
    
    
    
    
    
