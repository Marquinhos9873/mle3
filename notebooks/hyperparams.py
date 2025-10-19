from hyperopt import fmin, tpe, space_eval, Trials, STATUS_OK, hp
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna





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
