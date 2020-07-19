import os
from logging import getLogger
from pathlib import PosixPath
from typing import Tuple, Union
from copy import deepcopy
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score, f1_score, make_scorer, mean_squared_error

from .configs import (
    ALLOWED_TOPICS,
    DATA_CONFIG,
    FACEBOOK_QUERY,
    MODEL_CONFIG,
    TWITTER_QUERY,
)

logger = getLogger(__name__)
logger.setLevel(1)


def pearsonr_score(y_true, y_pred) -> float:
    try:
        value = pearsonr(y_true, y_pred)
        return value[0]
    except TypeError:
        value = pearsonr(y_true.reshape(-1), y_pred)
        return value[0]
    except Exception as e:
        logger.error(
            f"pearsonr_score, {e.__class__.__name__}: {e}\n" +
            f"type(y_true): {type(y_true)}\n" +
            f"type(y_pred): {type(y_pred)}\n"
        )
        return pd.np.nan


def get_base_model(target: str) -> GridSearchCV:
    """Creates the unfit model for a given target based on the configs.MODEL_CONFIG

    :param target: str
        must be a key in configs.MODEL_CONFIG
    :return: GridSearchCV

    """
    try:
        model_type = MODEL_CONFIG[target]["model_type"]
    except KeyError as e:
        raise e

    pipe = [("scaler", StandardScaler()), ("pca", PCA())]

    if model_type == "regression":
        pipe.append(("model", SVR(kernel="rbf")))
        scoring = make_scorer(r2_score, greater_is_better=True)
        param_grid = {
            "pca__n_components": np.linspace(0.8, 1., 21),
            "model__C": np.linspace(0.001, 2., 25),
        }
    elif model_type == "classification":
        pipe.append(("model", SVC(kernel="rbf")))
        scoring = make_scorer(f1_score, greater_is_better=True, needs_threshold=True)
        param_grid = {
            "pca__n_components": np.linspace(0.85, 1., 16),
            "model__C": np.linspace(10., 250., 25),
        }
    else:
        raise NotImplementedError(f"Unsupported model_type: {model_type}")

    model = GridSearchCV(
        Pipeline(deepcopy(pipe)),
        param_grid=param_grid,
        n_jobs=os.cpu_count() - 1,
        cv=10,
        scoring=scoring,
    )
    return model


def get_feature_extractors(df: pd.DataFrame, target: str) -> Tuple[ColumnTransformer, ColumnTransformer]:
    """Creates an x and y feature extractor

    :param df: pd.Dataframe
    :param target: str
    :return: Tuple[ColumnTransformer, ColumnTransformer]
        X ColumnTransformer
        y ColumnTransformer

    """
    try:
        feature_regex = MODEL_CONFIG[target]["feature_regex"]
    except KeyError as e:
        raise e

    feature_list = [i for i in df.columns if feature_regex.match(i) is not None]
    logger.info(f"Model Features: {feature_list}")
    x_extractor = ColumnTransformer([("Select_Features", "passthrough", feature_list)], remainder="drop")
    y_extractor = ColumnTransformer([("Select_Target", "passthrough", [target])], remainder="drop")
    return x_extractor, y_extractor


def get_metrics(target: str):
    try:
        model_type = MODEL_CONFIG[target]["model_type"]
    except KeyError as e:
        raise e

    if model_type == "regression":
        return {"r2": r2_score, "pearsonr": pearsonr_score}
    elif model_type == "classification":
        return {"accuracy": accuracy_score, "f1": f1_score}
    else:
        raise NotImplementedError(f"Unsupported model type: {model_type}")


def load_data(target: str, k: int, with_domain: bool, data_dir: PosixPath) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and test data sets

    :param target: str
    :param k: int
        must be in configs.ALLOWED_TOPICS
    :param with_domain: bool
    :return: Tuple[pd.DataFrame, pd.DataFrame]
        df_train
        df_test

    """
    assert k in ALLOWED_TOPICS
    if with_domain:
        test = DATA_CONFIG["with_domain"]["test"].format(k=k)
        train = DATA_CONFIG["with_domain"]["train"].format(k=k)
    else:
        test = DATA_CONFIG["without_domain"]["test"].format(k=k)
        train = DATA_CONFIG["without_domain"]["train"].format(k=k)

    try:
        query = MODEL_CONFIG[target]["filter_query"]
    except KeyError as e:
        raise e

    df_test = pd.read_csv(data_dir / test)
    df_train = pd.read_csv(data_dir / train)
    if len(query) > 0:
        df_test = df_test.query(query)
        df_train = df_train.query(query)

    return df_train, df_test


class Experiment:

    def __init__(self, target: str, k: int, with_domain: bool, data_dir: PosixPath):
        self.target = target
        self.k = k
        self.with_domain = with_domain
        self.data_dir = data_dir
        self.df_train, self.df_test = self.data()
        self.model = get_base_model(self.target)
        self.metric_funcs = get_metrics(self.target)
        self.metrics = {"Train": {}, "Test": {}}
        self.x_extractor = None
        self.y_extractor = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.p_train = None
        self.p_test = None

    def __str__(self):
        return (
            f"Experiment(target={self.target!r}, " +
            f"k={self.k}, with_domain={self.with_domain}, data_dir={self.data_dir})"
        )

    def data(self):
        return load_data(self.target, self.k, self.with_domain, self.data_dir)

    def calc_metrics(self):
        for k, v in self.metric_funcs.items():
            self.metrics["Train"][k] = v(self.y_train, self.p_train)
            self.metrics["Test"][k] = v(self.y_test, self.p_test)
            logger.info(f"Training Performance: ({k}, {self.metrics['Train'][k]})")
            logger.info(f"Test Performance: ({k}, {self.metrics['Test'][k]})")

    def run(self):
        self.x_extractor, self.y_extractor = get_feature_extractors(self.df_train, self.target)
        self.x_train = self.x_extractor.fit_transform(self.df_train)
        self.y_train = self.y_extractor.fit_transform(self.df_train)
        self.x_test = self.x_extractor.transform(self.df_test)
        self.y_test = self.y_extractor.transform(self.df_test)
        self.model.fit(self.x_train, self.y_train)
        self.p_train = self.model.predict(self.x_train)
        self.p_test = self.model.predict(self.x_test)
        self.calc_metrics()


class UserLevelExperiment(Experiment):

    def data(self):
        df_train, df_test = load_data(self.target, self.k, self.with_domain, self.data_dir)
        df_train = df_train.groupby("user_id").mean()
        df_test = df_test.groupby("user_id").mean()
        return df_train, df_test

class UserLevelTrainOnFacebookExperiment(Experiment):

    def data(self):
        df_train, df_test = load_data(self.target, self.k, self.with_domain, self.data_dir)
        df_train = df_train.query(FACEBOOK_QUERY).groupby("user_id").mean().reset_index(drop=False).assign(is_fb=1)
        df_test = df_test.groupby(["user_id", "is_fb"]).mean().reset_index(drop=False)
        return df_train, df_test[df_train.columns]

    def calc_metrics(self):
        is_fb = self.df_test["is_fb"].values == 1
        p_test_fb = self.p_test[is_fb]
        p_test_tw = self.p_test[~is_fb]
        y_test_fb = self.y_test[is_fb]
        y_test_tw = self.y_test[~is_fb]
        for k, v in self.metric_funcs.items():
            self.metrics["Train"][k] = v(self.y_train, self.p_train)
            self.metrics["Test"][k] = {
                "Twitter": v(y_test_tw, p_test_tw),
                "Facebook": v(y_test_fb, p_test_fb),
            }
            logger.info(f"Training Performance: ({k}, {self.metrics['Train'][k]})")
            logger.info(f"Test Facebook Performance: ({k}, {self.metrics['Test'][k]['Facebook']})")
            logger.info(f"Test Twitter Performance: ({k}, {self.metrics['Test'][k]['Twitter']})")


class UserLevelTrainOnTwitterExperiment(UserLevelTrainOnFacebookExperiment):

    def data(self):
        df_train, df_test = load_data(self.target, self.k, self.with_domain, self.data_dir)
        df_train = df_train.query(TWITTER_QUERY).groupby("user_id").mean().reset_index(drop=False).assign(is_fb=0)
        df_test = df_test.groupby(["user_id", "is_fb"]).mean().reset_index(drop=False)
        return df_train, df_test[df_train.columns]
