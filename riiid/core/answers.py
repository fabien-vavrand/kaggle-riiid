import logging
import typing as t
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from riiid.config import FLOAT_DTYPE
from riiid.core.computation import compute_user_answers_ratio
from riiid.core.encoders import Smoother


class AnswersEncoder(Smoother):
    def __init__(self, smoothing_min=5, smoothing_value=1):
        super().__init__(smoothing_min, smoothing_value)

        self.priors = {}
        self.answers = {}

        self.features = None
        self.context = {}

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def transform(self, X):
        X = self._merge_context(X)
        return X

    def fit_transform(self, X, y=None):
        logging.info("- Fit answers encoding")
        X = self._fit(X, y)
        return X

    def _fit(self, X, y=None):
        total_answers, answers = self.get_answers_ratios(X)
        answers = answers.drop(columns=["is_correct"])

        # Merge with X
        X = pd.merge(X, answers, on=["content_id", "user_answer"], how="left")
        tasks = X.groupby(["user_id", "task_container_id"], sort=False)["answer_ratio"].mean().reset_index()

        # Get context
        self.priors = total_answers.to_dict(orient="index")
        self.answers = pd.pivot(answers, index="content_id", columns="user_answer", values="answer_ratio").to_dict(
            orient="index"
        )

        self.features = ["answer_ratio"]
        self.context = (
            tasks[["user_id"] + self.features]
            .drop_duplicates("user_id", keep="last")
            .set_index("user_id")
            .to_dict(orient="index")
        )

        tasks["prior_answer_ratio"] = tasks.groupby("user_id")["answer_ratio"].shift()
        tasks = tasks.drop(columns=["answer_ratio"])

        X = pd.merge(X, tasks, on=["user_id", "task_container_id"], how="left")
        X = X.drop(columns=["answer_ratio"])
        X["prior_answer_ratio"] = X["prior_answer_ratio"].astype(FLOAT_DTYPE)
        return X

    def get_answers_ratios(self, X):
        answers = X.groupby(["content_id", "correct_answer", "user_answer"]).size().reset_index(name="n_answers")
        total_answers = answers.groupby("content_id")["n_answers"].sum().reset_index(name="total_answers")
        answers = pd.merge(answers, total_answers, on="content_id")
        answers["answer_ratio"] = answers["n_answers"] / answers["total_answers"]
        answers["is_correct"] = answers["user_answer"] == answers["correct_answer"]

        # Compute priors
        total_answers = answers.groupby("is_correct")[["n_answers", "total_answers"]].sum()
        total_answers["prior"] = total_answers["n_answers"] / total_answers["total_answers"]
        total_answers = total_answers[["prior"]]

        # Compute weighted answer ratio
        answers = pd.merge(answers, total_answers, on="is_correct")
        answers["weight"] = self.smooth(answers["total_answers"])
        answers["answer_ratio"] = (
            answers["weight"] * answers["answer_ratio"] + (1 - answers["weight"]) * answers["prior"]
        )
        answers = answers[["content_id", "user_answer", "is_correct", "answer_ratio"]]

        return total_answers, answers

    def _merge_context(self, X):
        users = X["user_id"].values
        rows = users.shape[0]
        results = np.zeros(rows, dtype=np.float64)
        for r in range(rows):
            try:
                results[r] = self.context[users[r]]["answer_ratio"]
            except KeyError:
                results[r] = np.nan

        X["prior_answer_ratio"] = results
        return X

    def update(self, X, y=None):
        user_id = X["user_id"].values
        content_id = X["content_id"].values
        user_answer = X["user_answer"].values
        correct_answer = X["correct_answer"].values

        # We first compute the average answer_ratio by task_container_id (or by user as 1 user = 1 task)
        results = {}
        for r in range(user_id.shape[0]):
            uid = int(user_id[r])
            cid = int(content_id[r])
            ua = int(user_answer[r])
            try:
                ratio = self.answers[cid][ua]
                if np.isnan(ratio):
                    raise KeyError()
            except KeyError:
                ratio = self.priors[ua == correct_answer[r]]["prior"]

            if uid not in results:
                results[uid] = []
            results[uid].append(ratio)

        # Then we update users
        for uid, ratios in results.items():
            ratio = float(np.mean(ratios))
            try:
                context = self.context[uid]
                context["answer_ratio"] = ratio
            except KeyError:
                self.context[uid] = {"answer_ratio": ratio}

        return self

    def update_transform(self, X, y=None):
        self.update(X, y)
        return X

    def get_user_context(self, user_id):
        return list(self.context[user_id].values())

    def set_user_context(self, user_id, context):
        self.context[user_id] = {f: v for f, v in zip(self.features, context)}


class IncorrectAnswersEncoder:
    def __init__(self, cv=None, noise=None):
        self.cv = cv
        self.noise = noise
        self.columns = [f"answer_{a}_ratio" for a in [2, 3, 4]]
        self.answers_ratios = None
        self.imputer = None

    def fit(self, X, y=None):
        logging.info("- Fit incorrect answers encoding")
        answers_ratios = self._fit(X)

        # Fit imputer
        self.imputer = SimpleImputer(strategy="mean")
        self.imputer.fit(answers_ratios[self.columns])

        answers_ratios = answers_ratios.to_dict(orient="index")
        self.answers_ratios = {content_id: list(ratios.values()) for content_id, ratios in answers_ratios.items()}
        return self

    def transform(self, X):
        content_id = X["content_id"].values
        results = np.full((len(X), 3), np.nan)
        for r in range(len(X)):
            try:
                results[r, :] = self.answers_ratios[content_id[r]]
            except KeyError:
                pass

        for i, column in enumerate(self.columns):
            X[column] = results[:, i]

        X[self.columns] = self.imputer.transform(X[self.columns])
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)

        if self.cv is None:
            answers_ratios = self._fit(X)
            X = self._transform(X, answers_ratios, noise=self.noise)
        else:
            results = []
            for fold, (train, test, noise) in enumerate(self.cv):
                answers_ratios = self._fit(X.loc[train])
                sub_results = self._transform(X.loc[test], answers_ratios, noise=self.noise if noise else None)
                results.append(sub_results)

            results = pd.concat(results, axis=0)
            X = results.loc[X.index, :]
            self.cv = None

        X[self.columns] = self.imputer.transform(X[self.columns])
        X[self.columns] = X[self.columns].astype(FLOAT_DTYPE)
        return X

    def _fit(self, X):
        _, answers_ratios = AnswersEncoder(smoothing_min=10, smoothing_value=2).get_answers_ratios(X)

        # We compute the ratios for incorrect answers. Contain the information of the number of valid answers
        answers_ratios = answers_ratios[answers_ratios["is_correct"] == False].sort_values(
            ["content_id", "answer_ratio"], ascending=[True, False]
        )
        answers_ratios["user_answer"] = answers_ratios.groupby("content_id")["user_answer"].transform(
            lambda x: x.rolling(window=5, min_periods=1).count()
        )
        answers_ratios = pd.pivot(
            answers_ratios, index="content_id", columns="user_answer", values="answer_ratio"
        ).fillna(0)
        answers_ratios.columns = self.columns
        return answers_ratios

    def _transform(self, X, answers_ratios, noise=None):
        X = pd.merge(X, answers_ratios, left_on="content_id", right_index=True, how="left").set_index(X.index)
        # Implement noise if requires
        return X

    def update(self, X, y=None):
        return self

    def update_transform(self, X, y=None):
        return X


class UserAnswersFrequencyEncoder(Smoother):
    def __init__(self, decay=0.99, smoothing_min=5, smoothing_value=1):
        super().__init__(smoothing_min, smoothing_value)
        self.decay = decay
        self.answers_ratios = None
        # Each user context contains 4 answers frequency + 4 answers expected frequency
        self.context: t.Dict[int, t.List[float]] = {}

    def fit(self, X, y=None):
        logging.info("- Fit incorrect answers encoder")
        self._fit(X)
        return self

    def transform(self, X):
        user_id = X["user_id"].values
        correct_answer = X["correct_answer"].values
        results = np.zeros((len(X), 2))
        for r in range(len(X)):
            try:
                context = self.context[user_id[r]]
                results[r, 0] = context[correct_answer[r]]
                results[r, 1] = context[4 + correct_answer[r]]
            except:
                pass
        X = self._transform(X, results)
        return X

    def fit_transform(self, X, y=None):
        logging.info("- Fit user answers frequency encoder")
        X, results = self._fit(X)
        X = self._transform(X, results)
        X = self._downcast(X)
        return X

    def _fit(self, X):
        _, answers_ratios = AnswersEncoder(smoothing_min=10, smoothing_value=2).get_answers_ratios(X)
        answers_ratios = pd.pivot(
            answers_ratios, index="content_id", columns="user_answer", values="answer_ratio"
        ).fillna(0)

        X = pd.merge(X, answers_ratios, left_on="content_id", right_index=True, how="left")
        results, users, contexts = compute_user_answers_ratio(X, self.decay)
        X = X.drop(columns=[0, 1, 2, 3])

        # Save context
        for user, context in zip(users, contexts):
            self.context[user] = list(map(float, context))

        # Transform ratios to dict
        answers_ratios = answers_ratios.to_dict(orient="index")
        self.answers_ratios = {content_id: list(ratios.values()) for content_id, ratios in answers_ratios.items()}

        return X, results

    def _transform(self, X, results):
        scores = results[:, 0] / results[:, 1]
        if self.is_smooting_activated:
            np.nan_to_num(scores, copy=False)
            weight = self.smooth(results[:, 1]) * (results[:, 1] > 0)
            X["user_answer_frequency"] = weight * scores + (1 - weight) * 1
        else:
            X["user_answer_frequency"] = scores
        return X

    def _downcast(self, X):
        X["user_answer_frequency"] = X["user_answer_frequency"].astype(FLOAT_DTYPE)
        return X

    def update(self, X, y=None):
        user_id = X["user_id"].values
        content_id = X["content_id"].values
        user_answer = X["user_answer"].values
        decayed = set()

        for r in range(len(X)):
            try:
                context = self.context[user_id[r]]
            except KeyError:
                context = [0] * 8
                self.context[user_id[r]] = context
            try:
                ratios = self.answers_ratios[content_id]
            except KeyError:
                ratios = [0.25] * 4  # should not occur on full dataset

            # decay user only once
            if user_id[r] not in decayed:
                decayed.add(user_id[r])
                context = context * self.decay

            context[user_answer[r]] += 1
            for i in range(4, 8):
                context[i] += ratios[i]
            self.context[user_id[r]] = context

        return self

    def update_transform(self, X, y=None):
        self.update(X)
        return X

    def get_user_context(self, user_id):
        return self.context[user_id]

    def set_user_context(self, user_id, context):
        self.context[user_id] = context
