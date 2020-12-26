import logging
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.base import TransformerMixin
from sklearn.model_selection import GroupKFold

from riiid.config import FLOAT_DTYPE
from riiid.core.utils import pre_filtered_indexed_merge
from riiid.cache import get_cache_manager


logging.getLogger("gensim.models.base_any2vec").setLevel(logging.WARNING)
logging.getLogger("gensim.models.word2vec").setLevel(logging.WARNING)


class QuestionsEmbedder(TransformerMixin):
    def __init__(self, questions, n_clusters=30, embedding_size=40, window=5, min_count=5, sg=0, iter=30, workers=1):
        self.questions = questions
        self.n_clusters = n_clusters
        self.size = embedding_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.iter = iter
        self.workers = workers
        self.kmean_n_init = 100

    def fit(self, X, y=None):
        logging.info("- Fit questions embedding")

        sentences = X.groupby("user_id")["content_id"].apply(lambda a: a.values).values
        sentences = [list(map(str, s)) for s in sentences]

        cache_id = "questions_embedder_{}_{}_{}_{}_{}_{}_sentences".format(
            self.size, self.window, self.min_count, self.sg, self.iter, len(sentences)
        )

        logging.info("fitting Word2Vec")
        if get_cache_manager().exists(cache_id):
            model = get_cache_manager().load(cache_id)
        else:
            model = Word2Vec(
                sentences=sentences,
                size=self.size,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg,
                iter=self.iter,
                workers=self.workers,
            )
            get_cache_manager().save(model, cache_id)

        wm = np.zeros((len(model.wv.vocab), model.vector_size))
        for i, word in enumerate(model.wv.vocab):
            wm[i, :] = model.wv[word]

        logging.info("fitting K-means")
        cache_id = "questions_kmean_{}_{}_{}_sentences".format(self.n_clusters, self.kmean_n_init, len(sentences))

        if get_cache_manager().exists(cache_id):
            kmeans = get_cache_manager().load(cache_id)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.kmean_n_init, random_state=123).fit(wm)
            get_cache_manager().save(kmeans, cache_id)

        clusters = pd.DataFrame(data={"content_id": list(model.wv.vocab.keys()), "question_category": kmeans.labels_})
        clusters["content_id"] = pd.to_numeric(clusters["content_id"], downcast="signed")
        self.questions = pd.merge(self.questions, clusters, on="content_id", how="left")

        missing_categories = self._log_questions_with_missing_cluster()
        if missing_categories > 0:
            self._fill_missing_category_with_most_frequent("question_tag")
            self._log_questions_with_missing_cluster()
            self._fill_missing_category_with_most_frequent("question_part")
            self._log_questions_with_missing_cluster()

        self.questions = self.questions[["content_id", "question_category"]]
        self.questions["question_category"] = self.questions["question_category"].astype(np.int8)
        self.questions = self.questions.set_index("content_id")
        return self

    def _log_questions_with_missing_cluster(self):
        n = pd.isnull(self.questions["question_category"]).sum()
        logging.info(f"{n} questions with no category")
        return n

    def _fill_missing_category_with_most_frequent(self, column):
        # Find most frequent category by tag for missing questions
        most_frequent = self.questions.groupby([column, "question_category"]).size().reset_index(name="n_questions")
        most_frequent = most_frequent.sort_values([column, "n_questions"], ascending=[True, False])
        most_frequent = most_frequent.drop_duplicates(column, keep="first")[[column, "question_category"]]
        most_frequent.columns = [column, "most_frequent_category"]

        self.questions = pd.merge(self.questions, most_frequent, on=column, how="left")
        self.questions["question_category"] = self.questions.apply(
            lambda x: x["question_category"] if not pd.isnull(x["question_category"]) else x["most_frequent_category"],
            axis=1,
        )
        self.questions = self.questions.drop(columns=["most_frequent_category"])

    def transform(self, X):
        X = pre_filtered_indexed_merge(X, self.questions, left_on="content_id")
        return X


class AnswersCorrectnessEmbedder:
    def __init__(self, n_fold=5, embedding_size=20, window=3, min_count=5, sg=0, iter=15, workers=1):
        self.n_fold = n_fold
        self.size = embedding_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.iter = iter
        self.workers = workers

        self.model = None
        self.lags = [1, 2]
        self.context = {}

    def fit(self, X, y=None):
        logging.info("- Fit answers embedding")
        self.model = self._fit(X)
        self._build_context(X)
        return self

    def transform(self, X):
        user_id = X["user_id"].values
        content_id = X["content_id"].values
        similarities = np.full((len(user_id), len(self.lags)), np.nan, dtype=FLOAT_DTYPE)
        for r in range(len(user_id)):
            try:
                context = self.context[user_id[r]]
                for lag in self.lags:
                    try:
                        sim_correct = self._get_similarity(
                            self.model, context[-lag][0], content_id[r], context[-lag][1], 1
                        )
                        sim_incorrect = self._get_similarity(
                            self.model, context[-lag][0], content_id[r], context[-lag][1], 0
                        )
                        similarities[r, lag - 1] = sim_correct - sim_incorrect
                    except:
                        pass
            except KeyError:
                pass

        for lag in self.lags:
            X[f"similarity_{lag}"] = similarities[:, lag - 1]

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)

        cv = GroupKFold(n_splits=self.n_fold).split(X, groups=X["user_id"])
        results = []
        for fold, (train, test) in enumerate(cv):
            model = self._fit(X.iloc[train], fold=fold)
            sub_results = self._transform(X.iloc[test], model, fold=fold)
            results.append(sub_results)

        results = pd.concat(results, axis=0)
        results = results.loc[X.index, :]
        X = pd.concat([X, results], axis=1)
        return X

    def _fit(self, X, fold=None):
        content_correct = X["content_id"] * (X["answered_correctly"] == 1).replace({0: -1})
        sentences = content_correct.groupby(X["user_id"]).apply(lambda a: a.values).values
        sentences = [list(map(str, s)) for s in sentences]

        cache_id = "answers_embedder_{}_{}_{}_{}_{}_{}_{}_sentences".format(
            self.n_fold, self.size, self.window, self.min_count, self.sg, self.iter, len(sentences)
        )

        if fold is not None:
            cache_id += "_fold_{}".format(fold)

        if get_cache_manager().exists(cache_id):
            model = get_cache_manager().load(cache_id)
        else:
            model = Word2Vec(
                sentences=sentences,
                size=self.size,
                window=self.window,
                min_count=self.min_count,
                iter=self.iter,
                sg=self.sg,
                workers=self.workers,
            )
            get_cache_manager().save(model, cache_id)

        return model

    def _transform(self, X, model, fold=None):
        cache_id = "answers_embedder_transformed_{}_{}_{}_{}_{}_{}_{}".format(
            self.n_fold, self.size, self.window, self.min_count, self.sg, self.iter, len(X)
        )

        if fold is not None:
            cache_id += "_fold_{}".format(fold)

        if get_cache_manager().exists(cache_id):
            similarities = get_cache_manager().load(cache_id)
        else:
            similarities = pd.concat([self._compute_similarity(X, model, lag) for lag in self.lags], axis=1)
            similarities = similarities.set_index(X.index)
            get_cache_manager().save(similarities, cache_id)

        return similarities

    def _compute_similarity(self, X, model, lag):
        user_id = X["user_id"].values
        content_id = X["content_id"].values
        task_container_id = X["task_container_id"].values
        answered_correctly = X["answered_correctly"].values
        similarities = np.full(len(user_id), np.nan, dtype=FLOAT_DTYPE)
        current_user_id = -1
        for r in range(len(user_id)):
            if current_user_id != user_id[r]:
                current_user_id = user_id[r]
                current_task_container_id = -1
                user_content_id = []
                user_answered_correctly = []
                user_task_content_id = []
                user_task_answered_correctly = []
            if current_task_container_id != task_container_id[r]:
                current_task_container_id = task_container_id[r]
                user_content_id.extend(user_task_content_id)
                user_answered_correctly.extend(user_task_answered_correctly)
                user_task_content_id = []
                user_task_answered_correctly = []
            if len(user_content_id) >= lag:
                sim_correct = self._get_similarity(
                    model, user_content_id[-lag], content_id[r], user_answered_correctly[-lag], 1
                )
                sim_incorrect = self._get_similarity(
                    model, user_content_id[-lag], content_id[r], user_answered_correctly[-lag], 0
                )
                similarities[r] = sim_correct - sim_incorrect

            user_task_content_id.append(content_id[r])
            user_task_answered_correctly.append(answered_correctly[r])

        results = pd.DataFrame({f"similarity_{lag}": similarities}, dtype=FLOAT_DTYPE)
        return results

    def _get_similarity(self, model, previous_content, content, previous_answered_correctly, answered_correctly):
        previous_content = "-" + str(previous_content) if previous_answered_correctly == 0 else str(previous_content)
        content = "-" + str(content) if answered_correctly == 0 else str(content)
        try:
            return model.wv.similarity(previous_content, content)
        except:
            return np.nan

    def _build_context(self, X):
        X = X.groupby("user_id").tail(self.lags[-1])
        user_id = X["user_id"].values
        content_id = X["content_id"].values
        answered_correctly = X["answered_correctly"].values
        for r in range(len(user_id)):
            self._add_to_context(user_id[r], content_id[r], answered_correctly[r])

    def _add_to_context(self, user_id, content_id, answered_correctly):
        user_id = int(user_id)
        if user_id not in self.context:
            self.context[user_id] = []
        context = self.context[user_id]
        if len(context) >= self.lags[-1]:
            context.pop(0)
        context.append([int(content_id), int(answered_correctly)])

    def update(self, X, y=None):
        user_id = X["user_id"].values
        content_id = X["content_id"].values
        answered_correctly = X["answered_correctly"].values
        for r in range(len(user_id)):
            self._add_to_context(user_id[r], content_id[r], answered_correctly[r])

    def update_transform(self, X, y=None):
        self.update(X, y)
        return X

    def get_user_context(self, user_id):
        return self.context[user_id]

    def set_user_context(self, user_id, context):
        self.context[user_id] = context
