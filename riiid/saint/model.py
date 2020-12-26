import datetime
import numpy as np
import pandas as pd


def classifyLagTime(lag_time_seconds):
    return int(min(lag_time_seconds, 300))


# train_set['lag_class'] = train_set.lag_time_seconds.apply(classifyLagTime)

# qdf = pd.read_csv("../data/questions.csv",usecols=['question_id','part'])
# train_set = train_set.merge(qdf,how="left",left_on="content_id",right_on="question_id")


# train_set = train_set.drop(dev_set.index)

# dev_set = pd.read_csv("./data/dev_sakt_top.csv",
#                     usecols = data_types_dict.keys(),dtype=data_types_dict)
# train_set = train_set[~train_set.row_id.isin(dev_set.row_id.values)]

N_t = 100
# step = 50
N_questions = 13523


class SaintModel:
    def __init__(self, questions, lectures, length=100):
        self.questions = questions
        self.lectures = lectures
        self.length = length
        self.pad_token = -1
        self.model_id: str = None
        self.metadata = {}

    def get_name(self, prefix=None):
        if prefix:
            return "saint_{}_{}".format(self.model_id, prefix)
        else:
            return "saint_{}.zip".format(self.model_id)

    def fit(self, X):
        self._init_fit(X)

        X = X[X["content_type_id"] == 0]
        X.replace([np.nan], 0, inplace=True)
        X = pd.merge(X, self.questions, on="content_id", how="left")

        user_ids = X["user_id"].unique()
        train_uids = user_ids[:-39000]
        dev_uids = user_ids[-39000:]

        train_set = X[X["user_id"].isin(train_uids)]
        dev_set = X[X["user_id"].isin(dev_uids)]

        train_ftrs = self.create_features(train_set)
        dev_ftrs = self.create_features(dev_set)

        return self

    def _init_fit(self, X):
        self.model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata["n_rows"] = len(X)
        self.metadata["n_users"] = X["user_id"].nunique()

    def create_features(self, df):
        features = ["content_id", "answered_correctly", "part", "bundle_id"]
        n_f = len(features)
        df = df[df.content_type_id == 0]
        udf = df.groupby("user_id")
        N_samples = df.user_id.nunique()
        #     ftr_arr = np.zeros((N_samples,N_t,n_f))
        ftrs = {}
        for i, feature in enumerate(features):
            cids = udf[feature].apply(lambda x: x.values)
            ftrlst = []

            for j, cid in enumerate(cids.values):
                n_int = len(cid)
                if n_int < self.length:
                    pad_width = self.length - n_int
                    cid = np.pad(cid, (pad_width, 0), mode="constant", constant_values=self.pad_token)
                elif n_int > self.length:
                    mod = n_int % self.length
                    if mod != 0:
                        cid = np.pad(cid, (self.length - mod, 0), mode="constant", constant_values=self.pad_token)
                cid = cid.reshape((-1, self.length))
                ftrlst.append(cid)
            #             ftr_arr[j,:,i] = cid
            ftr_arr = np.concatenate(ftrlst)
            ftrs[feature] = ftr_arr
        #    ftrs['dec_in'] = ftrs['answered_correctly']
        return ftrs

    def create_features_strided(self, df, train=True, step=50):
        features = ["content_id", "answered_correctly", "lag_class", "elapsed_time", "prior_question_had_explanation"]
        n_f = len(features)
        df = df[df.content_type_id == 0]
        udf = df.groupby("user_id")
        N_samples = df.user_id.nunique()
        #     ftr_arr = np.zeros((N_samples,N_t,n_f))
        if not train:
            step = self.length
        ftrs = {}
        for i, feature in enumerate(features):
            cids = udf[feature].apply(lambda x: x.values)
            ftrlst = []

            for j, cid in enumerate(cids.values):
                n_int = len(cid)
                tmplst = []

                for i in range(0, n_int, step):
                    atmp = cid[i : i + self.length].reshape(1, -1)
                    if atmp.shape[1] < self.length:
                        atmp = np.pad(atmp, ((0, 0), (self.length - atmp.shape[1], 0)))
                    tmplst.append(atmp)

                cid = np.concatenate(tmplst)
                ftrlst.append(cid)
            #             ftr_arr[j,:,i] = cid
            print(ftrlst[:2])

            ftr_arr = np.concatenate(ftrlst)
            ftrs[feature] = ftr_arr
        ftrs["dec_in"] = ftrs["content_id"] + N_questions * ftrs["answered_correctly"]
        return ftrs

    def create_user_data(self, df):
        df = df[df.content_type_id == 0]
        df = df[["content_id", "user_id", "answered_correctly", "part"]]
        udf = df.groupby("user_id").tail(self.length)
        udf = udf.groupby("user_id")
        vals = udf.apply(lambda x: x.values)
        udata = {}
        for uid, v in vals.iteritems():
            udata[uid] = [v[:, 0], v[:, 2], v[:, 3]]
            ud = udata[uid]
            nin = ud[0].shape[0]
            if nin > self.length:
                for i in range(3):
                    ud[i] = ud[i][-self.length :]
            elif nin < self.length:
                diff = self.length - nin
                for i in range(3):
                    ud[i] = np.pad(ud[i], (diff, 0), mode="constant", constant_values=self.pad_token)
            udata[uid] = ud
        return udata

    def predict(self, X):
        return 0

    def update(self, X):
        return self
