import os
import io
import gc
import pickle
import zipfile
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from riiid.utils import downcast_int


def save_pkl(data, path, zip=False):
    if not zip:
        bdata = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip:
            zip.writestr('object.pkl', pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        bdata = buffer.getvalue()
    with open(path, 'wb') as file:
        file.write(bdata)


def load_pkl(path, zip=False):
    if not zip:
        with open(path, 'rb') as file:
            return pickle.load(file)
    else:
        with zipfile.ZipFile(path, 'r') as zip:
            return pickle.loads(zip.read('object.pkl'))


class DataLoader:

    def __init__(self, path):
        self.path = path

    def load(self):
        logging.info('Loading data')
        cache_path = os.path.join(self.path, 'train.pkl')
        if os.path.exists(cache_path):
            train, questions, lectures = load_pkl(cache_path)
        else:
            train, questions, lectures = self._load()
            save_pkl((train, questions, lectures), cache_path)
        return train, questions, lectures

    def load_first_users(self, n):
        logging.info(f'Loading first {n} users data')
        cache_path = os.path.join(self.path, 'train_{}_users.pkl'.format(n))
        if os.path.exists(cache_path):
            train, questions, lectures = load_pkl(cache_path)
        else:
            train, questions, lectures = self.load()
            if n > 0:
                users = train['user_id'].unique()[:n]
            else:
                users = train['user_id'].unique()[n:]
            train = train[train['user_id'].isin(users)].reset_index(drop=True)
            save_pkl((train, questions, lectures), cache_path)
        return train, questions, lectures

    def load_tests(self, name):
        return load_pkl(os.path.join(self.path, name))

    def load_tests_examples(self):
        return load_pkl(os.path.join(self.path, 'tests_examples.pkl'))

    def _load(self, nrows=None):
        train_types = {
            #'row_id': 'int64',
            'timestamp': 'int64',
            'user_id': 'int32',
            'content_id': 'int16',
            'content_type_id': 'int8',
            'task_container_id': 'int16',
            'user_answer': 'int8',
            'answered_correctly': 'int8',
            'prior_question_elapsed_time': 'float32',
            'prior_question_had_explanation': 'boolean'
        }
        questions_types = {
             'question_id': 'int16',
             'bundle_id': 'int16',
             'correct_answer': 'int8',
             'part': 'int8',
             'tags': 'object'
        }
        lectures_types = {
             'lecture_id': 'int16',
             'tag': 'int16',
             'part': 'int8',
             'type_of': 'object'
        }

        train = self._read_csv('train.csv', train_types, nrows)
        questions = self._read_csv('questions.csv', questions_types)
        lectures = self._read_csv('lectures.csv', lectures_types)
        return train, questions, lectures

    def _read_csv(self, filename, filetypes, nrows=None):
        return pd.read_csv(os.path.join(self.path, filename), usecols=filetypes.keys(), dtype=filetypes, nrows=nrows, low_memory=False)

    def load_first_rows(self, nrows):
        return self._load(nrows)


def preprocess_questions(questions):
    questions['tags'] = questions['tags'].fillna('0')
    questions['question_tags'] = OrdinalEncoder().fit_transform(questions[['tags']])
    questions['question_tags'] = downcast_int(questions['question_tags'])
    questions['question_tag'] = questions['tags'].apply(lambda x: int(x.split(' ')[0]))
    def compute_communities(x):
        import networkx as nx
        import community as community_louvain

        questions = x['question_id'].values
        tags_split = x['tags'].apply(lambda x: [int(num) for num in str(x).split() if num != 'nan'])
        adjacency_matrix = np.zeros((questions.shape[0], questions.shape[0]))
        tag_questions = {}

        for tags, question in zip(tags_split, questions):
            for tag in tags:
                if tag in tag_questions:
                    tag_questions[tag].append(question)
                else:
                    tag_questions[tag] = [question]

        for questions in tag_questions.values():
            n = len(questions)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    adjacency_matrix[questions[i], questions[j]] += 1
        G = nx.from_numpy_matrix(adjacency_matrix)
        partitions = community_louvain.best_partition(G, random_state=42)
        df_partitions = pd.Series(partitions).rename('question_community')
        x = x.merge(df_partitions, how='left', left_on='question_id', right_index=True)
        return x

    questions = compute_communities(questions)
    questions['question_tag'] = downcast_int(questions['question_tag'])

    questions.rename(columns={'question_id': 'content_id', 'part': 'question_part', 'tags_id': 'question_tags'}, inplace=True)
    return questions


def preprocess_lectures(lectures):
    lectures.rename(columns={'part': 'lecture_part', 'tag': 'lecture_tag'}, inplace=True)
    lectures['type_of'] = lectures['type_of'].map({
        'concept': 0,
        'solving question': 1,
        'intention': 2,
        'starter': 2
    })
    lectures['type_of'] = downcast_int(lectures['type_of'])
    return lectures


class TagsFactory:

    def __init__(self, questions):
        questions = questions.copy()
        questions['tags'] = questions['tags'].apply(lambda x: set(x.split(' ')))
        tags = questions[['question_tags', 'tags']].drop_duplicates('question_tags')
        x_tags = pd.merge(tags.assign(key=0), tags.assign(key=0), on='key').drop(columns='key')
        x_tags['common'] = x_tags.apply(lambda r: len(r['tags_x'].intersection(r['tags_y'])) / len(r['tags_x'].union(r['tags_y'])), axis=1)

        self.questions_to_tags = questions.set_index('content_id')['question_tags'].to_dict()
        self.tags_similarity = x_tags.set_index(['question_tags_x', 'question_tags_y'])['common'].to_dict()

    def similarity(self, content_id_1, content_id_2):
        tag_1 = self.questions_to_tags[content_id_1]
        tag_2 = self.questions_to_tags[content_id_2]
        return self.tags_similarity[(tag_1, tag_2)]
