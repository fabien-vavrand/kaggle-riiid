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
            users = train['user_id'].unique()[:n]
            train = train[train['user_id'].isin(users)].reset_index(drop=True)
            save_pkl((train, questions, lectures), cache_path)
        return train, questions, lectures

    def load_tests(self, name):
        return load_pkl(os.path.join(self.path, name))

    def load_tests_examples(self):
        return load_pkl(os.path.join(self.path, 'tests_examples.pkl'))

    def _load(self):
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

        train = self._read_csv('train.csv', train_types)
        questions = self._read_csv('questions.csv', questions_types)
        lectures = self._read_csv('lectures.csv', lectures_types)
        return train, questions, lectures

    def _read_csv(self, filename, filetypes):
        return pd.read_csv(os.path.join(self.path, filename), usecols=filetypes.keys(), dtype=filetypes, low_memory=False)


def preprocess_questions(questions):
    def count_tags(x):
        if pd.isnull(x):
            return 0
        else:
            x = x.split(' ')
            return len(x)

    def get_first_tag(x):
        if pd.isnull(x):
            return 0
        else:
            x = x.split(' ')
            return int(x[0])

    def get_two_tags(x):
        if pd.isnull(x):
            return []
        else:
            x = x.split(' ')
            return x[:2]

    questions['n_tags'] = questions['tags'].apply(count_tags).astype(np.int8)
    questions['question_tag'] = questions['tags'].apply(get_first_tag)
    questions['question_tag'] = downcast_int(questions['question_tag'])
    questions['tags'] = questions['tags'].fillna('')
    questions['tags'] = OrdinalEncoder().fit_transform(questions[['tags']])
    questions['tags'] = downcast_int(questions['tags'])
    questions.rename(columns={'question_id': 'content_id', 'part': 'question_part', 'tags': 'question_tags'}, inplace=True)
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
