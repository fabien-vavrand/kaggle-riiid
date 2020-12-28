
import os
import logging
from riiid.core.data import DataLoader, preprocess_questions, preprocess_lectures, save_pkl, load_pkl
from riiid.core.model import RiiidModel
from riiid.validation import merge_test
from riiid.utils import configure_console_logging
from riiid.config import PATH, INPUT_PATH, MODELS_PATH, PARAMS


"""
TO TEST:
- in sorted_rolling_score, sum and count should not be decremented when rolling in the middle of a task

TODO: 
- update answers count in AnswersEncoder
- sum of prior_question_lost_time sur la session
- average et sum content_time par user, par session
- prior_question_time: check if it is better to compute it for the first question of a session or not ? (group by session_id in tasks)
- sum questions explanation time during session
- start session with lecture / session lecture time/learning time (time question lost)
- retester n_lectures_on_tag, maybe I had a bug previosuly
- user score by day (even if not easy to compute at prediction time)
- ratio question_part, question_tag / category per user (user had 45% of this question part... etc)
- apply a decay factor to lectures (n_lectures)
- precompute dictionnary of weights to speed inference
- try DBSCAN instead of k-means
- time since last session
- changer save with source pour save directement model/context.txt et model/model.pkl au lieu du zip
- raffiner les buckets des tasks_bucket_12
- les premières questions sont souvent les mêmes!
- encoder par previous_question_tag / previous_answered_correctly / content_id
- user_id content_id had explanations count

FEATURES TO TRY
- time since first content_id
- number of questions during last day / week / month
- time between n-1 and n-2 content_id
- % of tags in common with previous questions
- count lectures by tag (removed it because it was too weak but now...)
- Mean and std of content_id_encoded by part
- lecture_tag_content_id_encoded

TRAIN 10000 users
- Best score = 78.85%, in 305 iterations
- Best score = 78.89%, in 290 iterations
- Best score = 78.92%, in 435 iterations
- Best score = 78.97%, in 412 iterations (add decay on user and user/question_part)
- Best score = 78.95%, in 291 iterations
- Best score = 78.96%, in 646 iterations
- Best score = 78.92%, in 371 iterations (after refactoring incorrect answers encoder, maybe removing a bit of overfitting)

TRAIN 30000 users
- Best score = 79.50%, in 650 iterations
- Best score = 79.59%, in 742 iterations
- Best score = 79.60%, in 930 iterations
- Best score = 79.62%, in 855 iterations
- Best score = 79.64%, in 970 iterations (add prior_question_had_explanation_content_id_encoded)
"""


configure_console_logging()

loader = DataLoader(INPUT_PATH)
train, questions, lectures = loader.load_first_users(30000)
questions = preprocess_questions(questions)
lectures = preprocess_lectures(lectures)

test = loader.load_tests('tests_0.pkl')
train = merge_test(train, test)

model = RiiidModel(questions, lectures, params=PARAMS)
X, y, train, valid = model.fit_transform(train)
save_pkl((X, y, train, valid), path=os.path.join(MODELS_PATH, model.get_name('data.pkl')))

model.fit_lgbm(X[train], y[train], X[valid], y[valid])
#model.fit_catboost(X[train], y[train], X[valid], y[valid])

model.save(os.path.join(MODELS_PATH, model.get_name()))
