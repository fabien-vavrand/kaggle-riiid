
import os
import logging
from riiid.core.data import DataLoader, preprocess_questions, preprocess_lectures, save_pkl, load_pkl
from riiid.core.model import RiiidModel
from riiid.validation import merge_test
from riiid.utils import configure_console_logging
from riiid.config import PATH, INPUT_PATH, MODELS_PATH, PARAMS


"""
TODO: 
- TO TEST: in sorted_rolling_score, sum and count should not be decremented when rolling in the middle of a task
- update answers count in AnswersEncoder
- sum of prior_question_lost_time sur la session
- average et sum content_time par user, par session
- content time is null for first question of session: fill intelligently ? (mean, mean of previous session, ...)
- prior_question_time: check if it is better to compute it for the first question of a session or not ? (group by session_id in tasks)
- sum questions explanation time during session
- start session with lecture / session lecture time/learning time (time question lost)
- user_score / content_id
- retester n_lectures_on_tag, maybe I had a bug previosuly
- change the smoothing of user_id_content_id_score
- user score by day (even if not easy to compute at prediction time)
- ratio question_part, question_tag / category per user (user had 45% of this question part... etc)
- user % of different user_answer (% of answer 1, % of 2, ...)
- apply a decay factor to lectures (n_lectures)
- keep ids in features (content_id, etc...)
- sum of user_id / content_id answered_correctly
- precompute dictionnary of weights to speed inference
- maybe I could do better with content_time (not break by session, ...)
- previous task_size
- try DBSCAN instead of k-means
- previous lecture_time
- time since last session
- essayer de smoother davantage les content_id encoder pour limiter l'usage du content_id_ratio, qui doit compenser ca jimagine...
- changer save with source pour save directement model/context.txt et model/model.pkl au lieu du zip
- feature decaying
- last score for user_id / content_id
- user_id_session_density
- nombre de réponse possible par question
- raffiner les buckets des tasks_bucket_12
- les premières questions sont toujours les mêmes!
- encoder par previous_question_tag / previous_answered_correctly / contetn_id
- last lecture type
- % of tags in common with previous question
- investigate none in answer_0_ratio
- use weighting to rebalance the dataset

NOT IMPLEMENTED AT PREDICT TIME
- decay for UserScoreEncoder
- answer ratio in UserAnswerEncoder

TRAIN 10000 users
- Best score = 78.85%, in 305 iterations
- Best score = 78.89%, in 290 iterations
- Best score = 78.92%, in 435 iterations
- Best score = 78.97%, in 412 iterations (add decay on user and user/question_part)
- Best score = 78.95%, in 291 iterations

TRAIN 30000 users
- Best score = 79.50%, in 650 iterations
"""


configure_console_logging()

loader = DataLoader(INPUT_PATH)
train, questions, lectures = loader.load_first_users(10000)
questions = preprocess_questions(questions)
lectures = preprocess_lectures(lectures)

test = loader.load_tests('tests_0.pkl')
train = merge_test(train, test)

model = RiiidModel(questions, lectures, params=PARAMS)
X, y, train, valid = model.fit_transform(train)
save_pkl((X, y, train, valid), path=os.path.join(MODELS_PATH, model.get_name('data.pkl')))

model.fit_model(X[train], y[train], X[valid], y[valid])
#model.refit_model(X, y)

model.save(os.path.join(MODELS_PATH, model.get_name()))
