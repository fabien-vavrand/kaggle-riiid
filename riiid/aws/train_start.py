from riiid.utils import configure_console_logging
from doppel import DoppelContext, DoppelProject, destroy_all_projects


CONTEXT = DoppelContext() \
    .add_data(key='train.pkl', bucket='kaggle-riiid', source=r'C:\Users\chass\Kaggle\riiid\data\train.pkl') \
    .add_data(key='tests_0.pkl', bucket='kaggle-riiid', source=r'C:\Users\chass\Kaggle\riiid\data\tests_0.pkl') \
    .add_data(key='tests_1.pkl', bucket='kaggle-riiid', source=r'C:\Users\chass\Kaggle\riiid\data\tests_1.pkl')


if __name__ == '__main__':
    configure_console_logging()
    project = DoppelProject(
        name='riiid-train',
        path=r'C:\Users\chass\Kaggle\riiid\kaggle-riiid',
        entry_point='-m riiid.aws.train_neural',
        packages=[r'C:\Users\chass\app\aws-doppel'],
        python='3.7.6',
        n_instances=1,
        min_memory=128,
        context=CONTEXT,
        env_vars={'PYTHONHASHSEED': '1'})
    project.terminate()
    project.start()
    # project.monitore()
