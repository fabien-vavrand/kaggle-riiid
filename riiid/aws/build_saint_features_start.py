from riiid.utils import configure_console_logging
from doppel import DoppelContext, DoppelProject, destroy_all_projects


CONTEXT = DoppelContext() \
    .add_data(key='train.pkl', bucket='kaggle-riiid', source=r'C:\Users\chass\Kaggle\riiid\data\train.pkl')


if __name__ == '__main__':
    configure_console_logging()
    project = DoppelProject(
        name='riiid-saint-features',
        path=r'C:\Users\chass\Kaggle\riiid\kaggle-riiid',
        entry_point='-m riiid.aws.build_saint_features',
        packages=[r'C:\Users\chass\app\aws-doppel'],
        python='3.7.6',
        n_instances=1,
        min_memory=128,
        env_vars={'PYTHONHASHSEED': '1'},
        context=CONTEXT)
    project.terminate()
    project.start()
    # project.monitore()
