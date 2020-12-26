from riiid.utils import configure_console_logging
from doppel import DoppelProject, destroy_all_projects


if __name__ == "__main__":
    configure_console_logging()
    project = DoppelProject(
        name="riiid-train-neural",
        path=r"C:\Users\chass\Kaggle\riiid\kaggle-riiid",
        entry_point="-m riiid.aws.train_neural",
        packages=[r"C:\Users\chass\app\aws-doppel"],
        python="3.7.6",
        n_instances=1,
        min_memory=256,
        min_gpu=1,
        env_vars={"PYTHONHASHSEED": "1"},
    )
    project.terminate()
    # project.start()
    # project.monitore()
