from doppel import DoppelContext, DoppelProject, destroy_all_projects


CONTEXT = (
    DoppelContext()
    .add_data(key="train.pkl", bucket="kaggle-riiid", source=r"C:\Users\chass\Kaggle\riiid\data\train.pkl")
    .add_data(key="tests_0.pkl", bucket="kaggle-riiid", source=r"C:\Users\chass\Kaggle\riiid\data\tests_0.pkl")
    .add_data(key="tests_1.pkl", bucket="kaggle-riiid", source=r"C:\Users\chass\Kaggle\riiid\data\tests_1.pkl")
)


if __name__ == "__main__":
    CONTEXT.get_logger()
    project = DoppelProject(
        name="riiid-tune",
        path=r"C:\Users\chass\Kaggle\riiid\kaggle-riiid",
        entry_point="-m riiid.aws.tune",
        packages=[r"C:\Users\chass\app\aws-doppel"],
        python="3.7.6",
        n_instances=10,
        min_memory=16,
        context=CONTEXT,
    )
    project.terminate()
    # project.start()
    # project.monitore()
