from utils import utils


subject_list = ['datascience', 'machinelearning', 'physics', 'astrology', 'conspiracy']

if __name__ == '__main__':

    data, labels = utils.load_data(subject_list=subject_list)

    x_train, x_test, y_train, y_test = utils.split_data(data=data, labels=labels)

    pipeline = utils.preprocessing_pipeline()

    model_list = utils.build_model_list()

    result_list = utils.train_evaluation(
        model_list=model_list,
        pipeline=pipeline,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test)

print('Process Finished.')
