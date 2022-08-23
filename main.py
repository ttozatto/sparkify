import utils
from utils import printL as print
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf


def main():

    print("\n--------------------\n\n STARTING PROGRAM... \n\n--------------------\n")

    print('CREATING SPARK SESSION...')
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("Churn Prediction") \
        .config("spark.driver.memory", '1g') \
        .config("spark.executor.cores", '5') \
        .config("spark.shuffle.file.buffer", '1mb') \
        .config("spark.sql.shuffle.partitions", '5') \
        .getOrCreate()

    print(spark.sparkContext)
    conf = SparkConf()
    print("\n\nSPARK CONF: {}\n\n".format(conf.getAll()))

    print('READING FILE...')
    df = spark.read.json("medium-sparkify-event-data.json")
    df.persist()

    print('CLEANING DATA...')
    df = utils.clean_data(df)
    print('CREATING LABEL COLUMN...')
    df = utils.insert_churn(df)
    print('CREATING FEATURES...')
    df = utils.create_features(df)
    print('SPLITTING DATASET...')
    df_train, df_test = utils.split_users(df)

    models = []
    for filter in ['mean', 'last', None]:
        print("FILTER: ", filter)

        scores = []
        models_list = ['random forest', 'mlp', 'logistic regression', 'gbt']

        for model_type in models_list:

            print('CREATING MODEL [{}]...'.format(model_type))
            model = utils.create_model(model_type)
            print('TRAINING MODEL...')
            trained_model = utils.train_model(model, df_train)
            print('EVALUATING MODEL...')
            scores.append(utils.evaluate_model(trained_model, df_test, filter=filter))
            models.append(trained_model)
            print('PLOTTING TRAINING RESULT...')

            if model_type == 'random forest':
                utils.plot_pr_curve(trained_model)

        for i, model_type in enumerate(models_list):

            print("----------------------")
            print('F1 SCORE [{}]: {:.4f}'.format(model_type, scores[i][0]))
            print('ACC [{}]: {:.4f}'.format(model_type, scores[i][1]))
            print("----------------------")

    spark.stop()
    print("\n--------------------\n\n END OF PROGRAM... \n\n--------------------\n")


if __name__ == '__main__':
    main()
