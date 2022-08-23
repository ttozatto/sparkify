from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import when, col, lit, create_map, udf
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession

from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier, RandomForestClassifier, \
                                        LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline

from itertools import chain
import time
import random
from datetime import datetime
from matplotlib import pyplot as plt


spark = SparkSession.builder.getOrCreate()

evaluator = MulticlassClassificationEvaluator(metricName='f1', metricLabel=1)


def count_pages(df, new_columns, pages, partition='sessionId', time=False):
    """
    count the time spent in each page, can be grouped by session or user

    params:
    --------
    df: spark DataFrame
    new_columns: list of names for the new columns
    pages: list of pages to be counted
    partition: sessionId or userId

    returns the transformed spark DataFrame
    """
    
    assert len(new_columns) == len(pages)
    assert type(new_columns) == list
    assert type(pages) == list
    assert partition == 'sessionId' or partition == 'userId'
    
    windowval = (Window.partitionBy('sessionId').orderBy('ts')
             .rangeBetween(Window.unboundedPreceding, 0))
    
    if time:
        count_value = df.deltaT
    else:
        count_value = 1

    for i in range(len(new_columns)):
        df = df.withColumn(new_columns[i]+'_bool', when(df.page == pages[i], count_value)
                                           .otherwise(0))
        
        df = df.withColumn(new_columns[i], F.sum(new_columns[i]+'_bool').over(windowval))
        df = df.drop(new_columns[i]+'_bool')
        
    return df


def split_users(df, test_size=0.3, seed=42):

    random.seed(seed)
    
    users = [u[0] for u in df.select('userId').distinct().collect()]
    test_size = int(len(users)*test_size)
    test_users = random.sample(users, test_size)
    train_users = [u for u in users if (u not in test_users)]
    
    df_test = df.filter(col('userId').isin(test_users))
    df_train = df.filter(col('userId').isin(train_users))
    
    return df_train, df_test


def clean_data(df):
    
    return df.where(col('userId') != '')


def insert_churn(df):
    
    df_aux = df.withColumn('plan_change', when(df.page == 'Submit Downgrade', '-1')
                                       .when(df.page == 'Cancellation Confirmation', '-1')
                                       .otherwise('0'))

    churn_users = list(df_aux.select('userId').filter(col('plan_change')==-1).collect())
    churn_users = [u.userId for u in churn_users]

    df = df.withColumn('label', when(col('userId').isin(churn_users), 1).otherwise(0))

    return df


def create_features(df):

    df_lag = df.withColumn('prev_ts',
                                F.lag(df.ts)
                                    .over(Window.partitionBy("sessionId").orderBy('ts')))

    df_deltaT = df_lag.withColumn('deltaT', ((col('ts') - col('prev_ts'))/1000).cast(IntegerType()))
    df_deltaT = df_deltaT.na.fill({'deltaT':'0.0'})

    windowval = (Window.partitionBy('sessionId').orderBy('ts')
                    .rangeBetween(Window.unboundedPreceding, 0))

    windowval2 = (Window.partitionBy('userId').orderBy('ts')
                    .rangeBetween(Window.unboundedPreceding, 0))

    df_sessionT = df_deltaT.withColumn('sessionTime', F.sum('deltaT').over(windowval))

    df_sessionT = df_sessionT.withColumn('userTime', F.sum('deltaT').over(windowval2))

    df_songs = df_sessionT.withColumn('sessionSongs', F.count('song').over(windowval))

    df_songs = df_songs.withColumn('userSongs', F.count('song').over(windowval2))

    df_song_hour = df_songs.withColumn('songsHour', col('sessionSongs')/(col('sessionTime')/(60*60)))
    df_song_hour = df_song_hour.na.fill({'songsHour':'0.0'})

    df_pre_final = count_pages(df_song_hour, ['sessionTup', 'sessionTdown'], 
                           ['Thumbs Up', 'Thumbs Down'], partition='sessionId',
                           time=False)

    df_pre_final = count_pages(df_pre_final, ['sessionAds'], 
                           ['Roll Advert'], partition='sessionId',
                           time=True)

    df_pre_final = count_pages(df_pre_final, ['userTup', 'userTdown', 'about', 'downgrade', 
                                                'upgrade', 'submitUpgrade', 'playlist', 'friend',
                                                 'userError', 'save'], 
                           ['Thumbs Up', 'Thumbs Down', 'About', 'Downgrade', 'Upgrade', 
                           'Submit Upgrade', 'Add to Playlist', 'Add Friend',
                            'Error', 'Save Settings']
                           , partition='userId', time=False)

    df_pre_final = count_pages(df_pre_final, ['userAds'], 
                           ['Roll Advert']
                           , partition='userId', time=True)

    df_pre_final = df_pre_final.withColumn('male', when(col('gender')=='M', 1).otherwise(0))
    df_pre_final = df_pre_final.withColumn('female', when(col('gender')=='F', 1).otherwise(0))

    df_pre_final = df_pre_final.withColumn('paidDeltaTime', 
                    when(col('level')=='paid', col('deltaT')).otherwise(0))
    df_pre_final = df_pre_final.withColumn('paidPerc', 
                    F.sum('paidDeltaTime').over(windowval2)/col('userTime'))
    df_pre_final = df_pre_final.na.fill({'paidPerc':'0.0'})

    df_pre_final = df_pre_final.withColumn('paid', when(col('level')=='paid', 1).otherwise(0))
    df_pre_final = df_pre_final.withColumn('free', when(col('level')=='free', 1).otherwise(0))

    features = ['userId', 'itemInSession', 'songsHour', 'male', 'female', 'paid', 'free', 'paidPerc', 
                'sessionTime', 'userTime', 'sessionSongs', 'userSongs', 'sessionTup', 
                'userTup', 'sessionTdown', 'userTdown', 'sessionAds', 'userAds', 
                'userError', 'save', 'downgrade', 'upgrade', 'submitUpgrade', 'about',
                'playlist', 'friend', 'label']       

    return df_pre_final.select(*features[:-1], col('label'))


def create_model(classifier_type='random forest'):

    train_features = ['songsHour', 'male', 'female', 'paidPerc', 'sessionTime', 'userTime', 
                    'userSongs', 'sessionTup', 'userTup', 'sessionTdown', 'userTdown', 
                    'sessionAds', 'userAds', 'userError', 'save', 'downgrade', 'upgrade', 
                    'submitUpgrade', 'about', 'playlist', 'friend']

    assembler = VectorAssembler(inputCols=train_features, outputCol='vectorFeatures')
    scaler = MinMaxScaler(inputCol='vectorFeatures', outputCol='scaledFeatures')

    if classifier_type == 'random forest':
        classifier = RandomForestClassifier(featuresCol='scaledFeatures', seed=42)
        paramGrid = ParamGridBuilder() \
                    .addGrid(classifier.maxDepth, [8, 12, 24]) \
                    .addGrid(classifier.numTrees, [8, 16, 32]) \
                    .addGrid(classifier.maxBins, [24]) \
                    .addGrid(classifier.impurity, ['gini']) \
                    .build()

    elif classifier_type == 'mlp':
        classifier = MultilayerPerceptronClassifier(featuresCol='scaledFeatures', seed=42)
        paramGrid = ParamGridBuilder() \
                    .addGrid(classifier.maxIter, [500]) \
                    .addGrid(classifier.stepSize, [0.03]) \
                    .addGrid(classifier.layers, [
                        [len(train_features), 32, 2],
                        [len(train_features), 32, 8, 2],
                        [len(train_features), 64, 8, 2],
                        [len(train_features), 128, 64, 8, 2],
                        ]) \
                    .build()

    elif classifier_type == 'logistic regression':
        classifier = LogisticRegression(featuresCol='scaledFeatures', family='binomial')
        paramGrid = ParamGridBuilder() \
                    .addGrid(classifier.maxIter, [8]) \
                    .addGrid(classifier.regParam, [.0001, .001, .01]) \
                    .addGrid(classifier.elasticNetParam, [.001, .01, .1]) \
                    .addGrid(classifier.threshold, [.4, .45, .5, .55, .6]) \
                    .build()

    elif classifier_type == 'gbt':
        classifier = GBTClassifier(featuresCol='scaledFeatures', seed=42)
        paramGrid = ParamGridBuilder() \
                    .addGrid(classifier.maxIter, [8, 16, 32]) \
                    .addGrid(classifier.stepSize, [.001, .01, .1]) \
                    .build()

    tvs = TrainValidationSplit(estimator=classifier,
                            parallelism=5,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            seed=42, 
                            trainRatio=0.7)

    return Pipeline(stages=[assembler, scaler, tvs])


def train_model(model, df_train):

    start_time = time.time()
    model = model.fit(df_train)
    printL("TRAINING TIME: {:.2f}s".format(time.time() - start_time))
    return model


def evaluate_model(model, df_test, filter=None):

    predictions = model.transform(df_test)

    if filter:
        predictions = filter_predictions(predictions, filter)

    f1_score = evaluator.evaluate(predictions)

    print("------------")
    printL("MODEL: ", model.stages[2].bestModel)
    printL("F1: {:.4f}".format(f1_score))
    acc = predictions.filter(col('prediction') == col('label')).count()/predictions.count()
    printL("ACC: {:.4f}".format(acc))
    print("------------")

    return f1_score, acc
    

def printL(string, *args):

    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(now,'[CHURN-PREDICTION APP]', string, *args)


def predict_churn(pred, method='last'):

    if method == 'mean':
        df_aux = pred.groupBy('userId').mean('prediction')
        return dict(df_aux.select('userId', F.round(col('avg(prediction)')).alias('prediction')) \
                        .collect())

    elif method=='last':
        return dict(pred.join(pred.groupBy('userId').agg(F.max('userTime').alias('userTime')), 
                                on=['userId', 'userTime'], how='leftsemi') \
                                    .select('userId', 'prediction').collect())


def filter_predictions(pred, method='last'):    

    churn_dict = predict_churn(pred, method)

    mapping_expr = create_map([lit(x) for x in chain(*churn_dict.items())])

    df_aux = pred.withColumn('userPred', mapping_expr[col('userId')])

    return df_aux.drop('prediction').withColumnRenamed('userPred', 'prediction')


def plot_pr_curve(model):

    precision = model.stages[2].bestModel.summary.precisionByThreshold
    recall = model.stages[2].bestModel.summary.recallByThreshold
    f1 = model.stages[2].bestModel.summary.fMeasureByThreshold
    df = precision.join(recall, on='threshold')
    df = df.join(f1, on='threshold').toPandas()
    df = df.set_index('threshold').sort_values('threshold')

    df.plot(x='recall', y='precision', kind='scatter')
    plt.savefig('PR_curve.png')

    df.plot()
    df['delta'] = abs(df.precision - df.recall)
    plt.axvline(df.iloc[df.delta.argmin()].name, color='r')
    plt.legend(loc='lower left')
    plt.savefig('precision_recall.png')
