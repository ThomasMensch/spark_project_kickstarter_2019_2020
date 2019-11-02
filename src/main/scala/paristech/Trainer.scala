package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.DataFrame

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.ml.classification.LogisticRegression

import org.apache.spark.ml.Pipeline

import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /** *****************************************************************************
      *
      * TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    import spark.implicits._

    // 1. Load DataFrame from TP2 (parquet file)
    val path_to_data: String = "/home/thomas/MyDevel/workspace-github/spark_project_kickstarter_2019_2020/data/"
 //   val preprocessed: DataFrame = spark.read.parquet(path_to_data + "prepared_trainingset")
    val df: DataFrame = spark.read.parquet(path_to_data + "parquet")

    // Data cleaning before processing. We remove rows with records set to -1 and 'Unknown' (see TP2)
    val preprocessed: DataFrame = df
      .filter($"days_campaign" =!= -1)
      .filter($"hours_prepa" =!= -1)
      .filter($"goal" =!= -1)
      .filter($"country2" =!= "Unknown")
      .filter($"currency2" =!= "Unknown")

    // 2. Transform text into a feature vector
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // 3. Remove stop words
    StopWordsRemover.loadDefaultStopWords("english")
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    // 4. Compute TF part using CountVectorizerModel from the corpus
    val countVectorizer: CountVectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("tf")
    //  .setMinDF(55)

    // 5. Compute IDF part
    val idf = new IDF()
      .setInputCol(countVectorizer.getOutputCol)
      .setOutputCol("tfidf")

    // 6. Convert categorial variables into numerical variables
    // we add the 'setHandleInvalid("keep")' cmd to avoid problem in TVS
    val countryIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    val currencyIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")

    // 8. Hot encoder
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    // 9. Assemble all features as a unique vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    // 10. Create logistic regression model
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(30)

    // === Pipeline ===
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, countVectorizer, idf,
        countryIndexer, currencyIndexer, encoder, assembler, lr))

    // === Training, test and backup of the model ===
    // Split of the data in 2 sets: training and test
    // We split the data in 2 sets: 90 % of the data for training and 10 % for testing.
    // Note that we use the initial dataframe preprocessed.
    val Array(training, test) = preprocessed.randomSplit(Array(0.9, 0.1), 13)

    // Training of the model
    val model_one = pipeline.fit(training)

    // Make predictions from test data
    val dfWithSimplePredictions = model_one.transform(test)
      .select("features", "final_status", "predictions")

    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    // Evaluation of the model / test
    // We use f1-score to compare models
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val f1_score = evaluator.evaluate(dfWithSimplePredictions)

    printf("The f1 score on test set is : %.3f\n", f1_score)

    // Tuning of hyper-parameters of the model
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(countVectorizer.minDF, Array[Double](55, 75, 95))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7) // 70% of the data will be used for training and the remaining 30% for validation.

    // Fit the pipeline to training set.
    val tvs_model = trainValidationSplit.fit(training)

    val predictions = tvs_model
      .transform(test)
      .select("features", "final_status", "predictions")

    predictions.groupBy("final_status", "predictions").count.show()

    val f1_score = evaluator.evaluate(predictions)

    println("The f1 score on test set is : " + f1_score + "\n")

    tvs_model.write.overwrite().save("save/spark-logistic-regression-model")
    println("hello world ! from Trainer")
  }
}
