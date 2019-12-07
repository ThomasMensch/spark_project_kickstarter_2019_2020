package paristech

import java.io._

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

    // Create file to monitor steps of the program
    val writer = new PrintWriter(new File("output.txt"))

    // 1. Load DataFrame from TP2 (parquet file)
    val path_to_data: String = "ressources/"
    val df: DataFrame = spark.read.parquet(path_to_data + "preprocessed")

    writer.write("1.1 Load data (from TP2) -> done\n")

    writer.write("\tSize of data: %d x %d\n".format(df.count(), df.columns.length))
    
    // Data cleaning before processing.
    // We remove rows with records set to -1 and 'Unknown' (see TP2)
    val preprocessed: DataFrame = df
      .filter($"days_campaign" =!= -1)
      .filter($"hours_prepa" =!= -1)
      .filter($"goal" =!= -1)
      .filter($"country2" =!= "Unknown")
      .filter($"currency2" =!= "Unknown")

    writer.write("1.2 Extra cleaning of data -> done\n")
    writer.write("\tSize of data: %d x %d\n".format(preprocessed.count(),
                                                    preprocessed.columns.length))
    writer.flush()

    // 2. Transform text data
    // 2.1. Transform text into a feature vector
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // 2.2. Remove stop words
    StopWordsRemover.loadDefaultStopWords("english")
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    // 2.3. Compute TF part using CountVectorizerModel from the corpus
    val countVectorizer: CountVectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("tf")
    //  .setMinDF(55)

    // 2.4. Compute IDF part
    val idf = new IDF()
      .setInputCol(countVectorizer.getOutputCol)
      .setOutputCol("tfidf")

    writer.write("2. Transform text data -> done\n")
    writer.flush()

    // 3.1. Convert categorial variables into numerical variables
    // we add the 'setHandleInvalid("keep")' cmd to avoid problem in TVS
    val countryIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    val currencyIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")

    // 3.2. Hot encoder
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    writer.write("3. Convert categorial data -> done\n")
    writer.flush()

    // 4. Prepare data for ML processing
    // 4.1. Assemble all features as a unique vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    // 4.2. Create logistic regression model
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

    writer.write("4. Prepare data for ML processing -> done\n")
    writer.flush()

    writer.write("\n===== LOGISTIC REGRESSION =====\n\n")
    writer.flush()
    
    // 5. Create Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, countVectorizer, idf,
        countryIndexer, currencyIndexer, encoder, assembler, lr))

    writer.write("5. Create Pipeline -> done\n")
    writer.flush()

    // 6. Training, test and backup of the model
    // 6.1 We split the data in 2 sets: 90 % of the data for training and 10 % for testing.
    val Array(training, test) = preprocessed.randomSplit(Array(0.9, 0.1), 13)

    writer.write("6.1. Split data in 2 sets: training (90%) and test (10%) -> done\n")
    writer.flush()

    // 6.2 Training of the model
    val model_one = pipeline.fit(training)

    writer.write("6.2. Training of the model -> done\n")
    writer.flush()

    // 6.3 Make predictions from test data
    val dfWithSimplePredictions = model_one.transform(test)
      .select("features", "final_status", "predictions")

    writer.write("6.3. Make predictions from test data -> done\n")
    writer.flush()

    // 6.4 Evaluation of the model / test (before grid search)
    // We use f1-score to compare models
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val f1_score_one = evaluator.evaluate(dfWithSimplePredictions)

    writer.write("\nThe f1 score on test set [before grid search] is : %.3f\n\n".format(f1_score_one))
    writer.flush()

    // 7. Grid search
    // 7.1 Tuning of hyper-parameters of the model
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(countVectorizer.minDF, Array[Double](55, 75, 95))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7) // 70% of the data will be used for training and the remaining 30% for validation.

    writer.write("7.1. Tuning of hyper-parameters of the model -> done\n")
    writer.flush()

    // 7.2 Train validation
    val tvs_model = trainValidationSplit.fit(training)

    writer.write("7.2. Train validation -> done\n")
    writer.flush()

    // 7.3 Make predictions from test data
    val predictions = tvs_model
      .transform(test)
      .select("features", "final_status", "predictions")

    // 7.4 Evaluate F1 score
    val f1_score_2 = evaluator.evaluate(predictions)

    writer.write("\nThe f1 score on test set [after grid search] is : %.3f\n\n".format(f1_score_2))

    // 8. Save the best model
    tvs_model.write.overwrite().save("ressources/best-logistic-regression-model")

    writer.write("8. Save the best model -> done\n")
    writer.flush()
    
    writer.close()

    println("hello world ! from Trainer")
  }
}
