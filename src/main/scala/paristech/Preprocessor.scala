package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    import spark.implicits._

    // 1. Load the data
    val path_to_data: String = "ressources/train/"

    val df: DataFrame = spark
      .read
      .option("header", "true") // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv(path=path_to_data + "train_clean.csv")

    //println(s"Nombre de lignes : ${df.count}")
    //println(s"Nombre de colonnes : ${df.columns.length}")

    // 2. Recast some columns to Int
    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

     // 3. Cleaning using UDF functions
    val df2: DataFrame = dfCasted.drop("disable_communication")

    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    // 3.1 create UDF to clean 'country' records
    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else if (country != null && country.length != 2)
        null
      else
        country
    }

    val cleanCountryUdf = udf(cleanCountry _)
    
    // 3.2 create UDF to clean 'currency' records
    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCurrencyUdf = udf(cleanCurrency _)

    // 3.3 Apply udf
    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    // 4. Add columns
    // 4.1 Add columns related to duration of the campaign
    def timestampDiffInHours(end: Int, start: Int): Double = {
      val diff: Double = (end - start) / 3600.0
      diff
    }

    val timestampDiffInHoursUdf = udf(timestampDiffInHours _)

    val dfDatetime: DataFrame = dfCountry
      .withColumn("days_campaign", datediff(from_unixtime($"deadline"),
                                            from_unixtime($"launched_at")))
      .withColumn("hours_prepa", round(timestampDiffInHoursUdf($"launched_at",
                                                               $"created_at"), 3))
      .drop("launched_at", "created_at", "deadline")

    // 4.2 Transform text columns  
    val dfText: DataFrame = dfDatetime
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))
      .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))

    // 5. Null values
    val dfNotNull: DataFrame = dfText
      .na.fill(-1, Seq("days_campaign"))
      .na.fill(-1, Seq("hours_prepa"))
      .na.fill(-1, Seq("goal"))
      .na.fill("Unknown", Seq("country2"))
      .na.fill("Unknown", Seq("currency2"))

    // 6. Save as parquet files
    dfNotNull.write.parquet("ressources/preprocessed")

    println("\n")
    println("Hello World ! from Preprocessor")
    println("\n")
  }
}
