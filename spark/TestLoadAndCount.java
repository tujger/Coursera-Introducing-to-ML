package com.pantheoninc.sparktest;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;

import java.util.function.Consumer;

import scala.Tuple2;

public class TestLoadAndCount implements Consumer<JavaSparkContext> {
    @Override
    public void accept(JavaSparkContext sc) {
        //        sc.setLogLevel("ERROR");
// Загрузить исходные данные.
        JavaRDD<String> inputRDD = sc.textFile("C:\\Users\\eduardm\\Downloads\\odyssey.log");
//        inputRDD.saveAsObjectFile("C:\\Users\\eduardm\\Downloads\\spark.out");
//        JavaRDD<String> inputRDD = sc.objectFile("C:\\Users\\eduardm\\Downloads\\spark.out");
//            inputRDD.persist(StorageLevel.MEMORY_ONLY());

        Timer timer = new Timer().startLog();

        JavaRDD<String> errorsRDD = inputRDD.filter(line -> line.toLowerCase().contains("error"));
        // Разбить на слова.
//            JavaRDD<String> wordsRDD = errorsRDD.flatMap(line -> Arrays.asList(line.toLowerCase().split("error")).iterator());
//            wordsRDD.persist(StorageLevel.MEMORY_ONLY());

        timer.tickLog("A");

        // Преобразовать в пары и выполнить подсчет.
        JavaPairRDD<String, Integer> countsRDD = errorsRDD.mapToPair(word -> {
            return new Tuple2<>(word, 1);
        }).reduceByKey((a, b) -> a + b).sortByKey(false);

        countsRDD.persist(StorageLevel.MEMORY_ONLY());
        timer.tickLog("B");

        System.out.println("Count: " + countsRDD.count());

        timer.tickLog("C");

        System.out.println(countsRDD.take(10));

        timer.tickLog("D");
        timer.stopLog();

        System.out.println("Debug: " + countsRDD.toDebugString());
        System.out.println("Partitions: " + countsRDD.partitions().size());

    }
}
