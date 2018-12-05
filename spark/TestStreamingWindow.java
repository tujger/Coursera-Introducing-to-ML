package com.pantheoninc.sparktest;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

import java.util.List;
import java.util.function.Consumer;

import scala.Tuple2;

/**
 * For streamcontext run: C:\Users\eduardm\Downloads\temp\nmap\ncat -lk 7777
 */

public class TestStreamingWindow implements Consumer<SparkConf> {
    @Override
    public void accept(SparkConf conf) {
        JavaStreamingContext jssc = new JavaStreamingContext(conf, Durations.seconds(1));
        jssc.checkpoint("c:\\\\Spark\\temp");
        JavaReceiverInputDStream<String> lines = jssc.socketTextStream("localhost", 7777);

        JavaDStream<String> linesFiltered = lines.filter((Function<String, Boolean>) v1 -> v1.contains("error"));

//        JavaDStream<String> linesWindow = linesFiltered.window(Durations.seconds(10),
//                        Durations.seconds(10));
//        JavaDStream<Long> windowCounts = linesWindow.count();


        JavaPairDStream<String, Long> ipAddressPairDStream = linesFiltered.mapToPair(entry -> new Tuple2(entry, 1L));
        JavaPairDStream<String, Long> ipCountDStream = ipAddressPairDStream.reduceByKeyAndWindow(
                (v1,v2) -> v1 + v2, // Добавить элементы из новых пакетов в окне
                (v1,v2) -> v1 - v2,
// Удалить элементы из пакетов, покинувших окно
                Durations.seconds(10), // Размер окна
                Durations.seconds(10)); // Шаг перемещения окна
//        Timer timer = new Timer().startLog();

        ipCountDStream.print();

        jssc.start();
        try {
            jssc.awaitTermination();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
//        timer.tickLog("End");
//        timer.stopLog();
    }
}
