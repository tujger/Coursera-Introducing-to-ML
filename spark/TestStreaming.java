package com.pantheoninc.sparktest;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

import java.util.Arrays;
import java.util.Iterator;
import java.util.function.Consumer;
import java.util.regex.Pattern;

import scala.Tuple2;

/**
 * For streamcontext run: C:\Users\eduardm\Downloads\temp\nmap\ncat -lk 7777
 */

public class TestStreaming implements Consumer<SparkConf> {
    @Override
    public void accept(SparkConf conf) {
        JavaStreamingContext jssc = new JavaStreamingContext(conf, Durations.seconds(1));
        JavaReceiverInputDStream<String> lines = jssc.socketTextStream("localhost", 7777);

        JavaDStream<String> linesFiltered = lines.filter((Function<String, Boolean>) v1 -> v1.contains("error"));

        JavaDStream<String> words = linesFiltered.flatMap(x -> Arrays.asList(x.split(" ")).iterator());

        // Count each word in each batch
        JavaPairDStream<String, Integer> pairs = words.mapToPair(s -> new Tuple2<>(s, 1));

        JavaPairDStream<String, Integer> wordCounts = pairs.reduceByKey((i1, i2) -> i1 + i2);

        // Print the first ten elements of each RDD generated in this DStream to the console
//        JavaDStream<String> words = lines.flatMap(x -> Arrays.asList(Pattern.compile(" ").split(x)).iterator());
//
//        JavaPairDStream<String, Integer> wordCounts = words.mapToPair(s -> new Tuple2<>(s, 1))
//                                                              .reduceByKey((i1, i2) -> i1 + i2);

        wordCounts.print();

//        Timer timer = new Timer().startLog();
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
