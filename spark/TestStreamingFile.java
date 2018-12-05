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

public class TestStreamingFile implements Consumer<SparkConf> {
    @Override
    public void accept(SparkConf conf) {

        String checkpointDir = "c:\\\\Spark\\temp";

        JavaStreamingContext jssc = new JavaStreamingContext(conf, Durations.seconds(1));
        jssc.checkpoint(checkpointDir);
        JavaDStream<String> lines = jssc.textFileStream("C:\\pantheon\\odyssey12Server1\\database\\logs");

        JavaDStream<String> linesFiltered = lines.filter((Function<String, Boolean>) v1 -> v1.contains("Exception"));

        JavaPairDStream<String, Long> response = linesFiltered
             .mapToPair(entry -> new Tuple2<>(entry, 1L))
             .updateStateByKey((List<Long> v1, Optional<Long> v2) -> Optional.of(v2.or(0L) + v1.size()));

        response.print();


        jssc.start();
        try {
            jssc.awaitTermination();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
