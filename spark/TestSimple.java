package com.pantheoninc.sparktest;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.Optional;

import java.util.ArrayList;
import java.util.function.Consumer;

import scala.Tuple2;

public class TestSimple implements Consumer<JavaSparkContext> {
    @Override
    public void accept(JavaSparkContext sc) {
        JavaPairRDD<String, String> storeAddress = sc.parallelizePairs(new ArrayList<Tuple2<String, String>>() {{
            add(new Tuple2<>("Ritual", "1026 Valencia St"));
            add(new Tuple2<>("Philz", "748 Van Ness Ave"));
            add(new Tuple2<>("Philz", "3101 24th St"));
            add(new Tuple2<>("Starbucks", "Seattle"));
        }});
        JavaPairRDD<String, Double> storeRating = sc.parallelizePairs(new ArrayList<Tuple2<String, Double>>() {{
            add(new Tuple2<>("Ritual", 4.9));
            add(new Tuple2<>("Philz", 4.8));
        }});

        JavaPairRDD<String, Tuple2<String, Optional<Double>>> leftResult = storeAddress.leftOuterJoin(storeRating);
        JavaPairRDD<String, Tuple2<Optional<String>, Double>> rightResult = storeAddress.rightOuterJoin(storeRating);

        System.out.println(leftResult.take(3));

    }
}
