package com.pantheoninc.sparktest;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.Serializable;

/**
 * For streamcontext run: C:\Users\eduardm\Downloads\temp\nmap\ncat -lk 7777
 */

public class Main implements Serializable {

    public static void main(String[] args) throws InterruptedException {

        LogManager.getLogger("org").setLevel(Level.WARN);
        System.out.println("Hello World!");
        System.setProperty("hadoop.home.dir", "c:\\\\Spark\\winutils");
        System.setProperty("spark.driver.allowMultipleContexts", "true");

        // Создать SparkContext
        SparkConf conf = new SparkConf().setMaster("local[2]").setAppName("wordCount");
        JavaSparkContext sc = new JavaSparkContext(conf);

        int path = 6;


        if(path == 0) {
            new TestSimple().accept(sc);
        } else if(path == 1) {
            new TestLoadAndCount().accept(sc);
        } else if(path == 2) {
            new TestStreaming().accept(conf);
        } else if(path == 3) {
            new TestStreamingWindow().accept(conf);
        } else if(path == 4) {
            new TestStreamingCollectCount().accept(conf);
        } else if(path == 5) {
            new TestStreamingFile().accept(conf);
        } else if(path == 6) {
            new TestCheckSpam().accept(sc);
        }
        while(true) {}
    }

}
