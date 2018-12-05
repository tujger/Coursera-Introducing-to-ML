package com.pantheoninc.sparktest;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public class TestCheckSpam implements Consumer<JavaSparkContext>, Serializable {
    @Override
    public void accept(JavaSparkContext sc) {

        JavaRDD<String> spam = sc.textFile("C:\\Users\\eduardm\\Downloads\\ml_spam.txt");
        JavaRDD<String> normal = sc.textFile("C:\\Users\\eduardm\\Downloads\\ml_normal.txt");

        Timer timer = new Timer();
        timer.startLog();

        HashingTF tf = new HashingTF(10000);

        JavaRDD<LabeledPoint> positive = spam.map(
                email -> new LabeledPoint(1, tf.transform(Arrays.asList(normalize(email).split(" ")))));
        JavaRDD<LabeledPoint> negative = normal.map(
                email -> new LabeledPoint(0, tf.transform(Arrays.asList(normalize(email).split(" ")))));

        JavaRDD<LabeledPoint> trainData = positive.union(negative);
        trainData.cache();

        // classification models
        LogisticRegressionModel model = LogisticRegressionWithSGD.train(trainData.rdd(), 1000);
//        SVMModel model = SVMWithSGD.train(trainData.rdd(), 100);
//        NaiveBayesModel model = NaiveBayes.train(trainData.rdd(), 1.0);
//        LogisticRegressionModel model = new LogisticRegressionWithLBFGS().run(trainData.rdd());

        // regression models
//        LinearRegressionModel model = LinearRegressionWithSGD.train(trainData.rdd(), 100);
//        LassoModel model = LassoWithSGD.train(trainData.rdd(), 100);
//        RidgeRegressionModel model = RidgeRegressionWithSGD.train(trainData.rdd(), 100);


  /*      int numClasses = 2;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String impurity = "gini";
        int maxDepth = 5;
        int maxBins = 16;
        Integer numTrees = 5;
        String featureSubsetStrategy = "auto";
        Integer seed = new Double(Math.random() * 10000).intValue();
//        DecisionTreeModel model = DecisionTree.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);
//        RandomForestModel model = RandomForest.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);





        JavaRDD<Vector> parsedData = spam.map(s -> {
            String[] sarray = normalize(s).split(" ");
            double[] values = new double[sarray.length];
            for (int i = 0; i < sarray.length; i++) {
                values[i] = sarray[i].hashCode();
            }
            return Vectors.dense(values);
        });
        parsedData.cache();

        int numClusters = 10;
        int numIterations = 100;
        KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);

        System.out.println("Cluster centers:");
        for (Vector center: clusters.clusterCenters()) {
            System.out.println(" " + center);
        }
        double cost = clusters.computeCost(parsedData.rdd());
        System.out.println("Cost: " + cost);

// Evaluate clustering by computing Within Set Sum of Squared Errors
        double WSSSE = clusters.computeCost(parsedData.rdd());
        System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
*/


        Vector posTest = tf.transform(Arrays.asList(normalize("10 М G GET cheap stuff Ьу sending money to ...").split(" "))).toSparse();
        Vector negTest = tf.transform(Arrays.asList(normalize("Hello Ed. I started studying Spark the other").split(" "))).toSparse();

        System.out.println("Prediction for positive: " + model.predict(posTest));
        System.out.println("Prediction for negative: " + model.predict(negTest));


        timer.stopLog();

//        System.out.println(posTest.toJson());
//        System.out.println(negTest.toJson());
//        System.out.println(model.toDebugString());

    }

    private String normalize(String text) {
        return text.replaceAll("\\W", " ")
                       .replaceAll("\\s+", " ");
    }
}
