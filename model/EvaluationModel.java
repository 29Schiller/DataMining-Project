package model;

import weka.core.Instances;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

public class EvaluationModel {
    public static void main(String args[]) throws Exception{

        DataSource source = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\data\\training_data.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes()-1);

        J48 tree = new J48();
        tree.buildClassifier(dataset);

        Evaluation eval = new Evaluation(dataset);
        Random rand = new Random(1);
        int folds = 10;

        /* Notice we build the classifier with the training dataset
        we initialize evaluation with the training dataset and then
        evaluate using the test dataset */

        DataSource source1 = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\data\\testing_data.arff");
        Instances testDataset = source1.getDataSet();
        //set class index to the last attribute
        testDataset.setClassIndex(testDataset.numAttributes()-1);
        //now evaluate model
        //eval.evaluateModel(tree, testDataset);
        eval.crossValidateModel(tree, testDataset, folds, rand);
        System.out.println(eval.toSummaryString("Evaluation results:\n", false));

        System.out.println("Correct % = "+eval.pctCorrect());
        System.out.println("Incorrect % = "+eval.pctIncorrect());
        System.out.println("AUC = "+eval.areaUnderROC(1));
        System.out.println("kappa = "+eval.kappa());
        System.out.println("MAE = "+eval.meanAbsoluteError());
        System.out.println("RMSE = "+eval.rootMeanSquaredError());
        System.out.println("RAE = "+eval.relativeAbsoluteError());
        System.out.println("RRSE = "+eval.rootRelativeSquaredError());
        System.out.println("Precision = "+eval.precision(1));
        System.out.println("Recall = "+eval.recall(1));
        System.out.println("fMeasure = "+eval.fMeasure(1));
        System.out.println("Error Rate = "+eval.errorRate());
        //the confusion matrix
        System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));

    }
}
