package model;

import preprocessing.ImportData;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class LogisticRegressionModel implements Command {
    public static void main(String[] args) {
        Command cmd = new LogisticRegressionModel();
        cmd.exec(ImportData.trainSource, ImportData.testSource);
    }

    @Override
    public void exec(DataSource trainSource, DataSource testSource) {
        try {
            Instances trainData = trainSource.getDataSet();

            if (trainData.classIndex() == -1) {
                trainData.setClassIndex(trainData.numAttributes() - 1);
            }
            Instances testData = testSource.getDataSet();

            if (testData.classIndex() == -1) {
                testData.setClassIndex(testData.numAttributes() - 1);
            }

            // Create and train the NaiveBayes classifier
            Logistic lr = new Logistic();
            lr.buildClassifier(trainData);

            System.out.println("LR params" + String.join(" ", lr.getOptions()));

            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(lr, testData);

            // Output the evaluation results
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));

            // Print the confusion matrix
            System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

            // Print additional evaluation metrics
            System.out.println("Correct % = " + eval.pctCorrect());
            System.out.println("Incorrect % = " + eval.pctIncorrect());
            System.out.println("AUC = " + eval.areaUnderROC(1));
            System.out.println("Kappa = " + eval.kappa());
            System.out.println("MAE = " + eval.meanAbsoluteError());
            System.out.println("RMSE = " + eval.rootMeanSquaredError());
            System.out.println("RAE = " + eval.relativeAbsoluteError());
            System.out.println("RRSE = " + eval.rootRelativeSquaredError());
            System.out.println("Precision = " + eval.precision(1));
            System.out.println("Recall = " + eval.recall(1));
            System.out.println("F-Measure = " + eval.fMeasure(1));
            System.out.println("Error Rate = " + eval.errorRate());
            System.out.println(eval.toClassDetailsString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
