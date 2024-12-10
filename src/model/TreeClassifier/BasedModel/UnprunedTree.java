package TreeClassifier.BasedModel;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;

public class UnprunedTree {
    public static void main(String[] args) throws Exception {
        DataSource trainSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining\\src\\Data\\training_data.arff");
        Instances trainDataset = trainSource.getDataSet();
        DataSource testSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining\\src\\Data\\testing_data.arff");
        Instances testDataset = trainSource.getDataSet();

        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
        testDataset.setClassIndex(testDataset.numAttributes() - 1);

        J48 unprunedTree = new J48();
        unprunedTree.setUnpruned(true);
        unprunedTree.buildClassifier(trainDataset);

        System.out.println("J48 params" + String.join(" ", unprunedTree.getOptions()));

        Evaluation testEval = new Evaluation(trainDataset);
        testEval.evaluateModel(unprunedTree, testDataset);

        // Print the confusion matrix
        System.out.println("Confusion Matrix:\n" + testEval.toMatrixString());

        // Print additional evaluation metrics
        System.out.println(testEval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + testEval.precision(1));
        System.out.println("Recall = " + testEval.recall(1));
        System.out.println("F-Measure = " + testEval.fMeasure(1));
        System.out.println("Error Rate = " + testEval.errorRate());
        System.out.println(testEval.toClassDetailsString());
    }
}
