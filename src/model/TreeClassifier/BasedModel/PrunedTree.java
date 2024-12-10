package TreeClassifier.BasedModel;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;

public class PrunedTree {
    public static void main(String[] args) throws Exception {
        DataSource trainSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining - Copy\\src\\Data\\training_data.arff");
        Instances trainDataset = trainSource.getDataSet();
        DataSource testSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining - Copy\\src\\Data\\testing_data.arff");
        Instances testDataset = trainSource.getDataSet();

        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
        testDataset.setClassIndex(testDataset.numAttributes() - 1);

        J48 prunedTree = new J48();
        prunedTree.buildClassifier(trainDataset);

        System.out.println("J48 params" + String.join(" ", prunedTree.getOptions()));

        Evaluation testEval = new Evaluation(trainDataset);
        testEval.evaluateModel(prunedTree, testDataset);

        // Print the confusion matrix
        System.out.println("\n" + testEval.toMatrixString());
        // Print additional evaluation metrics
        System.out.println(testEval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + testEval.precision(1));
        System.out.println("Recall = " + testEval.recall(1));
        System.out.println("F-Measure = " + testEval.fMeasure(1));
        System.out.println("Error Rate = " + testEval.errorRate());
        System.out.println("Accuracy on Test Set: " + testEval.pctCorrect() + "%");
        System.out.println(testEval.toClassDetailsString());
    }
}
