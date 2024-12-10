package TreeClassifier.TuningModel;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class PrunedTreeTuning {

    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource trainSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining - Copy\\src\\Data\\training_data.arff");
        Instances trainDataset = trainSource.getDataSet();
        DataSource testSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining - Copy\\src\\Data\\testing_data.arff");
        Instances testDataset = testSource.getDataSet();
        DataSource validSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining - Copy\\src\\Data\\validation_data.arff");
        Instances validDataSet = validSource.getDataSet();

        // Set class index to the last attribute (target variable)
        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
        testDataset.setClassIndex(testDataset.numAttributes() - 1);
        validDataSet.setClassIndex(validDataSet.numAttributes() - 1);

        // Set up CVParameterSelection to tune hyperparameters
        CVParameterSelection paramSelection = new CVParameterSelection();
        paramSelection.setClassifier(new J48());
        paramSelection.setNumFolds(10);
        // Define the parameter ranges for tuning
        paramSelection.addCVParameter("M 2 10 1");    // Minimum number of instances per leaf (2 to 10 in steps of 1)
        paramSelection.buildClassifier(validDataSet);

        // Print the best parameters
        System.out.println("Best Parameters: " + String.join(" ", paramSelection.getBestClassifierOptions()));

        // Train the final model with the best parameters on the entire dataset
        J48 j48tuning = new J48();
        j48tuning.setOptions(paramSelection.getBestClassifierOptions());
        j48tuning.buildClassifier(trainDataset);

        // Evaluate the tuned classifier using the test dataset
        Evaluation eval = new Evaluation(trainDataset);
        int folds = 10;
        eval.crossValidateModel(j48tuning, testDataset, folds, new Random(1));

        // Print the confusion matrix
        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        // Print additional evaluation metrics
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString());
    }
}