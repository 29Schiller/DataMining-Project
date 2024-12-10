package RulesClassifier.TuningModel;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class OneRTuning {
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
        paramSelection.setClassifier(new OneR());
        paramSelection.setNumFolds(10);
        // Define the parameter ranges for tuning
        paramSelection.addCVParameter("B 2 50 5");
        paramSelection.buildClassifier(validDataSet);

        // Print the best parameters
        System.out.println("Best Parameters: " + String.join(" ", paramSelection.getBestClassifierOptions()));

        // Train the final model with the best parameters on the entire dataset
        OneR onertuning = new OneR();
        onertuning.setOptions(paramSelection.getBestClassifierOptions());
        onertuning.buildClassifier(trainDataset);

        // Evaluate the tuned classifier using the test dataset
        Evaluation testEval = new Evaluation(trainDataset);
        int folds = 10;
        testEval.crossValidateModel(onertuning, testDataset, folds, new Random(1));

        // Output evaluation results
        System.out.println(testEval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + testEval.precision(1));
        System.out.println("Recall = " + testEval.recall(1));
        System.out.println("F-Measure = " + testEval.fMeasure(1));
        System.out.println("Error Rate = " + testEval.errorRate());
        System.out.println(testEval.toClassDetailsString());
    }
}
