package RulesClassifier.BasedModel;

import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;


public class ZeroRClassifier {
    public static void main(String[] args) throws Exception {
        DataSource trainSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\BeforeRedDim\\src\\training_data.arff");
        Instances trainDataset = trainSource.getDataSet();
        DataSource testSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\BeforeRedDim\\src\\testing_data.arff");
        Instances testDataset = trainSource.getDataSet();

        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
        testDataset.setClassIndex(testDataset.numAttributes() - 1);


        // Create and train the OneR classifier
        weka.classifiers.rules.ZeroR zeror = new weka.classifiers.rules.ZeroR();
        zeror.buildClassifier(trainDataset);

        System.out.println("ZeroR params" + String.join(" ", zeror.getOptions()));

        Evaluation eval = new Evaluation(trainDataset);
        Random rand = new Random(42);
        int folds = 10;
        eval.crossValidateModel(zeror, testDataset, folds, rand);

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
