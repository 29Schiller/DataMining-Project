package FunctionClassifier.BaseModel;

import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.SMO;

public class SVMClassifier {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource trainSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining - Copy\\src\\Data\\training_data.arff");
        Instances trainDataset = trainSource.getDataSet();
        DataSource testSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining - Copy\\src\\Data\\testing_data.arff");
        Instances testDataset = trainSource.getDataSet();

        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
        testDataset.setClassIndex(testDataset.numAttributes() - 1);

        // Create and build the classifier
        SMO smo = new SMO();
        smo.buildClassifier(trainDataset);
        String[] options = {
                "-C", "1.0",
                "-L", "0.001",
                "-P", "1.0E-12",
                "-N", "0",
                "-V", "-1",
                "-W", "1",
                "-K", "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007", // Polynomial kernel
                "-calibrator", "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4"
        };
        smo.setOptions(options);

        System.out.println("SVM params" + String.join(" ", smo.getOptions()));

        weka.classifiers.evaluation.Evaluation eval = new Evaluation(trainDataset);
        eval.evaluateModel(smo, testDataset);

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
