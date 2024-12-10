import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import java.util.Random;

public class NaiveBayesClassifier {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource trainSource = new DataSource("src\\Data\\training_data.arff");
        Instances trainDataset = trainSource.getDataSet();
        DataSource testSource = new DataSource("src\\Data\\testing_data.arff");
        Instances testDataset = trainSource.getDataSet();

        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
        testDataset.setClassIndex(testDataset.numAttributes() - 1);

        // Create and build the classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(trainDataset);

        System.out.println("NB params" + String.join(" ", nb.getOptions()));

        Evaluation eval = new Evaluation(trainDataset);
        eval.evaluateModel(nb, testDataset);

        // Output the evaluation results
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));


        //loop through the new dataset and make predictions
        System.out.println("===================");
        System.out.println("Actual Class, NB Predicted");
        for (int i = 0; i < testDataset.numInstances(); i++) {
            //get class double value for current instance
            double actualClass = testDataset.instance(i).classValue();
            //get class string value using the class index using the class's int value
            String actual = testDataset.classAttribute().value((int) actualClass);
            //get Instance object of current instance
            Instance newInst = testDataset.instance(i);
            //call classifyInstance, which returns a double value for the class
            double predNB = nb.classifyInstance(newInst);
            //use this value to get string value of the predicted class
            String predString = testDataset.classAttribute().value((int) predNB);
            System.out.println(actual + ", " + predString);
        }
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
