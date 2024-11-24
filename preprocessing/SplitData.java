package preprocessing;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.util.Random;

public class SplitData {

    public static void splitData() throws Exception {
        // Create an instance of DataSource to load the dataset
        DataSource source = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\data\\apartment_prices.arff"); // Replace with your dataset path
        Instances allData = source.getDataSet(); // Use the instance to call getDataSet()

        // Set the class index (if not already set)
        if (allData.classIndex() == -1) {
            allData.setClassIndex(allData.numAttributes() - 1); // Assume the last attribute is the class
        }

        // Randomize the dataset with a seed for reproducibility
        allData.randomize(new Random(42));

        // Split the dataset into training, testing, and validation sets
        int totalInstances = allData.numInstances();
        int testSize = (int) Math.round(totalInstances * 0.2); // 20% for testing
        int validationSize = (int) Math.round((totalInstances - testSize) * 0.2); // 20% of the remaining for validation
        int trainSize = totalInstances - testSize - validationSize; // Remaining for training

        // Create the subsets
        Instances testData = new Instances(allData, 0, testSize);
        Instances validationData = new Instances(allData, testSize, validationSize);
        Instances trainData = new Instances(allData, testSize + validationSize, trainSize);

        // Save the subsets as separate ARFF files
        saveInstances(trainData, "C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\data\\training_data.arff");
        saveInstances(testData, "C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\data\\testing_data.arff");
        saveInstances(validationData, "C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\data\\evaluation_data.arff");

        System.out.println("Data split and saved successfully:");
        System.out.println("Training data: " + trainSize + " instances");
        System.out.println("Validation data: " + validationSize + " instances");
        System.out.println("Test data: " + testSize + " instances");
    }

    private static void saveInstances(Instances data, String fileName) throws Exception {
        // Save an Instances object to an ARFF file
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(fileName));
        saver.writeBatch();
    }

    public static void main(String[] args) {
        try {
            splitData();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
