package model.improvement;

import preprocessing.ImportData;
import model.Command;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;

public class PCA implements Command {
    public static void main(String[] args) {
        Command cmd = new PCA();
        cmd.exec(ImportData.trainSource, ImportData.testSource);
    }
    @Override
    public void exec(DataSource trainSource, DataSource testSource) {
        try {
            // Load the training data
            Instances trainData = trainSource.getDataSet();
            trainData.setClassIndex(trainData.numAttributes() - 1);

            // Load the test data
            Instances testData = testSource.getDataSet();
            testData.setClassIndex(testData.numAttributes() - 1);

            // Create and configure the PCA filter
            PrincipalComponents pca = new PrincipalComponents();
            pca.setInputFormat(trainData);

            // Apply PCA to the training data
            Instances transformedTrainData = Filter.useFilter(trainData, pca);

            // Apply PCA to the test data
            Instances transformedTestData = Filter.useFilter(testData, pca);

            // Output the transformed data
            System.out.println("Transformed Training Data:");
            System.out.println(transformedTrainData);
            System.out.println("Transformed Test Data:");
            System.out.println(transformedTestData);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}