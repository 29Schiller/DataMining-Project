package preprocessing;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class FilterAttribute {
    public static void main(String[] args) {
        Instances newData = null; // Declare newData outside try-catch
        try {
            // Load your dataset here (make sure it's in the right format)
            DataSource source = new DataSource("C:\\Users\\tonga\\OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY\\Documents\\DataMining-Project\\data\\apartment_prices.arff");
            Instances dataset = source.getDataSet();

            // Create a Remove object (this is the filter class)
            Remove remove = new Remove();

            // StringBuilder to hold the indices of attributes to be removed
            StringBuilder indicesToRemove = new StringBuilder();

            // Iterate through the attributes to find those that start with "CityPart_"
            for (int i = 0; i < dataset.numAttributes(); i++) {
                if (dataset.attribute(i).name().startsWith("CityPart_")) {
                    // Append the index (1-based for Weka) to the string
                    if (!indicesToRemove.isEmpty()) {
                        indicesToRemove.append(","); // Separate multiple indices with commas
                    }
                    indicesToRemove.append(i + 1); // Weka uses 1-based indexing
                }
            }

            // Set the indices to remove if any were found
            if (!indicesToRemove.isEmpty()) {
                remove.setAttributeIndices(indicesToRemove.toString());
            } else {
                System.out.println("No attributes starting with 'CityPart_' found.");
                return;
            }

            // Pass the dataset to the filter
            remove.setInputFormat(dataset);

            // Apply the filter
            newData = Filter.useFilter(dataset, remove);

            // Output the new dataset (for example, print the first few instances)
            System.out.println(newData);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Save the new dataset
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(newData);
            saver.setFile(new File("C:\\Users\\tonga\\OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY\\Documents\\DataMining-Project\\data\\apartment_prices.arff")); // Use File instead of FileReader
            saver.writeBatch();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}