package preprocessing;

import weka.core.converters.ConverterUtils.DataSource;

public class ImportData {

    public static DataSource dataSource;
    public static DataSource trainSource;
    public static DataSource testSource;
    public static DataSource validSource;

    static {

        try {
            dataSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\data\\apartment_prices.arff");
            trainSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\data\\apartment_prices.arff");
            testSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\data\\apartment_prices.arff");
            validSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\data\\apartment_prices.arff");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }


}