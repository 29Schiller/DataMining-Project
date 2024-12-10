package preprocessing;

import weka.core.converters.ConverterUtils.DataSource;

public class ImportData {

    public static DataSource dataSource;
    public static DataSource trainSource;
    public static DataSource testSource;
    public static DataSource validSource;

    static {

        try {
            dataSource = new DataSource("C:\\Users\\tonga\\OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY\\Documents\\DataMining-Project\\data\\apartment_prices.arff");
            trainSource = new DataSource("C:\\Users\\tonga\\OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY\\Documents\\DataMining-Project\\data\\training_data.arff");
            testSource = new DataSource("C:\\Users\\tonga\\OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY\\Documents\\DataMining-Project\\data\\testing_data.arff");
            validSource = new DataSource("C:\\Users\\tonga\\OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY\\Documents\\DataMining-Project\\data\\evaluation_data.arff");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }


}