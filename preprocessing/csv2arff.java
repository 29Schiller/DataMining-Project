package preprocessing;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;

public class csv2arff {

    public static void main(String[] args) throws Exception {

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/apartment_prices.csv"));
        Instances data = loader.getDataSet();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("data/apartment_prices.arff"));
        saver.writeBatch();
    }
}