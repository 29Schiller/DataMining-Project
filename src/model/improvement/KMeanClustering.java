package model.improvement;

import model.Command;
import preprocessing.ImportData;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class KMeanClustering implements Command {
    public static void main(String[] args) {
        Command cmd = new KMeanClustering();
        cmd.exec(ImportData.trainSource, ImportData.testSource);
    }
    @Override
    public void exec(DataSource trainSource, DataSource testSource) {
        try {
            Instances trainData = trainSource.getDataSet();

            SimpleKMeans model = new SimpleKMeans();

            model.setNumClusters(4);

            model.buildClusterer(trainData);
            System.out.println(model);

            ClusterEvaluation clsEval = new ClusterEvaluation();

            Instances testData = testSource.getDataSet();

            clsEval.setClusterer(model);
            clsEval.evaluateClusterer(testData);

            System.out.println("# of clusters: " + clsEval.getNumClusters());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}