package preprocessing;

import java.io.*;
import java.util.*;

public class PreprocessData {
    public static void main(String[] args) {
        String inputFilePath = "C:\\Users\\tonga\\OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY\\Documents\\DataMining-Project\\data\\Apartment Prices.csv";
        String outputFilePath = "C:\\Users\\tonga\\OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY\\Documents\\DataMining-Project\\data\\apartment_prices.csv";

        try {
            // Load the dataset
            List<String> headers = new ArrayList<>();
            List<Map<String, String>> records = new ArrayList<>();

            try (BufferedReader br = new BufferedReader(new FileReader(inputFilePath))) {
                String line = br.readLine();

                // Read headers
                if (line != null) {
                    headers = Arrays.asList(line.split(","));
                }

                // Read rows
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(",", -1); // Handle empty values
                    Map<String, String> record = new HashMap<>();
                    for (int i = 0; i < headers.size(); i++) {
                        record.put(headers.get(i), values[i]);
                    }
                    records.add(record);
                }
            }

            // Split into categorical, numeric, and target variables
            List<String> numericColumns = new ArrayList<>();
            List<String> categoricalColumns = new ArrayList<>();

            for (String header : headers) {
                if (header.equals("PRICE (GEL)") || isNumeric(records.get(0).get(header))) {
                    if (!header.equals("PRICE (GEL)")) {
                        numericColumns.add(header);
                    }
                } else {
                    categoricalColumns.add(header);
                }
            }

            List<Double> target = new ArrayList<>();
            for (Map<String, String> record : records) {
                target.add(Double.parseDouble(record.get("PRICE (GEL)")));
            }

            // Handle missing values for numeric columns
            for (String numericColumn : numericColumns) {
                List<Double> columnValues = new ArrayList<>();

                for (Map<String, String> record : records) {
                    String value = record.get(numericColumn);
                    if (value != null && !value.isEmpty()) {
                        columnValues.add(Double.parseDouble(value));
                    }
                }

                double median = getMedian(columnValues);

                for (Map<String, String> record : records) {
                    String value = record.get(numericColumn);
                    if (value == null || value.isEmpty()) {
                        record.put(numericColumn, String.valueOf(median));
                    }
                }
            }

            // Convert categorical columns to dummy variables
            Map<String, Set<String>> categoricalValues = new HashMap<>();
            for (String categoricalColumn : categoricalColumns) {
                Set<String> uniqueValues = new HashSet<>();
                for (Map<String, String> record : records) {
                    String value = record.get(categoricalColumn);
                    uniqueValues.add(value == null || value.isEmpty() ? "NaN" : value);
                }
                categoricalValues.put(categoricalColumn, uniqueValues);
            }

            List<Map<String, String>> newRecords = new ArrayList<>();
            for (Map<String, String> record : records) {
                Map<String, String> newRecord = new HashMap<>(record);

                for (String categoricalColumn : categoricalColumns) {
                    String value = record.get(categoricalColumn);
                    if (value == null || value.isEmpty()) {
                        value = "NaN";
                    }

                    for (String uniqueValue : categoricalValues.get(categoricalColumn)) {
                        String columnName = categoricalColumn + "_" + uniqueValue;
                        newRecord.put(columnName, uniqueValue.equals(value) ? "1" : "0");
                    }
                }

                // Remove original categorical columns
                for (String categoricalColumn : categoricalColumns) {
                    newRecord.remove(categoricalColumn);
                }

                newRecords.add(newRecord);
            }

            // Save processed data to CSV
            Set<String> outputHeaders = new LinkedHashSet<>(newRecords.get(0).keySet());

            try (BufferedWriter bw = new BufferedWriter(new FileWriter(outputFilePath))) {
                // Write headers
                bw.write(String.join(",", outputHeaders));
                bw.newLine();

                // Write rows
                for (Map<String, String> record : newRecords) {
                    List<String> row = new ArrayList<>();
                    for (String header : outputHeaders) {
                        row.add(record.get(header));
                    }
                    bw.write(String.join(",", row));
                    bw.newLine();
                }
            }

            System.out.println("Processed data saved to " + outputFilePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static boolean isNumeric(String str) {
        if (str == null || str.isEmpty()) return false;
        try {
            Double.parseDouble(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private static double getMedian(List<Double> values) {
        Collections.sort(values);
        int size = values.size();
        if (size % 2 == 0) {
            return (values.get(size / 2 - 1) + values.get(size / 2)) / 2.0;
        } else {
            return values.get(size / 2);
        }
    }
}
