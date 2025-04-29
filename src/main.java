import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class main {
    public static void main(String[] args) {
        String filePath = "iris.csv"; // your updated dataset
        List<double[]> features = new ArrayList<>();
        List<double[]> targets = new ArrayList<>();

        // Read CSV file
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",");
                double[] featureRow = new double[4];
                double[] targetRow = new double[3];

                for (int i = 0; i < 4; i++) {
                    featureRow[i] = Double.parseDouble(tokens[i]);
                }

                for (int i = 0; i < 3; i++) {
                    targetRow[i] = Double.parseDouble(tokens[4 + i]);
                }

                features.add(featureRow);
                targets.add(targetRow);
            }
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        // Initialize network
        network network = new network(4, 6, 3); // 6 hidden neurons
        int epochs = 200;
        int totalSamples = features.size();

        PrintWriter writer = null;
        try {
            writer = new PrintWriter(new FileWriter("training_results.txt"));
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0.0;

            for (int i = 0; i < totalSamples; i++) {
                double[] x = features.get(i);
                double[] y = targets.get(i);

                double error = network.train(x, y);
                totalError += error;

                double[] prediction = network.predict(x);
                String predictionStr = String.format(
                        "Epoch %d, Sample %d: Prediction = %s, Target = %s",
                        epoch + 1, i + 1,
                        Arrays.toString(prediction),
                        Arrays.toString(y));
                System.out.println(predictionStr);
                writer.println(predictionStr);
            }

            double mad = totalError / totalSamples;
            String madStr = String.format("Epoch %d Results: MAD = %.4f", epoch + 1, mad);
            System.out.println(madStr);
            writer.println(madStr);

            if (mad < 0.1) {
                System.out.println("Early stopping: MAD is low enough.");
                writer.println("Early stopping: MAD is low enough.");
                break;
            }
        }

        // Final evaluation
        int correctPredictions = 0;

        for (int i = 0; i < features.size(); i++) {
            double[] input = features.get(i);
            double[] target = targets.get(i);

            double[] output = network.predict(input);
            int[] predictedOneHot = network.getOneHotPrediction(output);

            boolean isCorrect = Arrays.equals(predictedOneHot, Arrays.stream(target).mapToInt(d -> (int) d).toArray());
            if (isCorrect) {
                correctPredictions++;
            }
        }

        double accuracy = (correctPredictions * 100.0) / features.size();
        String finalAccuracy = String.format("Final Training Accuracy: %.2f%% (%d/%d correct)",
                accuracy, correctPredictions, features.size());
        System.out.println(finalAccuracy);
        writer.println(finalAccuracy);

        writer.close();
    }
}
