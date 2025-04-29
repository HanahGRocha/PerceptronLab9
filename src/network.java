public class network {
    private layer hiddenLayer;
    private layer outputLayer;

    public network(int inputSize, int hiddenSize, int outputSize) {
        hiddenLayer = new layer(hiddenSize, inputSize);
        outputLayer = new layer(outputSize, hiddenSize);
    }

    // Forward pass: input -> hidden -> output
    public double[] forward(double[] inputs) {
        double[] hiddenOutputs = hiddenLayer.forward(inputs);
        return outputLayer.forward(hiddenOutputs);
    }

    // Backward pass
    public void backward(double[] expected) {
        outputLayer.backwardOutputLayer(expected);
        hiddenLayer.backwardHiddenLayer(outputLayer);
    }

    // Train on one example
    public double train(double[] inputs, double[] expected) {
        double[] outputs = forward(inputs);
        backward(expected);
        return computeL1Error(outputs, expected); // Return error for this sample
    }

    // Predict (forward pass only)
    public double[] predict(double[] inputs) {
        return forward(inputs);
    }

    // Mean absolute error for a single sample
    private double computeL1Error(double[] predicted, double[] actual) {
        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            sum += Math.abs(predicted[i] - actual[i]);
        }
        return sum;
    }

    // Get 1-hot prediction based on max value
    public int[] getOneHotPrediction(double[] outputs) {
        int[] oneHot = new int[outputs.length];
        int maxIndex = 0;
        for (int i = 1; i < outputs.length; i++) {
            if (outputs[i] > outputs[maxIndex]) {
                maxIndex = i;
            }
        }
        oneHot[maxIndex] = 1;
        return oneHot;
    }
}
