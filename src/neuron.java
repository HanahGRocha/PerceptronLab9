import java.util.Arrays;

public class neuron {
    private double[] weights;
    private double output;
    private double[] inputs;
    private double alpha = 0.1; // Learning rate
    private double delta; // Error term for backpropagation

    public neuron(int inputSize) {
        weights = new double[inputSize + 1]; // +1 for bias
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() - 0.5; // Random initialization between -0.5 and 0.5
        }
    }

    // Sigmoid activation function
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Derivative of sigmoid for backpropagation
    private double sigmoidDerivative(double x) {
        return x * (1.0 - x); // where x is already sigmoid(x)
    }

    // Forward pass: compute output
    public double forward(double[] inputVec) {
        this.inputs = Arrays.copyOf(inputVec, inputVec.length);
        double sum = 0.0;

        for (int i = 0; i < inputVec.length; i++) {
            sum += inputVec[i] * weights[i];
        }
        sum += 1 * weights[weights.length - 1]; // Bias

        output = sigmoid(sum);
        return output;
    }

    // Backpropagation: compute delta and update weights
    public void backward(double targetOrDelta, boolean isOutputNeuron) {
        if (isOutputNeuron) {
            delta = (targetOrDelta - output) * sigmoidDerivative(output);
        } else {
            delta = targetOrDelta * sigmoidDerivative(output);
        }

        for (int i = 0; i < inputs.length; i++) {
            weights[i] += alpha * delta * inputs[i];
        }
        weights[weights.length - 1] += alpha * delta; // Bias update
    }

    public double getOutput() {
        return output;
    }

    public double getDelta() {
        return delta;
    }

    public double[] getWeights() {
        return weights;
    }

    public void adjustWeights(double[] deltasFromNextLayer, double[] nextLayerWeights) {
        double sum = 0.0;
        for (int i = 0; i < deltasFromNextLayer.length; i++) {
            sum += deltasFromNextLayer[i] * nextLayerWeights[i];
        }
        backward(sum, false);
    }
}
