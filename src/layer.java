import java.util.ArrayList;
import java.util.List;

public class layer {
    private List<neuron> neurons;

    public layer(int numNeurons, int inputSizePerNeuron) {
        neurons = new ArrayList<>();
        for (int i = 0; i < numNeurons; i++) {
            neurons.add(new neuron(inputSizePerNeuron));
        }
    }

    // Forward pass for the layer
    public double[] forward(double[] inputs) {
        double[] outputs = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            outputs[i] = neurons.get(i).forward(inputs);
        }
        return outputs;
    }

    // Backward pass for output layer
    public void backwardOutputLayer(double[] expected) {
        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).backward(expected[i], true);
        }
    }

    // Backward pass for hidden layer
    public void backwardHiddenLayer(layer nextLayer) {
        for (int i = 0; i < neurons.size(); i++) {
            double sum = 0.0;
            for (neuron nextNeuron : nextLayer.getNeurons()) {
                sum += nextNeuron.getDelta() * nextNeuron.getWeights()[i];
            }
            neurons.get(i).backward(sum, false);
        }
    }

    public List<neuron> getNeurons() {
        return neurons;
    }

    public double[] getOutputs() {
        double[] outputs = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            outputs[i] = neurons.get(i).getOutput();
        }
        return outputs;
    }
}
