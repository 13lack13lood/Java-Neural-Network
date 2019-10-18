package nn;

import java.util.Arrays;
import nn.NetworkTools.*;

public class Network {
	private double[][] output;
	//layer, neuron
	private double[][][] weights;
	//layer, neuron, prev neuron
	private double[][] bias;
	//layer, neuron
	
	public double[][] error_signal;
	public double[][] output_dx;
	
	public final int[] networkLayerSizes;
	public final int inputSize;
	public final int outputSize;
	public final int networkSize;
		
	public Network(int... networkLayerSizes) {
		this.networkLayerSizes = networkLayerSizes;
		this.inputSize = networkLayerSizes[0];
		this.networkSize = networkLayerSizes.length;
		this.outputSize = networkLayerSizes[networkSize - 1];
		
		this.output = new double[networkSize][];
		this.weights = new double[networkSize][][];
		this.bias = new double[networkSize][];
		
		this.error_signal = new double[networkSize][];
		this.output_dx = new double[networkSize][];
		
		for(int i = 0; i < networkSize; i++) {
			this.output[i] = new double[networkLayerSizes[i]];
			this.error_signal[i] = new double[networkLayerSizes[i]];
			this.output_dx[i] = new double[networkLayerSizes[i]];
			
			this.bias[i] = NetworkTools.createRandomArray(networkLayerSizes[i], 0.3, 0.8);
			
			if(i > 0) {
				
				weights[i] = NetworkTools.createRandomArray(networkLayerSizes[i], networkLayerSizes[i - 1], -0.5, 0.5);
			}
		}
	}
	
	public double[] calculate(double... input) {
		if(input.length != this.inputSize) 
			return null;
		
		this.output[0] = input;
		
		for(int layer = 1; layer < networkSize; layer++) {
			for(int neuron = 0; neuron < networkLayerSizes[layer]; neuron++) {
				double sum = bias[layer][neuron];
				
				for(int prevNeuron = 0; prevNeuron < networkLayerSizes[layer - 1]; prevNeuron++) {
					sum += output[layer - 1][prevNeuron] * weights[layer][neuron][prevNeuron];
				}
				
				output[layer][neuron] = sigmoid(sum);
				output_dx[layer][neuron] = output[layer][neuron] * (1- output[layer][neuron]);
			}
		}
		return output[networkSize - 1];
	}
	
	public void train(double[] input, double[] target, double eta) {
		if(input.length != inputSize || target.length != outputSize)
			return;
		
		calculate(input);
		backpropError(target);
		updateWeights(eta);
		
	}
	
	public void backpropError(double[]target) {
		for(int neuron = 0; neuron < networkLayerSizes[networkSize - 1]; neuron++) {
			error_signal[networkSize - 1][neuron] = (output[networkSize - 1][neuron] - target[neuron]) * output_dx[networkSize - 1][neuron];
		}
		
		for(int layer = networkSize - 2; layer > 0; layer--) {
			for(int neuron = 0; neuron < networkLayerSizes[layer]; neuron++) {
				double sum = 0;
				
				for(int nextN = 0; nextN < networkLayerSizes[layer + 1]; nextN++) {
					sum += weights[layer + 1][nextN][neuron] * error_signal[layer + 1][nextN];
				}
				
				this.error_signal[layer][neuron] = sum * output_dx[layer][neuron];
			}
		}
	}
	
	public void updateWeights(double eta) {
		for(int layer = 1; layer < networkSize; layer++) {
			for(int neuron = 0; neuron < networkLayerSizes[layer]; neuron++) {
				for(int prevNeuron = 0; prevNeuron < networkLayerSizes[layer - 1]; prevNeuron++) {
					double delta = -eta * output[layer - 1][prevNeuron] * error_signal[layer][neuron];
					weights[layer][neuron][prevNeuron] += delta;
				}
				
				double delta = -eta * error_signal[layer][neuron];
				bias[layer][neuron] += delta;
			}
		}
	}
	
	private double sigmoid(double x) {
		return 1d / (1 + Math.exp(-x));
	}
	
	public static void main(String[] args) {
		Network net = new Network(4, 1, 3, 4);
		
		double[] input = new double[] {0.1, 0.5, 0.6, 0.8};
		double[] target = new double[] {0, 1, 0, 0};
		
		double[] in2 = new double[] {0.6, 0.3, 0.5, 0.4};
		double[] target2 = new double[] {0.1, 0.9, 0.1, 0};
		
		for(int i = 0; i < 1000000; i++) {
			net.train(input, target, 0.3);
			net.train(in2, target2, 0.5);
		}
		
		double[] o = net.calculate(input);
		double[] o2 = net.calculate(in2);
				
		System.out.println(Arrays.toString(o));
		System.out.println(Arrays.toString(o2));
	}
}
