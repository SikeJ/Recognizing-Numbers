import java.util.Random;
import java.util.Hashtable;

/**
 * Creates a Neural Network class to easily modify & adjust the network
 * 	that this assignment seeks to create
 * @author Jackson
 */
public class NeuralNetwork {
	
	//Initializes the variables used by the Neural Network
	private int trainingSize;
	public int inputSize;
	public int hiddenSize;
	private int outputSize;
	private int batchSize;
	private int layerSize;
	private double learnRate;
	private int epochSize;
	
	private Random random = new Random();
		
	//Initializes array's capable of handling all of the layers, rows, and cols
	public double[][] bias;
	public double[][][] weights;
	public double[][] trainingCases;	
	public double[][] outputCases;
	private double[][] activation;
	private double[][] sigmoidZ;
	private double[][] delta;
	private double[][] deltaSum;
	private double[][] biasGrad;
	private double[][][] weightGrad;
	public double accurate;
	
	
	/** Allows you to easily edit the size of the input, hidden, output, and mini batch size */
	public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, int batchSize) {
		this.layerSize = 3;
		this.inputSize = inputNodes;
		this.hiddenSize = hiddenNodes;
		this.outputSize = outputNodes;
		this.batchSize = batchSize;
		this.learnRate = 5;
		this.epochSize = 2;
		
		//Creates jagged array's in order to keep the data per layer intack
		//Starts at 1 because the input layer doesn't have anything
		this.weights = new double[2][][];
		this.weights[0] = new double[this.hiddenSize][this.inputSize];
		this.weights[1] = new double[this.outputSize][this.hiddenSize];
		
		this.bias = new double[2][];
		this.bias[0] = new double[hiddenSize];
		this.bias[1] = new double[outputSize];
		
		this.biasGrad = new double[2][];
		this.biasGrad[0] = new double[hiddenSize];
		this.biasGrad[1] = new double[outputSize];
		
		this.weightGrad = new double[2][][];
		this.weightGrad[0] = new double[hiddenSize][inputSize];
		this.weightGrad[1] = new double[outputSize][hiddenSize];
		
		this.activation = new double[2][];
		this.activation[0] = new double[hiddenSize];
		this.activation[1] = new double[outputSize];
		
		this.sigmoidZ = new double[2][];
		this.sigmoidZ[0] = new double[hiddenSize];
		this.sigmoidZ[1] = new double[outputSize];
		
		this.delta = new double[2][];
		this.delta[0] = new double[hiddenSize];
		this.delta[1] = new double[outputSize];
		
		this.deltaSum = new double[2][];
		this.deltaSum[0] = new double[hiddenSize];
		this.deltaSum[1] = new double[outputSize];
		
		//Randomly sets the starting weights for the network
		randomizeWeights();				
	}
	
	/** Allows the weights & bias to be set by a saved state of the network */
	public void setWeightandBias(double[][][] weights, double[][] bias) {
		this.weights = weights;
		this.bias = bias;
	}
	
	/** This sets the inputs & expected outs of the network*/
	public void setInandOut(double[][] inputs, double[][] outs){
		this.trainingCases = inputs;
		this.outputCases = outs;
		this.trainingSize = this.trainingCases.length;
	}
	
	/** This randomizes the weights of each node in the neural network */
	private void randomizeWeights() {
		for(int layer = 0; layer < weights.length; layer++)
			for(int row = 0; row < weights[layer].length; row++) {
				for(int col = 0; col < weights[layer][row].length; col++)
					//Creates a random double between 1 & 0 and then randomly decides if it's negative
					weights[layer][row][col] = random.nextDouble()*(random.nextBoolean() ? -1:1);
				
				//Also randomly set's the bias
				bias[layer][row] = random.nextDouble()*(random.nextBoolean() ? -1:1);
			}
	}
	
	/** This randomizes the order of the input & output array's for the Stochastic Gradient Descent */
	private void randomizeInOut() {
		for(int i = trainingCases.length - 1; i > 0; i--) {
			int m = random.nextInt(i);
			
			double[] tempI = trainingCases[i];
			trainingCases[i] = trainingCases[m];
			trainingCases[m] = tempI;
			
			double[] tempA = outputCases[i];
			outputCases[i] = outputCases[m];
			outputCases[m] = tempA;
		}
	}
	
	/** This tests the newly calculated weights, and records for accuracy */
	public void testNetwork() {
		//Array's that will hold the statistic's for the accuracy of the network
		int[] correct = new int[outputSize];
		int[] total = new int[outputSize];
		
		//Tests all of the input's and compares the output of the network to the real answer
		for(int i = 0; i < trainingCases.length; i++) {
			double[] testing = trainingCases[i];
			double[][] outTest = forwardPass(testing);
			
			//Since the forwardPass function returns the activations of both layers,
			// we only take the activation states of the output layer
			double[] results = outTest[1];
			
			//Loops through the results to compare the output to the expected
			double[] expected = outputCases[i];
			int index = 0;
			
			//Determines the index of the most likely answer
			for(int j = 0; j < results.length; j++)
				index = results[j] > results[index] ? j:index;
			
			//printA(results);
			//printA(out);
			//Checks if the max index is a 1 in the expected result
			if(expected[index] == 1) {
				correct[index]++;
				total[index]++;
			}
			
			//If it isn't find where the expected output is & add it to the total
			else {
				int Eindex = 0;
				for(int j = 0; j < expected.length; j++)
					Eindex = expected[j] == 1 ? j:Eindex;
				
				total[Eindex]++;
			}
			
		}
		
		int totalVal = 0;
		int totalGot = 0;
		
		//Prints out the total results of the test
		for(int row = 0; row < 10; row++) {
			if(row%5 == 0 && row !=0)
				System.out.println();
			
			totalVal += total[row];
			totalGot += correct[row];
			System.out.printf("%d = %d/%d ", row, correct[row], total[row]);
		}
		this.accurate = ((double)totalGot)/totalVal*100;
		System.out.printf("\nAccuracy: %d/%d = %.3f%%\n", totalGot, totalVal, this.accurate);
		
	}
	
	/** This allows the network to be tested for any value of inputs for testing purposes*/
	public double[] testNetwork(double[] input) {
		double[][] outcomes = forwardPass(input, weights);
		return outcomes[1];
	}
	
	/** This function allows the network to be trained */
	public void trainNetwork() {
			
		//System.out.printf("There are %d mini batches to be used.\n", trainingSize/batchSize);
		//System.out.printf("The mini batches contain %d input cases.\n", batchSize);
		
		double[][][] batches = new double[trainingSize/batchSize][batchSize][];
		double[][][] batchesOut = new double[trainingSize/batchSize][batchSize][];
		int batchNum = -1;
		//Loops through the input cases to form the mini batches
		for(int i = 0; i < trainingCases.length; i++) {
			//Keeps track & updates the batch number when the previous batch is complete
			if(i % batchSize == 0)
				batchNum++;
			
			//Assign's input & output cases to the batches
			batches[batchNum][i%batchSize] = trainingCases[i];
			batchesOut[batchNum][i%batchSize] = outputCases[i];
		}
		//Stats about the accuracy before the network goes under training
		System.out.println("Accuracy Before Training:");
		testNetwork();
		System.out.println();
		
		//Loops through the required Epoch's to train the data
		for(int epoch = 0; epoch < epochSize; epoch++) {
			System.out.printf("Epoch %d/%d\n",epoch+1,epochSize);

			//Randomizes the inputs & outputs to match the SGD method
			randomizeInOut();
			
			//Loops through all of the batches,
			for(int num = 0; num < batches.length; num++) {
				double[][] batch = batches[num];
				double[][] batchOut = batchesOut[num];
				
				//Reset's the sum of gradient & bias global variables before starting another mini-batch
				resetVar(deltaSum);
				resetVar(weightGrad);
				
				//Performs the forward & backward pass for everything in the batch
				for(int row = 0; row < batch.length; row++) {
					double[] inputRow = batch[row];
					double[] outputRow = batchOut[row];
					
					//Performs the feed forward & back propagation on the network
					forwardPass(inputRow, this.weights);
					backwardPass(inputRow, outputRow, this.weights);
				}
				
				//Calculates the new weight & bias by adding the gradient to the old weights & bias
				for(int layer = 0; layer < weightGrad.length; layer++)
					for(int row = 0; row < weightGrad[layer].length; row++) {
						for(int col = 0; col < weightGrad[layer][row].length; col++)
							this.weights[layer][row][col] -= (learnRate/batchSize)*weightGrad[layer][row][col];
						
						//takes care of calculating the new bias
						this.bias[layer][row] -= (learnRate/batchSize)*deltaSum[layer][row];		
					}
			}
			testNetwork();
			System.out.println();
		}
		/*
		System.out.println("New Weights:");
		printA(weights);
		System.out.println("New Bias:");
		printA(bias);
		*/
	}
	
	/** Overloaded feed forward pass to calculate the activations */
	private double[][] forwardPass(double[] input){
		return forwardPass(input, weights);
	}
	
	/** This goes forward through the network to calculate the activation states */
	private double[][] forwardPass(double[] input, double[][][] weights) {
		double sum;
		//Has to loop through layer by layer starting at layer 1
		for(int layer = 0; layer < layerSize -1; layer++) {
			//Perform the dot product between the weight & input array
			for(int row = 0; row < weights[layer].length; row++) {
				//System.out.println("Performing Dot Product on:");
				//printA(weights[layer][row]);
				
				//When on the first layer, use the actual inputs as the inputs
				if(layer == 0)
					sum = dotProduct(weights[layer][row], input);
				
				//When on any other layer, use the previous layer activations as the input
				else
					sum = dotProduct(weights[layer][row], activation[layer-1]);
				
				//Calculate & store the sigmoid & activation for the current node.
				sigmoidZ[layer][row] = sum + bias[layer][row];
				activation[layer][row] = sigmoidFunction(sigmoidZ[layer][row]);
			}
		}
		/*
		System.out.println("Z:");
		printA(sigmoidZ);
		System.out.println("A:");
		printA(activation);
		*/
		return activation;
	}
	
	/** Performs back propagation on the network to find the delta & gradient */
	private double[][][] backwardPass(double[] inBatch, double[] outBatch, double[][][] weights) {
		double[][][] gradient;
		gradient = new double[2][][]; 
		gradient[0] = new double[hiddenSize][outputSize]; 
		gradient[1] = new double[hiddenSize][outputSize];
		 
		for(int layer = 1; layer >= 0; layer--) {
			//Performs the delta calculations
			//Accounts for the difference of the delta calculation for the 'first' layer
			if(layer == 1) {
				for(int row = 0; row < activation[layer].length; row++) {
					//Error calculation between the activation value & the expected value
					double act = activation[layer][row];
					delta[layer][row] = (act - outBatch[row])*act*(1 - act);
					deltaSum[layer][row] += delta[layer][row];
				}
			}
			
			//For every other layer do this
			else {
				for(int row = 0; row < activation[layer].length; row++) {
					//Error calculation based off the dot product of the previous layer's weights & calculated delta values
					double act = activation[layer][row];
					double[] nodeDelt = new double[weights[layer+1].length];
					
					//Creates an array based on the previous layer weight's to easily calculate the delta for the current layer
					for(int i = 0; i < nodeDelt.length; i++) { 
						nodeDelt[i] = weights[layer+1][i][row];
					}
					
					
					//Calculates the delta for the current layer & add's it to the mini batch sum
					delta[layer][row] = dotProduct(nodeDelt, delta[layer+1]) * act * (1 - act);
					deltaSum[layer][row] += delta[layer][row];
				}
			}
			
			//Calculates the gradient matrix for the current layer
			//Uses the input values if the current layer is the lowest hidden layer
			if(layer == 0)
				gradient[layer] = multiplyMatrix(delta[layer], inBatch);
				
			//Any other layer in the neural network
			else
				gradient[layer] = multiplyMatrix(delta[layer], activation[layer-1]);
			
		}
		//Keeps track of the summation of all the gradients in a mini batch
		for(int layer = 0; layer < gradient.length; layer++) {
			for(int row = 0; row < gradient[layer].length; row++) {
				for(int col = 0; col < gradient[layer][row].length; col++) {
					weightGrad[layer][row][col] += gradient[layer][row][col];
				}
			}
		}
		//Returns the gradient in case it needs to be used somewhere
		return gradient;
	}
	
	/** Creates a NxM matrix with two matrices provided, Nx1 & 1xM */
	private double[][] multiplyMatrix(double[] N, double[] M){
		double[][] matrix = new double[N.length][M.length];
		
		for(int j = 0; j < N.length; j++)
			for(int k = 0; k < M.length; k++)
				matrix[j][k] = N[j]*M[k];
		
		return matrix;
	}
	
	/** Function that performs the dot product of two arrays */
	private double dotProduct(double[] weight, double[] input) {
		double sum = 0;
		
		for(int i = 0; i < weight.length; i++)
			sum += weight[i]*input[i];
		return sum;
	}
	
	/** Function that calculates the sigmoid activation from a given z */
	private double sigmoidFunction(double z)	{
		return 1/(1 + Math.exp(-z));
	}
	
	/** Reset's the array given to all 0's */
	public void resetVar(double[] array) {
		for(int i = 0; i < array.length; i++)
			array[i] = 0;
	}
	
	/** Overloads it to allow to reset multi-dimensional array's */
	public void resetVar(double[][] array) {
		for(int i = 0; i < array.length; i++)
			resetVar(array[i]);
	}
	
	/** Overloads to allow for three-dimensional array's */
	public void resetVar(double[][][] array) {
		for(int i = 0; i < array.length; i++) {
			resetVar(array[i]);
		}
	}
	
	/** This allows the updated weights & bias to be printed after each Epoch */
	public void printWeightsBias() {
		System.out.println("Weights:\t\t\tBias:");
		for(int layer = 0; layer < weights.length; layer++) {
			System.out.printf("Layer: %d\n", layer);
			for(int row = 0; row < weights[layer].length; row++) {
				for(int col = 0; col < weights[layer][row].length; col++) {
					System.out.printf("%.3f, ", weights[layer][row][col]);
				}
				System.out.printf("\t%.3f\n", bias[layer][row]);
			}
		}
	}
	
	/** This function prints the N-Dimensional array's out */
	public void printA(double[] array) {
		for(int i = 0; i < array.length; i++)
			System.out.printf("%.3f, ", array[i]);
		System.out.println();
	}
	
	/** Overloaded function to print 2-D array's */
	public void printA(double[][] array) {
		for(int i = 0; i < array.length; i++) {
			for(int k = 0; k < array[i].length; k++)
				System.out.printf("%.3f, ", array[i][k]);
			System.out.println();
		}
	}
	
	/** Overloaded function to print 3-D array's */
	public void printA(double[][][] array) {
		for(int i = 0; i < array.length; i++) {
			System.out.printf("Layer %d:\n", i+1);
			for(int k = 0; k < array[i].length; k++) {
				for(int j = 0; j < array[i][k].length; j++)
					System.out.printf("%.3f, ", array[i][k][j]);
				System.out.println();
			}
		}
	}
}
