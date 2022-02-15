/*
 * Name: Jackson Sikes
 * CWID: 10260611
 * Date: 10/28/21
 * Assignment: MNIST AI Training Program
 * Description: This program trains a multi-layered network to recognize hand written numbers.  
 */

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;


public class Main {
	
	//Initializes the variables to use throughout the entire program
	static Hashtable<String, double[]> hotVector = new Hashtable<String, double[]>();
	
	//Sets up the initial neural network for the program
	static int inputNum = 784;
	static int hiddenNum = 10;
	static int outputNum = 10;
	static int batchNum = 10;
	static NeuralNetwork network = new NeuralNetwork(inputNum, hiddenNum, outputNum, batchNum);
	
	//Allows this program to be ran on different machines and still work properly
	static String address = System.getProperty("user.dir");
	static File trainingFile = new File(address + "/src/mnist_train.csv");
	static File testingFile = new File(address + "/src/mnist_test.csv");
	
	//Starts the IO for the program
	static Scanner choice = new Scanner(System.in);
	
	//Ways to store what the current inputs & correct outputs are being provided
	static double[][] inputForNet;
	static double[][] outputForNet;
	
	/** Start point of this program 
	 * @throws IOException */
	public static void main(String[] args) throws IOException{
		
		//If this program is run through cmd, edit the file path
		if(System.console() != null) {
			address = System.getProperty("user.dir");
			trainingFile = new File(address + "/mnist_train.csv");
			testingFile = new File(address + "/mnist_test.csv");
		}
		
		//Creates hot vectors for 0-9 & puts all of them into a dictionary to easily have the results
		for(int i = 0; i < 10; i++) {
			double[] truth = new double[10];
			truth[i] = 1.0;
			hotVector.put(String.valueOf(i), truth);
		}
		boolean stay = true;
		boolean stay2 = true;
		clearScreen();
		System.out.println("Welcome to my first neural network...");
		while(stay) {
			//Clears the screen and shows the first menu option
			System.out.println("Please follow the instructions on the screen...");
			System.out.println("[1] to train the network with random weights.");
			System.out.println("[2] to load a pre-trained network.");
			System.out.println("[0] to abandon the program");
			System.out.print("Chose Option: ");
			String chose = choice.nextLine();
			int chosen;
			try {
				chosen = Integer.valueOf(chose);
				switch(chosen) {
				case 1:
					clearScreen();
					System.out.println("Training the network with random weights");
					dataReciever(trainingFile);
					network.trainNetwork();
					pauseForASec();
					stay = false;
					break;
					
				case 2:
					clearScreen();
					pickFile();
					pauseForASec();
					stay = false;
					break;
				
				case 0:
					clearScreen();
					stay = false;
					stay2 = false;
					break;
					
				default:
					clearScreen();
					System.out.println("ERROR: Please enter one of the options below.");
				}
			}catch(Exception InputMismatchException) {
				clearScreen();
				System.out.println("ERROR: Please enter a number. ");
			}
		}
		
		boolean error = false;
		String errorS = "";
		while(stay2) {
			clearScreen();
			if(error) {
				error = false;
				System.out.println("ERROR: " + errorS);	
			}
			System.out.println("Neural Network Program select what you would like to do:");
			System.out.println("[1] Trains the network with random weights.");
			System.out.println("[2] Loads a pre-trained network.");
			System.out.println("[3] Network's accuracy on the training data");
			System.out.println("[4] Network's accuracy on the testing data");
			System.out.println("[5] Saves the state of the network to a file");
			System.out.println("[6] Shows all classifications of the testing data with ASCII art");
			System.out.println("[7] Shows only incorrect classifications of the testing data with ASCII art");
			System.out.println("[0] Abandons the program");

			
			System.out.print("Chose Option: ");
			int commandIn;
			
			//The try catch block ensures that only an int was supplied into the console
			try {
				//Allows user to chose what they want to do with the neural network
				String chose = choice.nextLine();
				commandIn = Integer.valueOf(chose);
				
				//After they decide, clear the screen & do what they said 
				//	depending on the chosen option
				clearScreen();
				switch(commandIn) {
				//Trains the network randomly
				case 1:
					System.out.println("Training the network with random weights");
					dataReciever(trainingFile);
					network.trainNetwork();
					pauseForASec();
					break;
					
				//Selects a previous network's state	
				case 2:
					pickFile();
					pauseForASec();
					break;
				
				//Displays the accuracy over the training set
				case 3:
					System.out.println("The accuracy for the network over the training set is:");
					dataReciever(trainingFile);
					findAccuracy();
					pauseForASec();
					break;
					
				//Displays the accuracy over the testing set
				case 4:
					System.out.println("The accuracy for the network over the testing set is:");
					dataReciever(testingFile);
					findAccuracy();
					pauseForASec();
					break;
				
				//Saves the current network's weights to a file
				case 5:
					saveWeights();
					pauseForASec();
					break;
					
				//Shows all of the test cases as ASCII art
				case 6:
					showASCII(false);
					break;
					
				//Shows only the incorrect classifications
				case 7:
					showASCII(true);
					break;
				
				//Leaves the program
				case 0:
					stay2 = false;
					break;
					
				//If the number supplied isn't one of the options give an error & reset
				default:
					errorS = "Please enter one of the options below.";
					error = true;
					break;
				}
			}catch(Exception InputMismatchException) {
				error = true;
				errorS = "Please enter a number";
			}			
		}
		choice.close();
	}
	
	/** Function that saves the weight's of the network to a file 
	 * @throws IOException */
	public static void saveWeights() throws IOException {
		//Creates a new file, with the accuracy, time, & day it was calculated as it's name
		dataReciever(trainingFile);
		System.out.println("Current Network Training Accuracy:");
		double percent = findAccuracy();
		Date date = new Date();
		SimpleDateFormat dateFormat = new SimpleDateFormat("HH-mm-ss dd-MM-yyyy");
		String fileName = String.format("%.2f", percent) + " " + dateFormat.format(date);
		if(System.console() != null)
			fileName = System.getProperty("user.dir").replace("src", "") + fileName;
		File file = new File(fileName + ".csv");
		FileWriter write = new FileWriter(file);
		
		//Writes the layer size & inputs of the current neural network to the file
		int[] printNums = {inputNum, hiddenNum, outputNum, batchNum};
		for(int i = 0; i < printNums.length; i++)
			write.write(String.valueOf(printNums[i]) + ',');
		write.write("\n");
		
		//Writes the weights & bias by layer, separating them by new rows
		double[][][] weights = network.weights;
		double[][] bias = network.bias;
		for(int layer = 0; layer < weights.length; layer++) {
			//Prints the weights for a particular layer
			for(int row = 0; row < weights[layer].length; row++) {
				for(int col = 0; col < weights[layer][row].length; col++)
					write.write(String.valueOf(weights[layer][row][col] + ","));
				
				write.write("\n");
				
			}
			
			//Prints the bias for the same layer
			for(int row = 0; row < bias[layer].length; row++) {
				write.write(String.valueOf(bias[layer][row]) + ",");
			}
			write.write("\n");
		}
		write.close();
		System.out.printf("File saved as: %s\n", fileName + ".csv");
	}
	
	/** Writes ASCII art depending on the inputs */
	public static int ascii(double[] values) {
		//Loops through the 28x28 pixel image & prints ascii art depending on the grey scale
		for(int i = 0; i < 28; i++) {
			for(int j = 0; j < 28; j++) {
				double val = values[i*28 + j];
				
				//The scale for greyscale to ascii
				if(val==0)
					System.out.print(' ');
				else if(val < 0.1)
					System.out.print('.');
				else if(val < 0.2)
					System.out.print(',');
				else if(val < 0.3)
					System.out.print(';');
				else if(val < 0.4)
					System.out.print('!');
				else if(val < 0.5)
					System.out.print('v');
				else if(val < 0.6)
					System.out.print('1');
				else if(val < 0.7)
					System.out.print('L');
				else if(val < 0.8)
					System.out.print('F');
				else if(val < 0.9)
					System.out.print('E');
				else if(val <= 1)
					System.out.print('$');
			}
			System.out.println();
		}
		//Allows user to escape the art show
		System.out.print("Enter 1 to continue (anything else to quit): ");
		
		//Validates user input to check if it's a 1
		String chose = choice.nextLine();
		int chosen;
		try {
			chosen = Integer.valueOf(chose);
			if(chosen == 1)
				return 1;
			
			else
				return 0;
		}catch(Exception InputMismatchException) {
			return 0;
		}
	}
	
	/** Function that loops through the inputs & shows the ASCII art for the values 
	 * @throws FileNotFoundException */
	public static void showASCII(boolean incorrect) throws FileNotFoundException {
		//Ensures that it is the testing data that is being shown
		dataReciever(testingFile);
		
		//Loops through all inputs & finds what the network thinks the answer is
		for(int caseNum = 0; caseNum < inputForNet.length; caseNum++) {
			double[] providedInput = inputForNet[caseNum];
			double[] providedOutput = outputForNet[caseNum];
			double[] networkOutput = network.testNetwork(providedInput);
			
			//Finds what the network thinks the answer is & what the answer is
			int netMax = 0;
			int outMax = 0;
			for(int j = 0; j < networkOutput.length; j++) {
				netMax = networkOutput[j] > networkOutput[netMax] ? j:netMax;
				outMax = providedOutput[j] > providedOutput[outMax] ? j:outMax;
			}
			
			clearScreen();
			int quit = 1;
			//Shows purely the incorrect outputs from the network
			if(incorrect && outMax != netMax) {
				System.out.printf("Testing Case #%d: Expected:%d Output:%d\tMatch:", caseNum, outMax, netMax);
				System.out.print("Incorrect\n");
				quit = ascii(providedInput);				
			}
			
			//Shows all outputs from the network
			else if(!incorrect) {
				System.out.printf("Testing Case #%d: Expected:%d Output:%d\tMatch:", caseNum, outMax, netMax);
				//Formats the string on whether the expected & output matched
				System.out.printf("%s",  netMax == outMax ? "Correct\n":"Incorrect\n");
				quit = ascii(providedInput);
			}
			
			if(quit == 0)
				break;
		}
	}
	
	/** Function that allows there to be stops for every menu option */
	public static void pauseForASec() {
		System.out.println();
		System.out.print("Enter any key when you are ready to return to the menu: ");
		choice.nextLine();
		
	}
	
	/** Using the inputs & outputs stored in the global variables, calculate the accuracy */
	public static double findAccuracy() {
		//Two array's to keep track of the number correct & the total for 0-9
		int[] correct = new int[10];
		int[] total = new int[10];
		
		//Loops through the inputs & finds what the network thinks it is
		for(int i = 0; i < inputForNet.length; i++) {
			double[] answer = network.testNetwork(inputForNet[i]);
			double[] out = outputForNet[i];
			int maxI = 0;
			for(int j = 0; j < answer.length; j++)
				maxI = answer[j] > answer[maxI] ? j:maxI;
				
			//If the network's output is right, add to both
			if(out[maxI] == 1) {
				correct[maxI]++;
				total[maxI]++;
			}
			
			//Else only add to the total for the correct value
			else {
				int expI = 0;
				for(int j = 0; j < out.length; j++)
					expI = out[j] == 1 ? j:expI;
				total[expI]++;
			}
		}
		
		int totalVal = 0;
		int totalGot = 0;
		double percent;
		
		//Prints out the accuracy tests for all of the output values
		for(int row = 0; row < 10; row++) {
			if(row%5 == 0 && row !=0)
				System.out.println();
			
			totalVal += total[row];
			totalGot += correct[row];
			percent = ((double)correct[row])/total[row]*100;
			System.out.printf("%d = %d/%d ", row, correct[row], total[row]);
		}
		//Calculates the total accuracy of the network
		percent = ((double)totalGot)/totalVal*100;
		System.out.printf("\nAccuracy: %d/%d = %.3f%%\n", totalGot, totalVal, percent);
		return percent;
	}
	
	/** Displays all of the options of pre-trained weights, and allows to choose one */
	public static void pickFile() throws FileNotFoundException {
		File dir;
		String newPath;
		
		//If the program is being ran from cmd, adjust the path accordingly
		//	to easily show the file to choose from
		if(System.console() != null) {
			String path = System.getProperty("user.dir");
			newPath = path.replace("\\src", "");
			dir = new File(newPath);
		}
		
		//Else if it is run from eclipse
		else {
			newPath = "./";
			dir = new File(".");
		}
		File[] files = dir.listFiles((d, name) -> name.endsWith(".csv"));;
		int chose;
		while(true) {
			System.out.println("Previously Saved Weights & Bias sets:");
			
			//Removes the system path from the names of the files & prints them to the console
			for(int file = 0; file < files.length; file++) {
				String fileName = files[file].getName();
				String noPath = fileName.replace(newPath, "");
				System.out.printf("[%d]: %s\n", file, noPath);
			}
			
			//Lets you chose which file you want to select
			System.out.print("Chosen Set: ");
			String chosen = choice.nextLine();
			try {
				chose = Integer.valueOf(chosen);
				if(chose >= 0 && chose < files.length)
					break;
				else {
					clearScreen();
					System.out.println("ERROR: Please pick an option that is listed.");
				}
			}catch(Exception InputMismatchException){
				clearScreen();
				System.out.println("ERROR: Please enter a number.");
			}
		}
		
		String fileName = files[chose].getName();
		String noPath = fileName.replace(newPath, "");
		clearScreen();
		//Loads the selected file
		System.out.printf("Loading File [%d]: %s\n", chose, noPath);
		Scanner weightFile = new Scanner(files[chose]);
		
		//Checks the first line of the file for the network sizes & sets up the network to match
		String line1 = weightFile.nextLine();
		String[] line1A = line1.split(",");
		
		//Sets the values, and creates a new neural network with those values just incase
		inputNum = Integer.valueOf(line1A[0]);
		hiddenNum = Integer.valueOf(line1A[1]);
		outputNum = Integer.valueOf(line1A[2]);
		batchNum = Integer.valueOf(line1A[3]);
		network = new NeuralNetwork(inputNum, hiddenNum, outputNum, batchNum);
		
		//Creates the necessary sized weight & bias vectors based on the loaded file
		double[][][] weight = new double[2][][];
		weight[0] = new double[hiddenNum][inputNum];
		weight[1] = new double[outputNum][hiddenNum];
		
		double[][] bias = new double[2][];
		bias[0] = new double[hiddenNum];
		bias[1] = new double[outputNum];
		
		//Sets the actual values of the weights & bias according to the file
		//Reads the weight's from all of the row's for the first layer
		for(int row = 0; row < hiddenNum; row++) {
			String line = weightFile.nextLine();
			String[] rowValues = line.split(",");
			//System.out.println(line);
			for(int col = 0; col < rowValues.length; col++)
				weight[0][row][col] = Double.valueOf(rowValues[col]);
		}
		
		//Reads the bias for the first layer
		String line = weightFile.nextLine();
		//System.out.println(line);
		String[] biasValues = line.split(",");
		for(int col = 0; col < biasValues.length; col++)
			bias[0][col] = Double.valueOf(biasValues[col]);
		
		//For the second layer
		//Read's the weight's of all the hiddenNodes
		for(int row = 0; row < outputNum; row++) {
			line = weightFile.nextLine();
			//System.out.println(line);
			String[] rowValues = line.split(",");
			for(int col = 0; col < rowValues.length; col++)
				weight[1][row][col] = Double.valueOf(rowValues[col]);
		}
		
		//All of the bias values
		line = weightFile.nextLine();
		//System.out.println(line);
		biasValues = line.split(",");
		for(int col = 0; col < biasValues.length; col++)
			bias[1][col] = Double.valueOf(biasValues[col]);
	
		//Closes the written file & ensures to set the weights & bias to the network
		weightFile.close();
		network.setWeightandBias(weight, bias);
		dataReciever(trainingFile);
		System.out.println("Accuracy with Training File:");
		findAccuracy();
		dataReciever(testingFile);
		System.out.println("\nAccuracy with Testing File:");
		findAccuracy();
	}
	
	/** Overloads the dataReciever function to return the input & output array 
	 * @throws FileNotFoundException */
	public static void dataReciever(File file) throws FileNotFoundException {
		double[][] input;
		double[][] ans;
		
		//Creates proper arrays based on if the file wanted to be loaded is the training
		//	or testing file
		//System.out.println(file.getName());
		if(file.getName().endsWith("mnist_train.csv")) {
			input = new double[60000][784];
			ans = new double[60000][];
		}
		else {
			input = new double[10000][784];
			ans = new double[10000][784];
		}
		
		Scanner scan = new Scanner(file);
		
		//Loops through the provided file and collects the data
		int inputPos = 0;
		while(scan.hasNextLine()) {
			//Grabs a line from the file, and splits it into an array of all the values
			String line = scan.nextLine();
			String[] numArray = line.split(",");
			
			//Grabs the hot vector for the answer
			ans[inputPos] = hotVector.get(numArray[0]);
			
			//Since the first value was the answer, this loops through the rest of the values
			//	ensuring to adjust the scale of the pixel to 0-1
			for(int lineI = 1; lineI < numArray.length; lineI++) {
				double num = Double.valueOf(numArray[lineI]) / 255;
				input[inputPos][lineI-1] = num;
			}
			inputPos++;
		}
		scan.close();
		//Sets the global variables
		inputForNet = input;
		outputForNet = ans;
		network.setInandOut(inputForNet, outputForNet);
	}
	
	/** Clears either the cmd screen or eclipse console, depending on which is in use */
	public static void clearScreen() {
		if(System.console() != null)
			System.out.print("\033[2J");
		else
			for(int i = 0; i < 50; i++)
				System.out.println();
	}
}
