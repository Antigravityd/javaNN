import java.io.FileInputStream;
import java.io.IOException;
import java.io.File;
import java.util.Scanner;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.lang.Math;
import java.util.stream.*;


public class iq {
	public static void main(String[] args) throws IOException, Exception {
		List<List<Double>> xorData = Arrays.asList(Arrays.asList(0.0, 1.0, 1.0), Arrays.asList(1.0, 0.0, 1.0), Arrays.asList(0.0, 0.0, 0.0), Arrays.asList(1.0, 1.0, 0.0));
		System.out.println(xorData);
		Feedforward myNet = new Feedforward("C:\\Users\\antig\\OneDrive\\Desktop\\Code\\Java\\JavaNN\\cfg.txt");
		myNet.debug1();
		myNet.debug2();
		System.out.println(myNet.propagate(Arrays.asList(1.0,1.0)));
		myNet.debug1();
		myNet.debug2();

		Trainer myGym = new Trainer(myNet);
		double a;
		int count = 0;
		do {
			a = myGym.backpropagate(xorData);
			System.out.println(a);
			++count;
		} while (a >= .01 && count < 100000);

		myNet.debug2();
		for (List<Double> i : xorData) {
			System.out.println(i.subList(0,2));
			System.out.println(myNet.propagate(i.subList(0,2)).get(0));
		}






	}
}




class Trainer {
	Feedforward trainee;
	Neuron outputNeuron;
	double lastChange = 0.0;
	public Trainer(Feedforward trainee) {
			this.trainee = trainee;
	}
	private double discrepancy(double expectedVal, double realOut) {
		return Math.pow(expectedVal - realOut, 2);
	}
	private double sigma(Neuron currNeuron, double expectedOut) {
		if (currNeuron.equals(trainee.getOutputNeuron())) {
			return (currNeuron.activationOf() - expectedOut) *
			trainee.activationDerivative(currNeuron.activationOf());
		}
		else {
			double summation = 0;
			for (EdgeGroup i : trainee.getEdgeGroupList()) {
				for (Edge j : i.getEdgeList()) {
					if (j.getStartNeuron().equals(currNeuron)) {
						summation += j.getWeight() * sigma(j.getEndNeuron(), expectedOut);
					}
				}
			}
			return summation * trainee.activationDerivative(currNeuron.activationOf());
		}

	}

	public double backpropagate(List<List<Double>> trainingData) {
		List<Double> errorArray = new ArrayList<>();
		for (List<Double> i : trainingData) {
			errorArray.add(discrepancy(i.get(i.size() - 1), trainee.propagate(
			i.subList(0, i.size() - 1)).get(0)));
			for (EdgeGroup j : trainee.getEdgeGroupList()) {
				for (Edge k : j.getEdgeList()) {
					double momentum = trainee.getMomentumParameter() * lastChange;
					lastChange =  -1.0 * trainee.getLearningRate() * sigma(k.getEndNeuron(),
					i.get(i.size() - 1)) *
					k.getStartNeuron().activationOf();
					k.setWeight(k.getWeight() + momentum + lastChange);
				}
			}
		}
		double errorSum = 0.0;
		for(int i = 0; i < errorArray.size(); ++i) {
			errorSum += errorArray.get(i);
		}
		return errorSum / errorArray.size();
	}
}




abstract class Network {
	String type;
	String activationFunctionID;
	double bias;
	double learningRate;
	double momentumParameter;
	String configPath;
	String restOfFile = "";

	protected Network(String cfgPath) throws IOException, Exception {
			File cfg = new File(cfgPath);
			FileInputStream fileByteStream = new FileInputStream(cfg);
			Scanner fileScan = new Scanner(fileByteStream);

			type = fileScan.nextLine();
			activationFunctionID = fileScan.nextLine();
			bias = Double.parseDouble(fileScan.nextLine());
			momentumParameter = Double.parseDouble(fileScan.nextLine());
			learningRate = Double.parseDouble(fileScan.nextLine());




		while (fileScan.hasNextLine()) {
			String line = fileScan.nextLine();
			restOfFile += line;
			restOfFile += '\n';
		}

		fileByteStream.close();
	}


	public double activationFunction(double input) {
		if (activationFunctionID.equals("linear")) {
			return input;
		}
		else if (activationFunctionID.equals("logistic")) {
			return 1.0 / (1.0 + Math.pow(Math.E, -1.0 * input + bias));
		}
		else {
			System.out.println("activation function not read");
			return 0.0;
		}


	}

	public double activationDerivative(double input) {
		if (activationFunctionID.equals("identity")) {
			return 1.0;
		}
		else if (activationFunctionID.equals("logistic")) {
			return activationFunction(input) * (1.0 - activationFunction(input));
		}
		else {
			System.out.println("activation function not read");
			return 0.0;
		}
	}




	/*FIXME: figure out what needs to be in here, probably code to read from
		a file and initialize fields like the activation function, etc.


		Must have a create() method that extablishes a network that has fields of type
		Neuron and Edge.


		Probably needs to specify input and output neurons too.
	*/







	public String getType() {
		return type;
	}

	public String getActivationFunction() {
		return activationFunctionID;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public double getMomentumParameter() {
		return momentumParameter;
	}

	public double getBias() {
		return bias;
	}

	public String getConfig() {
		return configPath;
	}




	/*FIXME: all methods are protected. Some need to be public, oc.
		Gotta write them all first tho. Probably constructor methods for
		network-type level classes (Feedforward,BiasFeedforward, etc.) are public.
		FIXME: add activation functions  MIGHT HAVE DONE only identity implemented
		FIXME: longterm eventually add different Neuron types(memory, recurrent, probablistic, I/O)
		FIXME: longerm add GUI for creating networks
	*/

}




class Feedforward extends Network {

	public List<Layer> layerList = new ArrayList<>();
	public List<EdgeGroup> edgeGroupList = new ArrayList<>();
	public Neuron outputNeuron;


	public Feedforward(String cfg) throws Exception {
		super(cfg);
		String tempstr = "";
		int currNeuronNum = 0;
		int currEdgeNum = 0;
		int currLayerID = 0;
		int count = 0;

		if (!(type).equals("feedforward")) {
			throw new Exception("cfgError: config has type " + type + ", not feedforward");
		}


		for (int i = 0; i < restOfFile.length(); ++i) {
			if (restOfFile.charAt(i) != '\n') {
				count++;
				tempstr += restOfFile.charAt(i);


			}


			else {
				int newLayerSize = Integer.parseInt(tempstr);
				try {
					layerList.add(new Layer(newLayerSize, currNeuronNum, currLayerID));

					currNeuronNum += newLayerSize;
					++currLayerID;
				}


				catch (Exception e) {
					throw new Exception("cfgError: Please enter valid positive integer values for layer size/quantity.");
				}

				tempstr = "";
			}
		}




		int counter = 0;
		for (int i = 1; i < layerList.size(); ++i) {
			for (Neuron j : layerList.get(i).getNeuronList()) {
				edgeGroupList.add(new EdgeGroup(layerList.get(i-1), j, counter));
				++counter;
			}
		}


		currEdgeNum = 0;
		outputNeuron = layerList.get(layerList.size() - 1).getNeuronAt(0);
	}

	protected void debug1() {
		System.out.println("Layer count: " + layerList.size());
		int neuronCount = 0;
		for (Layer i : layerList) {
			for (Neuron j : i.neuronList) {
				++neuronCount;
			}
		}
		System.out.println("Neuron count: " + neuronCount);
		int edgeCount = 0;
		for (EdgeGroup i : edgeGroupList) {
			for (Edge j : i.edgeList) {
				++edgeCount;
			}
		}
		System.out.println("Edge count: " + edgeCount);
	}

	protected void debug2() {
		System.out.println("***START NEURON VALS***\n");
		for (Layer i : layerList) {
			for (Neuron j : i.neuronList) {
				System.out.println(j.activationOf());
			}
		}
		System.out.println("***END NEURON VALS***\n");


		System.out.println("***START EDGE VALS***\n");
		for (EdgeGroup i : edgeGroupList) {
			for (Edge j : i.edgeList) {
				System.out.println(j.getWeight());
			}
		}
		System.out.println("***END EDGE VALS***\n");

	}


	/*private int[] propagateTrain(two arrays, one of input-layer
	activations and one of expected outputs (for one evaluation of the network)) {
		code

		something that returns error from expected value(s)
	}*/

	public List<Double> propagate(List<Double> inputs) {
		int count = 0;
		for (int i = 0; i < layerList.size(); ++i) {
			switch(i) {
				case 0:
					for (int j = 0; j < layerList.get(0).getLayerSize(); ++j) {
						layerList.get(0).getNeuronAt(j).setActivation(inputs.get(j));
					}
					break;

				default:
					for (int j = 0; j < layerList.get(i).getLayerSize(); ++j) {

						layerList.get(i).getNeuronAt(j).setActivation(activationFunction(
						 edgeGroupList.get(count).activate()));

						++count;
					}
					break;


					}


				}
				List<Double> returnList = new ArrayList<>();
				for (int i = 0; i < layerList.get(layerList.size() - 1).getLayerSize(); ++i) {
					returnList.add(layerList.get(layerList.size() - 1).getNeuronList().get(i).activationOf());
				}

				return returnList;
		}

		public List<EdgeGroup> getEdgeGroupList() {
			return edgeGroupList;
		}

		public List<Layer> getLayerList() {
			return layerList;
		}
		public Neuron getOutputNeuron() {
			return outputNeuron;
		}
}




class Layer {

	int layerSize;
	public int ID;
	public List<Neuron> neuronList = new ArrayList<>();


	protected Layer(int size, int firstNeuronNum, int IDin) {
		for (int i = 0; i < size; i++) {
			neuronList.add(new Neuron(firstNeuronNum + i));
			layerSize = size;
			ID = IDin;
		}

	}

	protected Neuron getNeuronAt(int i) {
		return neuronList.get(i);
	}

	public List<Neuron> getNeuronList() {
		return neuronList;
	}

	protected int getLayerSize() {
		return layerSize;
	}

}




class EdgeGroup {
	int groupID;
	public List<Edge> edgeList = new ArrayList<>();
	double out;
	Neuron endNeuron;


	protected EdgeGroup(Layer fromLayer, Neuron targetNeuron, int ID) {
		for (int i = 0; i < fromLayer.layerSize; ++i) {
			edgeList.add(new Edge(fromLayer.getNeuronAt(i), targetNeuron, i));
		}
		groupID = ID;
		endNeuron = targetNeuron;
	}

	protected Edge getEdgeAt(int i) {
		return edgeList.get(i);
	}

	protected double activate() {
		double inputLayerOut = 0;
		for (int i = 0; i < edgeList.size(); ++i) {
			inputLayerOut += edgeList.get(i).getWeight() *
			edgeList.get(i).getStartNeuron().activationOf();
		}





		return inputLayerOut;
	}

	public List<Edge> getEdgeList() {
		return edgeList;
	}
}




class Edge {

	int edgeID;
	Neuron startNeuron;
	Neuron endNeuron;
	double weightVal;
	String activationFunct;


	protected Edge(Neuron neur1, Neuron neur2, int id) {
		startNeuron = neur1;
		endNeuron = neur2;
		weightVal = Math.random();
		edgeID = id;
	}


	protected Edge(Neuron neur1, Neuron neur2, int id , double weightToSet) {
		startNeuron = neur1;
		endNeuron = neur2;
		weightVal = weightToSet;
		edgeID = id;
	}



	public void setWeight(double inWeight) {
		weightVal = inWeight;
	}

	protected double getWeight() {
		return weightVal;
	}

	protected Neuron getStartNeuron() {
		return startNeuron;
	}

	protected Neuron getEndNeuron() {
		return endNeuron;
	}



}



class Neuron {
	double bias;
	double activation;
	int nodeNum;
	String activationFunction;


	protected Neuron(int num) {
		activation = Math.random();
		nodeNum = num;
		bias = 0;
	}

	protected Neuron(int num, double act) {
		activation = act;
		nodeNum = num;
		bias = 0;
	}

	protected Neuron(int num, double act, double biasIn) {
		activation = act;
		nodeNum = num;
		bias = biasIn;
	}


	public void setActivation(double toSet) {
		activation = toSet;
	}

	protected void setNumOf(int ntoSet) {
		nodeNum = ntoSet;
	}


	protected double activationOf() {
		return activation;
	}

	protected int numberOf() {
		return nodeNum;
	}

	protected double biasOf() {
		return bias;
	}




}
