package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.*;

import java.util.*;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;
	Instances nodeInstances;
}

public class DecisionTree implements Classifier {
	private Node rootNode;
	protected boolean isGini;
	private int class_index;
	protected int pValueIndex;
	protected int maxHeight;
	protected int currentHeight;
	protected double averageHeight;

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
	    rootNode = new Node();
		rootNode.nodeInstances = arg0;
		class_index = arg0.classIndex();
		buildTree(arg0);
	}

	private void buildTree(Instances instancesSet) throws Exception {

		java.util.Queue<Node> queue = new LinkedList<Node>();
		Node currentNode;
		rootNode.nodeInstances = instancesSet;
		queue.add(rootNode);

		while (!queue.isEmpty())
		{
			currentNode = (Node)queue.remove();
			// the HomeWork2.Node is perfectly classified
			if(perfectlyClassified(currentNode.nodeInstances))
			{
				//Leaf
				// sets return value as the value of one of the instances (since all the instances have the same value)
				currentNode.returnValue = currentNode.nodeInstances.instance(0).classValue();
				// moves to the next node in the queue
				continue;
			}
			// the best attribute for the set in current node
			currentNode.attributeIndex = bestAttribute(currentNode.nodeInstances);
			// the function returns -1 if splitting the node won't improve the tree (gain = 0)
			if(currentNode.attributeIndex == -1)
			{
				// Leaf
				// setting the return value to be the return value of the majority of the instances
				currentNode.returnValue = majorityValue(currentNode);
				// moves to next node in the queue
				continue;
			}
			else
			{
				// splits the node instances to the relevant attribute's values
				splitNode(currentNode, queue);
			}
		}
	}

	private boolean perfectlyClassified (Instances currentNodeInstances)
	{
		// the class of the first instance
		double classification = currentNodeInstances.instance(0).classValue();
		// for all the instances in the current node
		for(int i = 1; i < currentNodeInstances.numInstances(); i++)
		{
			// the instance class is different than the class of the first instance
			// the instances are not perfectly classified
			if(currentNodeInstances.instance(i).classValue() != classification)
				return false;
		}
		// all the instances have the same classification
		return true;
	}

	private int bestAttribute (Instances currentNodeInstances)
	{
		int bestAttributeIndex = 0;
		// calculates the gain with attribute number 0
		double maxGain = calcGain(currentNodeInstances,0);
		double currentGain;

		// for all the attributes of the instances
		for(int i = 1; i < currentNodeInstances.numAttributes() - 1; i++)
		{
			// calculates the gain by attribute index i
			currentGain = calcGain(currentNodeInstances,i);
			// the calculated gain is larger than the max gain calculated so far
			if(currentGain > maxGain)
			{
				maxGain = currentGain;
				bestAttributeIndex = i;
			}
		}

		// splitting the node won't improve the tree (gain = 0 for every attribute)
		if(maxGain == 0)
		{
			return (-1);
		}

		return bestAttributeIndex;
	}

	private double calcGain(Instances instancesInCurrentNode, int attributeIndex)
	{
		// the number of possible values for this attribute
		int numOfAttributeValue = instancesInCurrentNode.attribute(attributeIndex).numValues();
		double sum;
		double fatherProb;

		double[] attributesDistribution = new double[numOfAttributeValue];
		double[] kidsRecurrenceProb = new double[numOfAttributeValue];

		// fills attributesDistribution array with the probabilities of attribute's values in every attribute
		// fills kidsRecurrenceProb array with the probabilities of 'recurrence-events' for every attribute value
		// returns the probability of 'recurrence-events' for the instances in the father node
		double recurrenceFather = calcProb(instancesInCurrentNode, attributeIndex, attributesDistribution, kidsRecurrenceProb);

		// using Gini as the impurity measure
		if (isGini){

			fatherProb = calcGini(recurrenceFather);
			sum = 0;

			for (int i = 0; i < numOfAttributeValue; i++){

				// |Sv|/|S| * GiniIndex(Sv)
				sum += attributesDistribution[i] * calcGini(kidsRecurrenceProb[i]);

			}

			return  fatherProb - sum;
		}
		// using Entropy as the impurity measure
		else {

			fatherProb = calcEntropy(recurrenceFather);
			sum = 0;

			for (int i = 0; i < numOfAttributeValue; i++){

				// calculates only if the value will contribute to the sum
				if (kidsRecurrenceProb[i] != 0 && kidsRecurrenceProb[i] != 1) {
					// |Sv|/|S| * Entropy(Sv)
					sum += attributesDistribution[i] * calcEntropy(kidsRecurrenceProb[i]);
				}
			}

			return  fatherProb - sum;
		}
	}

	private double calcGini(double probabilitySick)
	{
		// (|S1|/|S|)^2 + (|S2|/|S|)^2
		double sum = Math.pow(probabilitySick, 2) + Math.pow(1 - probabilitySick, 2);

		return 1 - sum;
	}

	private double calcEntropy(double probabilitySick)
	{
		double sum = 0;

		// |Si|/|S| log |Si|/|S|

		sum = probabilitySick * Math.log(probabilitySick) + (1 - probabilitySick) * Math.log(1 - probabilitySick);

		return sum * (-1);
	}

	// returns the father probability for 'recurrence-events'
	// adds to attributesDistribution array the probabilities of 'recurrence-events' for every attribute value
	// adds to kidsRecurrenceProb array the probabilities of 'recurrence-events' for every attribute value
	private double calcProb(Instances instancesInCurrentNode, int attributeIndex,
							double[] attributesDistribution, double[] kidsRecurrenceProb){

		// the probability of 'recurrence-events' in the father node
		double recurrenceFather = 0;

		// for every instance in the current node
		for (int i = 0; i < instancesInCurrentNode.numInstances(); i++ ){

			Instance currentInstance = instancesInCurrentNode.instance(i);
			// gets the current instance attribute value
			String currentVal = currentInstance.stringValue(attributeIndex);
			// gets the index of the instance attribute value
			int AttributeValueIndex = instancesInCurrentNode.attribute(attributeIndex).indexOfValue(currentVal);
			// increments the number of instances with currentVal attribute value
			attributesDistribution[AttributeValueIndex]++;

			// if the classification of this instance is 'recurrence-events'
			if (currentInstance.classValue() == 0){
				// increments the number of instances with classification 'recurrence-events' (in the father node)
				recurrenceFather++;
				// increments the number of instances with classification 'recurrence-events' (in the children nodes)
				kidsRecurrenceProb[AttributeValueIndex]++;
			}
		}

		//calculates the probability of 'recurrence-events' in the father node
		recurrenceFather = recurrenceFather / instancesInCurrentNode.numInstances();

		//calculates the probability of 'recurrence-events' in every child node
		//calculates the probability of being in class 'classIndex' in the father node
		for (int i = 0; i < attributesDistribution.length; i++){

			if (attributesDistribution[i] != 0) {
				// num of instances that are classified as 'recurrence-events' in attribute value i / num of
				// instances in attribute value i
				kidsRecurrenceProb[i] = kidsRecurrenceProb[i] / attributesDistribution[i];
				// num of instances with attribute value i / num of instances in the father node
				attributesDistribution[i] = attributesDistribution[i] / instancesInCurrentNode.numInstances();
			}
		}

		return recurrenceFather;
	}

	// returns the classification of the majority of instances in the current node
	private double majorityValue(Node currentNode)
	{
		int numOfInstances = currentNode.nodeInstances.numInstances();
		int counterOfRecurrence = 0;
		Instance currentInstance;

		// counts the number of instance with 'recurrent-events' classification
		for (int i = 0; i < numOfInstances; i++)
		{
			currentInstance = currentNode.nodeInstances.instance(i);
			if (currentInstance.classValue() == 0) {
				counterOfRecurrence++;
			}
		}
		// checks the classification of the majority
		if(counterOfRecurrence > numOfInstances - counterOfRecurrence)
		{
			// the classification of the majority is 'recurrence-events'
			return 0;
		}
		// the classification of the majority is 'no-recurrence-events'
		return 1;
	}

	private void splitNode(Node parent, java.util.Queue queue) throws Exception {
		double[][] tableOfChiSquaredProbabilities =
				{
						{0, 0.102, 0.455, 1.323, 3.841, 7.879},
						{0, 0.575, 1.386, 2.773, 5.991, 10.597},
						{0, 1.213, 2.366, 4.108, 7.815, 12.838},
						{0, 1.923, 3.357, 5.385, 9.488, 14.860},
						{0, 2.675, 4.351, 6.626, 11.070, 16.750},
						{0, 3.455, 5.348, 7.841, 12.592, 18.548},
						{0, 4.255, 6.346, 9.037, 14.067, 20.278},
						{0, 5.071, 7.344, 10.219, 15.507, 21.955},
						{0, 5.899, 8.343, 11.389, 16.919, 23.589},
						{0, 6.737, 9.342, 12.549, 18.307, 25.188},
						{0, 7.584, 10.341, 13.701, 19.675, 26.757},
						{0, 8.438, 11.340, 14.845, 21.026, 28.300},
						{0, 9.299, 12.340, 15.984, 22.362, 29.819}
				};

		Instances nodeInstances = parent.nodeInstances;
		Attribute nodeAttribute = nodeInstances.attribute(parent.attributeIndex);
		//initializes the children array by the number of possible values for the parent attribute index
		parent.children = new Node[nodeAttribute.numValues()];
		int recurrenceCounterParent = 0;
		int degreeOfFreedom = 0;
		double chiSquare;
		Instances[] childrenInstances = new Instances[nodeAttribute.numValues()];
		Instance currentInstance;
		String currentInstanceValue;
		int currentInstanceAttributeValueIndex;

		// constructs an array of instances by the attribute's values
		for (int i = 0; i < childrenInstances.length; i++){
			childrenInstances[i] = new Instances(nodeInstances, 0, 0);
		}
		// for every instance in the node
		for (int i = 0; i < nodeInstances.numInstances(); i++){
			currentInstance = nodeInstances.instance(i);
			// gets the value of the current instance in attribute attributeIndex
			currentInstanceValue = currentInstance.stringValue(parent.attributeIndex);
			// gets the index of this value
			currentInstanceAttributeValueIndex = currentInstance.attribute(parent.attributeIndex).indexOfValue(currentInstanceValue);
			// adds current instance to the suitable instances set according to the attribute's value
			childrenInstances[currentInstanceAttributeValueIndex].add(currentInstance);
			// increments the number of instances with 'recurrence-events' classification in parent if necessary
			if(currentInstance.classValue() == 0)
			{
				recurrenceCounterParent++;
			}
		}

		// sets the node's return value by the value of the majority
		if (recurrenceCounterParent >= nodeInstances.numInstances() - recurrenceCounterParent){
			// the classification of the majority is 'recurrence-events'
			parent.returnValue = 0;
		}
		// the classification of the majority is 'no-recurrence-events'
		else {
			parent.returnValue = 1;
		}
		// counts the number of relevant attribute values (number of values that exists in the training data)
		for (int i = 0; i < childrenInstances.length; i++){
			if (!childrenInstances[i].isEmpty()){
				degreeOfFreedom++;
			}
		}

		// pruning according to p-value
		// pValueIndex 0 -> p-value = 1 (no pruning)
		if (pValueIndex != 0){
			chiSquare = calcChiSquare(nodeInstances, parent.attributeIndex);
			// calculates the degree of freedom according to the relevant number of values
			// -1 for calculating degree of freedom, -1 for array indexes
			degreeOfFreedom -=  2;
			// checks the need to prune
			if (chiSquare < (tableOfChiSquaredProbabilities[degreeOfFreedom][pValueIndex])){
				// pruning - parent node is a leaf
				parent.children = null;
				return;
			}
		}
		// for every children of current node construct a node with the relevant instances set
		for (int i = 0; i < childrenInstances.length; i++){
			if(childrenInstances[i].isEmpty())
			{
				continue;
			}

			Node child = new Node();
			child.nodeInstances = childrenInstances[i];
			child.parent = parent;
			parent.children[i] = child;
			// adds the descendant nodes to the queue
			queue.add(child);
		}
	}

	protected double calcAvgError(Instances instancesSet){
		Instance currentInstance;
		double predictedClass;
		double numOfErrors = 0;
		double realClass;
		double sumOfHeights = 0;
		maxHeight = -1;


		// for every instance
		for (int i = 0; i < instancesSet.numInstances(); i++){
			currentInstance = instancesSet.instance(i);
			// finds the predicted classification for the current instance
			predictedClass = classifyInstance(currentInstance);

			// checks if the current height is bigger than maxHeight
			if (currentHeight > maxHeight){
				maxHeight = currentHeight;
			}

			// adds current height to the sum of all heights
			 sumOfHeights += currentHeight;

			// gets the real classification of this instance
			realClass = currentInstance.classValue();
			// checks if the classification is correct
			if (predictedClass != realClass){
				// if not, increments the number of errors
				numOfErrors++;
			}
		}

		averageHeight = sumOfHeights / instancesSet.numInstances();

		// returns the average error
		return numOfErrors / instancesSet.numInstances();

	}

	@Override
	public double classifyInstance(Instance instance) {

		Node currentNode = rootNode;
		int currentAttributeIndex;
		String instanceAttributeValue;
		int instanceValueIndex;
		// root height is 0
		currentHeight = 0;

		while (currentNode.children != null){
			// gets the relevant attribute
			currentAttributeIndex = currentNode.attributeIndex;
			// gets the value of current instance in attribute attributeIndex
			instanceAttributeValue = instance.stringValue(currentAttributeIndex);
			// gets the index of the instance attribute value
			instanceValueIndex = instance.attribute(currentAttributeIndex).indexOfValue(instanceAttributeValue);

			// checks if there were instances with current attribute value in the trainingData
			if(currentNode.children[instanceValueIndex] == null)
			{
				// there were no instances with current attribute value - returns current node classification
				return currentNode.returnValue;
			}
			// increments the height of the tree according to this instance path in the tree
			currentHeight++;
			currentNode = currentNode.children[instanceValueIndex];
		}
		return currentNode.returnValue;
	}

	private double calcChiSquare(Instances instancesSet, int attributeIndex){

		int numOfAttributeValue =instancesSet.attribute(attributeIndex).numValues();
		double chiSquare = 0;

		double[] attributesDistribution = new double[numOfAttributeValue];
		double[] kidsRecurrenceProb = new double[numOfAttributeValue];

		// fills attributesDistribution array with the probabilities of attribute's values in every attribute
		// fills kidsRecurrenceProb array with the probabilities of 'recurrence-events' for every attribute value
		// returns the probability of 'recurrence-events' for the instances in the father node
		double probRecurrence = calcProb(instancesSet, attributeIndex, attributesDistribution, kidsRecurrenceProb);
		double Df;
		double pf;
		double nf;
		double E0;
		double E1;

		for (int i = 0; i < numOfAttributeValue; i++){
			// if there are instances with this attribute value
			if(attributesDistribution[i] != 0) {
				// the number of instances with this attribute value
				Df = attributesDistribution[i] * instancesSet.numInstances();
				// the number of instances with this attribute value and with 'recurrence-events 'classification
				pf = kidsRecurrenceProb[i] * Df;
				// the number of instances with this attribute value  and with 'no-recurrence-events 'classification
				nf = (1 - kidsRecurrenceProb[i]) * Df;
				E0 = Df * probRecurrence;
				E1 = Df * (1 - probRecurrence);

				chiSquare += (Math.pow((pf - E0), 2) / E0) + (Math.pow((nf - E1), 2) / E1);
			}
		}
		return chiSquare;
	}

	public void printTree() {
		System.out.println("Root");
		printTree(this.rootNode, 0);
	}

	private void printTree(Node currentNode, int numOfTabs)
	{
		if (currentNode.children == null)
		{
			// current node is a leaf
			System.out.println(getTabs(numOfTabs + 1) + "Leaf. Returning value: " + currentNode.returnValue);
		}
		else {
			// current node has children
			System.out.println(getTabs(numOfTabs) + "Returning value: " + currentNode.returnValue);
			// executes printTree method for every child
			for (int i = 0; i < currentNode.children.length; i++) {
				if (currentNode.children[i] != null) {
					System.out.println(getTabs(numOfTabs + 1) + "If attribute " + currentNode.attributeIndex + " = " + i);
					printTree(currentNode.children[i], numOfTabs +1);
				}
			}
		}
	}

	// generates a string of numOfTabs tabs
	private String getTabs (int numOfTabs)
	{
		StringBuilder tabs = new StringBuilder("");
		for (int i = 0; i < numOfTabs ; i++)
		{
			tabs.append("\t");
		}

		return tabs.toString();
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}



	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
