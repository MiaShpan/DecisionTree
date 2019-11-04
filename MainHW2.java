package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW2 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 *
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		double giniError;
		double entropyError;
		double pruneError;
		boolean chosenImpurityMethod; // true = Gini, false = Entropy
		double[] pValue = {1, 0.75, 0.5, 0.25, 0.05, 0.005};
		int bestPValueIndex = 0;
		double minError = 1;

		Instances trainingCancer = loadData("src/cancer_train.txt");
		Instances testingCancer = loadData("src/cancer_test.txt");
		Instances validationCancer = loadData("src/cancer_validation.txt");

		DecisionTree entropyTree = new DecisionTree();
		DecisionTree giniTree = new DecisionTree();

		// setting Decision tree fields
		entropyTree.isGini = false;
		entropyTree.pValueIndex = 0; // pValue = 1 (no pruning)

		giniTree.isGini = true;
		giniTree.pValueIndex = 0; // pValue = 1 (no pruning)

		// building the decision trees on the training data
		entropyTree.buildClassifier(trainingCancer);
		giniTree.buildClassifier(trainingCancer);

		// calculating the validation errors
		giniError = giniTree.calcAvgError(validationCancer);
		entropyError = entropyTree.calcAvgError(validationCancer);

		System.out.println("Validation error using Entropy: " + entropyError);
		System.out.println("Validation error using Gini: " + giniError);
		System.out.println("---------------------------------------------------------");

		// choosing the best impurity method (Gini = true, Entropy = false)
		chosenImpurityMethod = entropyError > giniError;

		// building decision trees on the training data with pruning according to pValue
		for (int i = 0; i < pValue.length; i++) {
			DecisionTree prunedTree = new DecisionTree();
			// setting prunedTree fields data: p-value and impurity method
			prunedTree.pValueIndex = i;
			prunedTree.isGini = chosenImpurityMethod;
			prunedTree.buildClassifier(trainingCancer);
			System.out.println("Decision Tree with p_value of: " + pValue[i]);
			System.out.println("The train error of the decision tree is: " + prunedTree.calcAvgError(trainingCancer));
			// calculating the validation error
			pruneError = prunedTree.calcAvgError(validationCancer);
			System.out.println("Max height on validation data: " + prunedTree.maxHeight);
			System.out.println("Average height on validation data: " + prunedTree.averageHeight);
			System.out.println("The validation error of the decision tree is: " + pruneError);
			System.out.println("---------------------------------------------------------");
			// current tree error is smaller than minimum error so far
			if(pruneError < minError)
			{
				minError = pruneError;
				bestPValueIndex = i;
			}
		}
		// constructing a new tree with the best p-value and chosen impurity method
		System.out.println("Best validation error at p_value = " + pValue[bestPValueIndex]);
		DecisionTree bestTree = new DecisionTree();
		// setting the bestTree p-value and impurity method
		bestTree.isGini = chosenImpurityMethod;
		bestTree.pValueIndex = bestPValueIndex;
		bestTree.buildClassifier(trainingCancer);
		System.out.println("Test error with best tree: " + bestTree.calcAvgError(testingCancer));
		System.out.println();
		// printing the bestTree
		bestTree.printTree();
	}
}

