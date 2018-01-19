package nn;

import java.util.Random;
import Jama.Matrix;

/**
 * 
 * @author naharrison
 */
public class NeuralNetwork {
	
	public int nInputNodes, nHiddenNodes, nOutputNodes;
	public double learningRate;
	public Matrix Wih, Who;

	
	public NeuralNetwork(int nInputNodes, int nHiddenNodes, int nOutputNodes, double learningRate) {
		this.nInputNodes = nInputNodes;
		this.nHiddenNodes = nHiddenNodes;
		this.nOutputNodes = nOutputNodes;
		this.learningRate = learningRate;
		initWih();
		initWho();
	}


	private void initWih() {
		double[][] WihArr = new double[nHiddenNodes][nInputNodes];
		Random r = new Random();
		for(int j = 0; j < nHiddenNodes; j++) {
			for(int k = 0; k < nInputNodes; k++) {
				WihArr[j][k] = (1.0/Math.sqrt((double) nHiddenNodes))*r.nextGaussian();
			}
		}
		Wih = new Matrix(WihArr);
	}

	
	private void initWho() {
		double[][] WhoArr = new double[nOutputNodes][nHiddenNodes];
		Random r = new Random();
		for(int j = 0; j < nOutputNodes; j++) {
			for(int k = 0; k < nHiddenNodes; k++) {
				WhoArr[j][k] = (1.0/Math.sqrt((double) nOutputNodes))*r.nextGaussian();
			}
		}
		Who = new Matrix(WhoArr);
	}

	
	public static Matrix activationFcn(Matrix m) {
		int nRows = m.getRowDimension();
		int nColumns = m.getColumnDimension();
		double[][] resultArr = new double[nRows][nColumns];
		for(int j = 0; j < nRows; j++) {
			for(int k = 0; k < nColumns; k++) {
				double x = m.get(j, k);
				resultArr[j][k] = 1.0/(1.0 + Math.exp(-1.0*x));
			}
		}
		return new Matrix(resultArr);
	}
	
	
	// {{a}, {b}, {c}} * {{A}, {B}, {C}} = {{aA}, {bB}, {cC}}
	public static Matrix acrossProduct(Matrix a, Matrix b) {
		if(a.getRowDimension() != b.getRowDimension() || a.getColumnDimension() != 1 || b.getColumnDimension() != 1) {
			System.out.println("acrossProduct ERROR");
		}
		double[][] resultArr = new double[a.getRowDimension()][1];
		for(int k = 0; k < a.getRowDimension(); k++) resultArr[k][0] = a.get(k, 0) * b.get(k, 0);
		return new Matrix(resultArr);
	}

	
	public void train(Matrix inputData, Matrix targetData) {
		Matrix hiddenInputs = Wih.times(inputData);
		Matrix hiddenOutputs = activationFcn(hiddenInputs);
		
		Matrix finalInputs = Who.times(hiddenOutputs);
		Matrix finalOutputs = activationFcn(finalInputs);
		
		Matrix outputErrors = targetData.minus(finalOutputs);
		Matrix hiddenErrors = Who.transpose().times(outputErrors);
		
		double[][] ones_lengthHiddenOutputsArr = new double[hiddenOutputs.getRowDimension()][1];
		for(int k = 0; k < hiddenOutputs.getRowDimension(); k++) ones_lengthHiddenOutputsArr[k][0] = 1.0;
		Matrix ones_lengthHiddenOutputs = new Matrix(ones_lengthHiddenOutputsArr);

		double[][] ones_lengthFinalOutputsArr = new double[finalOutputs.getRowDimension()][1];
		for(int k = 0; k < finalOutputs.getRowDimension(); k++) ones_lengthFinalOutputsArr[k][0] = 1.0;
		Matrix ones_lengthFinalOutputs = new Matrix(ones_lengthFinalOutputsArr);

		Wih.plusEquals(acrossProduct(hiddenErrors, acrossProduct(hiddenOutputs, ones_lengthHiddenOutputs.minus(hiddenOutputs))).times(inputData.transpose()).times(learningRate));
		Who.plusEquals(acrossProduct(outputErrors, acrossProduct(finalOutputs, ones_lengthFinalOutputs.minus(finalOutputs))).times(hiddenOutputs.transpose()).times(learningRate));
	}
	
	
	public Matrix query(Matrix queryData) {
		Matrix hiddenInputs = Wih.times(queryData);
		Matrix hiddenOutputs = activationFcn(hiddenInputs);
		Matrix finalInputs = Who.times(hiddenOutputs);
		return activationFcn(finalInputs);
	}

	
	public int getnInputNodes() {
		return nInputNodes;
	}


	public void setnInputNodes(int nInputNodes) {
		this.nInputNodes = nInputNodes;
	}


	public int getnHiddenNodes() {
		return nHiddenNodes;
	}


	public void setnHiddenNodes(int nHiddenNodes) {
		this.nHiddenNodes = nHiddenNodes;
	}


	public int getnOutputNodes() {
		return nOutputNodes;
	}


	public void setnOutputNodes(int nOutputNodes) {
		this.nOutputNodes = nOutputNodes;
	}


	public double getLearningRate() {
		return learningRate;
	}


	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}


	public Matrix getWih() {
		return Wih;
	}


	public void setWih(Matrix wih) {
		Wih = wih;
	}


	public Matrix getWho() {
		return Who;
	}


	public void setWho(Matrix who) {
		Who = who;
	}


	public static void main(String[] args) {
		NeuralNetwork nn = new NeuralNetwork(5, 7, 3, 0.3);
		
		double[][] inArr = {{0.7}, {0.8}, {0.6}, {0.2}, {0.3}};
		double[][] targArr = {{0.01}, {0.99}, {0.01}};
		Matrix input = new Matrix(inArr);
		Matrix target = new Matrix(targArr);
		
		nn.Who.print(8, 3); // width of column, number of decimal places
		nn.train(input, target);
		nn.Who.print(8, 3);
		
		double[][] queryArr = {{0.1}, {0.1}, {0.4}, {0.3}, {0.2}};
		Matrix queryData = new Matrix(queryArr);
		
		nn.query(queryData).print(8, 3);
	}

}
