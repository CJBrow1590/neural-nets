package nn;

import java.util.ArrayList;
import org.jlab.analysis.plotting.TCanvasPTabbed;
import org.jlab.groot.data.H1F;
import Jama.Matrix;

/**
 * 
 * @author naharrison
 */
public class NetworkDiagnostic {
	
	public NeuralNetwork nn;

	/* if the network's best guess output is less than this, don't count it */
	public double confidenceCut;

	/*  a histo for each possible output
	 *  shows how often the network gets it right or wrong */
	public ArrayList<H1F> summaryHistos;
	
	/* an N x N collection of histos of the confidence score
	 * one histo for each gen/rec combination */
	public ArrayList<ArrayList<H1F>> confidences;

	
	public NetworkDiagnostic(int nInputNodes, int nHiddenNodes, int nOutputNodes, double learningRate, double confidenceCut) {
		nn = new NeuralNetwork(nInputNodes, nHiddenNodes, nOutputNodes, learningRate);
		this.confidenceCut = confidenceCut;
		this.summaryHistos = new ArrayList<>();
		this.confidences = new ArrayList<>();
		for(int k = 0; k < nOutputNodes; k++) {
			summaryHistos.add(new H1F("summaryHisto_"+k, nOutputNodes, 0, nOutputNodes));
			summaryHistos.get(k).setTitle("summaryHisto_"+k);
			confidences.add(new ArrayList<>());
			for(int j = 0; j < nOutputNodes; j++) {
				confidences.get(k).add(new H1F("confidences_"+k+"_"+j, 100, 0.2, 1.1));
				confidences.get(k).get(j).setTitle("confidences_true"+k+"_guess"+j);
			}
		}
	}
	
	
	public void trainNetwork(Matrix inputData, Matrix targetData) {
		nn.train(inputData, targetData);
	}
	
	
	public void testNetworkQuery(int trueIndex, Matrix queryData) {
		Matrix queryResult = nn.query(queryData);
		int networkAnswerIndex = getIndexOfMaxValue(queryResult);
		summaryHistos.get(trueIndex).fill((double) networkAnswerIndex + 0.1);
		confidences.get(trueIndex).get(networkAnswerIndex).fill(queryResult.get(networkAnswerIndex, 0));
	}
	
	
	// for N x 1 matricies only
	public static int getIndexOfMaxValue(Matrix m) {
		int answer = 0;
		for(int k = 1; k < m.getRowDimension(); k++) {
			if(m.get(k, 0) > m.get(answer, 0)) answer = k;
		}
		return answer;
	}
	
	
	public void plotResults(String name) {
		TCanvasPTabbed can = new TCanvasPTabbed(name, 1000, 800);
		can.addTab("summary");
		can.addTab("confidence");
		can.getTab("summary").divide(nn.getnOutputNodes(), 1);
		can.getTab("confidence").divide(nn.getnOutputNodes(), nn.getnOutputNodes());
		for(int j = 0; j < nn.getnOutputNodes(); j++) {
			can.getTab("summary").cd(j);
			can.getTab("summary").draw(summaryHistos.get(j));
			for(int k = 0; k < nn.getnOutputNodes(); k++) {
				can.getTab("confidence").cd(nn.getnOutputNodes()*(nn.getnOutputNodes() - k - 1) + j);
				can.getTab("confidence").draw(confidences.get(j).get(k));
			}
		}
	}

}
