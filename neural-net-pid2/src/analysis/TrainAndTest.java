package analysis;

import java.util.ArrayList;

import org.jlab.analysis.plotting.TCanvasP;
import org.jlab.groot.data.H1F;
import org.jlab.groot.data.H2F;

import Jama.Matrix;
import nn.NetworkDiagnostic;
import nn.NeuralNetwork;
import physics.Narticle;
import physics.Nid;

public class TrainAndTest {
	public static void main(String[] args) {
		
		
		
		ArrayList<H2F> betaVp = new ArrayList<>();
		ArrayList<H2F> v2Vv1 = new ArrayList<>();
		ArrayList<H1F> hv3 = new ArrayList<>();
		
		for(int k = 0; k < 3; k++) {
			betaVp.add(new H2F("betaVp_"+k, 200, 0, 1.05, 200, 0.4, 1.05));
			v2Vv1.add(new H2F("v2Vv1_"+k, 200, 0, 1, 200, 0, 1));
			hv3.add(new H1F("hv3"+k, 200, 0, 1));
		}

		NeuralNetwork nn = new NeuralNetwork(5, 50, 3, 0.1);

		// train network
		int nTrainingEvents = 120000;
		for(int ie = 0; ie < nTrainingEvents; ie++) {
	         Nid nid;
	         if(ie%3 == 0) nid = Nid.A;
	         else if(ie%3 == 1) nid = Nid.B;
	         else nid = Nid.C;

	         Narticle nart = physics.Narticle.initWithRandom(nid);

	         // scaling the values so they're ~ 0.01 - 1.00 for the neural network
	         Matrix input = new Matrix(new double[][] {{nart.getP()/10.0}, {nart.getBeta()/1.02}, {(nart.getV1() - 10.0)/30.0}, {nart.getV2()/9.0}, {(nart.getV3() - 0.5)/0.55}});
	         Matrix target;
	         if(nid == Nid.A) target = new Matrix(new double[][] {{0.99}, {0.01}, {0.01}});
	         else if(nid == Nid.B) target = new Matrix(new double[][] {{0.01}, {0.99}, {0.01}});
	         else target = new Matrix(new double[][] {{0.01}, {0.01}, {0.99}});
	         
	         nn.train(input, target);
		}
		
		// test network
		int nTestingEvents = 60000;
		for(int ie = 0; ie < nTestingEvents; ie++) {
	         Nid nid;
	         if(ie%3 == 0) nid = Nid.A;
	         else if(ie%3 == 1) nid = Nid.B;
	         else nid = Nid.C;

	         Narticle nart = physics.Narticle.initWithRandom(nid);
	         
	         // scaling the values so they're ~ 0.01 - 1.00 for the neural network
	         Matrix input = new Matrix(new double[][] {{nart.getP()/10.0}, {nart.getBeta()/1.02}, {(nart.getV1() - 10.0)/30.0}, {nart.getV2()/9.0}, {(nart.getV3() - 0.5)/0.55}});
	         
	         Matrix queryResult = nn.query(input);
	         
	         int networkAnswerIndex = NetworkDiagnostic.getIndexOfMaxValue(queryResult);
	         
	         if(queryResult.get(networkAnswerIndex, 0) > 0.85) {
	        	 betaVp.get(networkAnswerIndex).fill(input.get(0, 0), input.get(1, 0));
	        	 v2Vv1.get(networkAnswerIndex).fill(input.get(2, 0), input.get(3, 0));
	        	 hv3.get(networkAnswerIndex).fill(input.get(4, 0));
	         }
		}
		
		// plot results
		TCanvasP betaCan = new TCanvasP("betaCan", 1000, 500, 3, 3);
		for(int k = 0; k < 3; k++) {
			betaCan.cd(k);
			betaCan.draw(betaVp.get(k));
			betaCan.cd(k+3);
			betaCan.draw(v2Vv1.get(k));
			betaCan.cd(k+6);
			betaCan.draw(hv3.get(k));
		}
		
		
		
	}
}
