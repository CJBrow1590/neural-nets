package analysis;

import java.util.ArrayList;

import org.jlab.analysis.plotting.TCanvasP;
import org.jlab.groot.data.GraphErrors;

import Jama.Matrix;
import nn.NetworkDiagnostic;
import physics.Narticle;
import physics.Nid;

public class Optimize {
	public static void main(String[] args) {
		
		
		
		int nHNtries = 3; // try this many values for number of Hidden Nodes
		int HNmin = 50;
		int HNstep = 50;
		int nLRtries = 8; // try this many values for the learning rate
		double LRmin = 0.05;
		double LRstep = 0.05;

		ArrayList<GraphErrors> AefficiencyVLR = new ArrayList<>(); // eff vs learning rate, binned in nHiddenNodes
		ArrayList<GraphErrors> BefficiencyVLR = new ArrayList<>();
		ArrayList<GraphErrors> CefficiencyVLR = new ArrayList<>();

		ArrayList<ArrayList<NetworkDiagnostic>> diags = new ArrayList<>();

		for(int j = 0; j < nHNtries; j++) {
			diags.add(new ArrayList<>());
			AefficiencyVLR.add(new GraphErrors());
			AefficiencyVLR.get(j).setMarkerColor(j+1);
			BefficiencyVLR.add(new GraphErrors());
			BefficiencyVLR.get(j).setMarkerColor(j+1);
			CefficiencyVLR.add(new GraphErrors());
			CefficiencyVLR.get(j).setMarkerColor(j+1);

			for(int k = 0; k < nLRtries; k++) {
				diags.get(j).add(new NetworkDiagnostic(5, HNmin + HNstep*j, 3, LRmin + LRstep*k, -1.0));
			}
		}

		// train network(s)
		int nTrainingEvents = 120000;
		for(int ie = 0; ie < nTrainingEvents; ie++) {
	         Nid nid;
	         if(ie%3 == 0) nid = Nid.A;
	         else if(ie%3 == 1) nid = Nid.B;
	         else nid = Nid.C;

	         Narticle nart = physics.Narticle.initWithRandom(nid);
	         // scaling the values so they're ~ 0.01 - 1.00 for the neural network
	         double pScaled = nart.getP()/10.0;
	         double betaScaled = nart.getBeta()/1.02;
	         double v1Scaled = (nart.getV1() - 10.0)/30.0;
	         double v2Scaled = nart.getV2()/9.0;
	         double v3Scaled = (nart.getV3() - 0.5)/0.55;
	         
	         Matrix input = new Matrix(new double[][] {{pScaled}, {betaScaled}, {v1Scaled}, {v2Scaled}, {v3Scaled}});
	         Matrix target;
	         if(nid == Nid.A) target = new Matrix(new double[][] {{0.99}, {0.01}, {0.01}});
	         else if(nid == Nid.B) target = new Matrix(new double[][] {{0.01}, {0.99}, {0.01}});
	         else target = new Matrix(new double[][] {{0.01}, {0.01}, {0.99}});
	         
	         for(int j = 0; j < nHNtries; j++) {
	        	 for(int k = 0; k < nLRtries; k++) {
	        		 diags.get(j).get(k).trainNetwork(input, target);
	        	 }
	         }
		}

		// test network(s)
		int nTestingEvents = 12000;
		for(int ie = 0; ie < nTestingEvents; ie++) {
	         Nid nid;
	         if(ie%3 == 0) nid = Nid.A;
	         else if(ie%3 == 1) nid = Nid.B;
	         else nid = Nid.C;

	         Narticle nart = physics.Narticle.initWithRandom(nid);
	         // scaling the values so they're ~ 0.01 - 1.00 for the neural network
	         double pScaled = nart.getP()/10.0;
	         double betaScaled = nart.getBeta()/1.02;
	         double v1Scaled = (nart.getV1() - 10.0)/30.0;
	         double v2Scaled = nart.getV2()/9.0;
	         double v3Scaled = (nart.getV3() - 0.5)/0.55;
	         
	         Matrix input = new Matrix(new double[][] {{pScaled}, {betaScaled}, {v1Scaled}, {v2Scaled}, {v3Scaled}});
	         
	         for(int j = 0; j < nHNtries; j++) {
	        	 for(int k = 0; k < nLRtries; k++) {
	        		 diags.get(j).get(k).testNetworkQuery(nid.getId(), input);
	        	 }
	         }
		}
		
		// plot results
		TCanvasP can = new TCanvasP("can", 1000, 500, 3, 1);

		for(int j = 0; j < nHNtries; j++) {
			for(int k = 0; k < nLRtries; k++) {
				AefficiencyVLR.get(j).addPoint(LRmin + k*LRstep, diags.get(j).get(k).summaryHistos.get(0).getBinContent(0)/(nTestingEvents/3.0), 0.0, 0.0);
				BefficiencyVLR.get(j).addPoint(LRmin + k*LRstep, diags.get(j).get(k).summaryHistos.get(1).getBinContent(1)/(nTestingEvents/3.0), 0.0, 0.0);
				CefficiencyVLR.get(j).addPoint(LRmin + k*LRstep, diags.get(j).get(k).summaryHistos.get(2).getBinContent(2)/(nTestingEvents/3.0), 0.0, 0.0);
				
				can.cd(0);
				can.getCanvas().getPad(0).getAxisY().setRange(0.82, 1.02);
				can.draw(AefficiencyVLR.get(j), "same");
				can.cd(1);
				can.getCanvas().getPad(1).getAxisY().setRange(0.82, 1.02);
				can.draw(BefficiencyVLR.get(j), "same");
				can.cd(2);
				can.getCanvas().getPad(2).getAxisY().setRange(0.82, 1.02);
				can.draw(CefficiencyVLR.get(j), "same");

				diags.get(j).get(k).plotResults("can_"+j+"_"+k);
			}
		}
		
		
		
	}
}
