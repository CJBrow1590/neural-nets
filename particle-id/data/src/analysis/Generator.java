package analysis;

import org.jlab.analysis.plotting.TCanvasP;
import org.jlab.groot.data.H1F;
import org.jlab.groot.data.H2F;

import physics.Narticle;
import physics.Nid;

public class Generator {
	public static void main(String[] args) {
		

		
		H2F betaVp = new H2F("betaVp", 200, 0, 11, 200, 0.6, 1.05);
		betaVp.setTitleX("p");
		betaVp.setTitleY("#beta");
		H2F v2Vv1 = new H2F("v2Vv1", 200, 5, 45, 200, 0, 10);
		v2Vv1.setTitleX("v1");
		v2Vv1.setTitleY("v2");
		H1F hv3 = new H1F("hv3", 200, 0.3, 1.3);
		hv3.setTitleX("v3");

		H2F betaVpScaled = new H2F("betaVpScaled", 200, -0.1, 1.1, 200, -0.1, 1.1);
		H2F v2Vv1Scaled = new H2F("v2Vv1Scaled", 200, -0.1, 1.1, 200, -0.1, 1.1);
		H1F hv3Scaled = new H1F("hv3Scaled", 200, -0.1, 1.1);
		
		for(int k = 0; k < Integer.parseInt(args[0]); k++) {
			Nid nid;
			if(k%3 == 0) nid = Nid.A;
			else if(k%3 == 1) nid = Nid.B;
			else nid = Nid.C;
			
			Narticle nart = physics.Narticle.initWithRandom(nid);
			double p = nart.getP();
			double beta = nart.getBeta();
			double v1 = nart.getV1();
			double v2 = nart.getV2();
			double v3 = nart.getV3();
			// scaling the values so they're ~ 0.01 - 1.00 for the neural network
			double pScaled = nart.getP()/10.0;
			double betaScaled = nart.getBeta()/1.02;
			double v1Scaled = (nart.getV1() - 10.0)/30.0;
			double v2Scaled = nart.getV2()/9.0;
			double v3Scaled = (nart.getV3() - 0.5)/0.55;

			betaVp.fill(p, beta);
			v2Vv1.fill(v1, v2);
			hv3.fill(v3);
			betaVpScaled.fill(pScaled, betaScaled);
			v2Vv1Scaled.fill(v1Scaled, v2Scaled);
			hv3Scaled.fill(v3Scaled);

			System.out.println(
				nart.getNid().getId() + "," +
				pScaled + "," +
				betaScaled + "," +
				v1Scaled + "," +
				v2Scaled + "," +
				v3Scaled);
		}
		
		// plot the training data
		TCanvasP trainCan = new TCanvasP("trainCan", 1000, 300, 3, 2);
		trainCan.cd(0);
		trainCan.draw(betaVp);
		trainCan.cd(1);
		trainCan.draw(v2Vv1);
		trainCan.cd(2);
		trainCan.draw(hv3);
		trainCan.cd(3);
		trainCan.draw(betaVpScaled);
		trainCan.cd(4);
		trainCan.draw(v2Vv1Scaled);
		trainCan.cd(5);
		trainCan.draw(hv3Scaled);
		trainCan.update();
		
		
		
	}
}
