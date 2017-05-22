package physics;

import java.util.Random;

public class Narticle {
	
	Nid nid;
	double p, beta, v1, v2, v3;
	
	public Narticle(Nid nid, double p, double beta, double v1, double v2, double v3) {
		this.nid = nid;
		this.p = p;
		this.beta = beta;
		this.v1 = v1;
		this.v2 = v2;
		this.v3 = v3;
	}
	
	public static Narticle initWithRandom(Nid nid) {
		Random r = new Random();
		double mass = nid.getMass();
		double p = 9.75*r.nextDouble() + 0.25;
		double trueBeta = p/Math.sqrt(mass*mass + p*p);
		double smearedBeta = 0.0075*r.nextGaussian() + trueBeta;
		double smearedV1 = 2.55*r.nextGaussian() + nid.getV1();
		double smearedV2 = 0.75*r.nextGaussian() + nid.getV2();
		double smearedV3 = 0.03*r.nextGaussian() + nid.getV3();
		return new Narticle(nid, p, smearedBeta, smearedV1, smearedV2, smearedV3);
	}

	public Nid getNid() {
		return nid;
	}

	public void setNid(Nid nid) {
		this.nid = nid;
	}

	public double getP() {
		return p;
	}

	public void setP(double p) {
		this.p = p;
	}

	public double getBeta() {
		return beta;
	}

	public void setBeta(double beta) {
		this.beta = beta;
	}

	public double getV1() {
		return v1;
	}

	public void setV1(double v1) {
		this.v1 = v1;
	}

	public double getV2() {
		return v2;
	}

	public void setV2(double v2) {
		this.v2 = v2;
	}

	public double getV3() {
		return v3;
	}

	public void setV3(double v3) {
		this.v3 = v3;
	}

}
