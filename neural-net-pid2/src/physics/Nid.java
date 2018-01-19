package physics;

public enum Nid {
	A(0, 0.14, 20.0, 3.0, 0.9), B(1, 0.5, 25.0, 4.0, 0.8), C(2, 0.938, 30.0, 5.5, 0.7);
	
	private int id;
	private double mass, v1, v2, v3;
	
	Nid(int id, double mass, double v1, double v2, double v3) {
		this.id = id;
		this.mass = mass;
		this.v1 = v1;
		this.v2 = v2;
		this.v3 = v3;
	}

	public int getId() {
		return this.id;
	}

	public double getMass() {
	    return this.mass;
	}

	public double getV1() {
	    return this.v1;
	}

	public double getV2() {
	    return this.v2;
	}

	public double getV3() {
	    return this.v3;
	}

}