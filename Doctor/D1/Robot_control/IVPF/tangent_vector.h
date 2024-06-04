#pragma once

#include "stdafx.h"

class TangentVector {
private:
	const double epsilon = 1e-6;//equal to 0
public:
	TangentVector() {
		std::cout << "construct TangentVector class" << std::endl;
	}

	std::vector<std::vector<double>> main(double& k1, double& k2, double& k3, double& rtan);

	std::vector<double> quadraticEquation(double& a, double& b, double& c);

	void checkSolution(double& k1, double& k2, double& k3, double& rtan, double& s, double& t);
};
