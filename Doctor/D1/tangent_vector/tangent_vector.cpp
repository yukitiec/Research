#include "tangent_vector.h"

std::vector<std::vector<double>> TangentVector::main(double& k1, double& k2, double& k3, double& rtan) {
	/**
	* @brief calculate tangent vector coefficients, {s,t}. vec = s*vec_a+t*vec_b
	* @param[in] k1 |vec_a|^2
	* @param[in] k2 dot(vec_a,vec_b)
	* @param[in] k3 |vec_b|^2
	* @param[in] rtan radius of circle
	* @return {s,t} coefficients
	*/

	double a = k1 * (k1*k3 - k2 * k2) / k2 / k2;
	double b = k1 - rtan * rtan - k1 * k1 * k3 / k2 / k2 + k1 * k3 * rtan * rtan / k2 / k2;
	double c = -k1 + rtan * rtan + k1 * k1 * k3 / k2 / k2 - 2 * k1 * k3 * rtan * rtan / k2 / k2 + k3 * std::pow(rtan, 4) / k2 / k2;
	std::vector<double> s = quadraticEquation(a, b, c);
	std::vector<std::vector<double>> coeffs;
	if (!s.empty()) {
		for (int i = 0; i < s.size(); i++) {
			double t = (k1 * (1 - s[i]) - rtan * rtan) / k2;
			coeffs.push_back(std::vector<double>{s[i], t});
			std::cout << "s=" << s[i] << ", t=" << t << std::endl;
		}
		return coeffs;
	}
	else {
		return std::vector<std::vector<double>>{};//no solution
	}
	
}

std::vector<double> TangentVector::quadraticEquation(double& a, double& b, double& c) {
	/**
	* @brief calculate quadratic equation ax^2+2bx+c = 0
	* @param[in] a,b,c coefficients
	* @return solution x = (-b(+-)(b^2-ac)^(0.5))/a
	*/
	if (a < epsilon) {
		if (b > epsilon) {
			double solution = -c / 2.0 / b;
			return std::vector<double>{solution};//one solution
		}
		else
			return std::vector<double>{};//no solution
			
	}
	double solution1  = (-b + std::pow(b * b - a * c, 0.5)) / a;
	double solution2 = (-b - std::pow(b * b - a * c, 0.5)) / a;
	std::vector<double> solutions{ solution1,solution2 };
	return solutions;
}

void TangentVector::checkSolution(double& k1, double& k2, double& k3, double& rtan,double& s, double& t) {
	/**
	* @brief check solution
	* @param[in] k1 |vec_a|^2
	* @param[in] k2 dot(vec_a,vec_b)
	* @param[in] k3 |vec_b|^2
	* @param[in] rtan radius of circle
	* @param[in] s,t coefficients
	*/

	//first condition
	double ele_left = std::pow(1 - s, 2) * k1 + 2 * (s - 1) * t * k2 + t * t * k3;
	double ele_right = rtan * rtan;
	std::cout << "first condition :: difference is " << (ele_left - ele_right) << std::endl;
	//second condition
	double ele2_left = s * (s - 1) * k1 + (2 * s * t - t) * k2 + t * t * k3;
	std::cout << "second condition :: difference is " << ele2_left << std::endl;
}
