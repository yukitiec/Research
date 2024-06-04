#pragma once

#include "stdafx.h"

class MinimumDist {
private:
	const double epsilon = 1e-3; //minimum inner dots
	const int method = 2;//0: iterationAlgorithm, 1: mix of iteration and geometric, 2:only geometric
	double k1, k2, k3, k4; //constant coefficients in iterationAlgorithm
	const double lambda=0.1;//how much to move per iterate. Used in IterationAlgorithm
	double dist_previous,s_previous,delta_dist,delta_t,delta_s;
	const double threshold_dist = 0.01;//1cm order
	const double threshold_param = 0.005;//1mm order
	const int max_iteration = 100;
public:
	std::vector<double> s_list, t_list, dist_list, alpha_list;//for checking convergence

	MinimumDist() {
		std::cout << "construct minimumDist class" << std::endl;
	};

	//calculate minimum distance of 2 segments of lines
	double minDist_2lines(std::vector<double>& h1, std::vector<double>& h2, std::vector<double>& r1, std::vector<double>& r2, std::vector<double>& point_h, std::vector<double>& point_r);

	void normalize(std::vector<double>& vector, double& norm);

	double dot(std::vector<double>& vec1, std::vector<double>& vec2);

	double distance(std::vector<double>& point1, std::vector<double>& point2);

	double iterationAlgorithm(std::vector<double>& a0, std::vector<double>& b0, double& norm_a, double& norm_b, std::vector<double>&vec_a,std::vector<double>& vec_b,
		std::vector<double>& vec_AB,std::vector<double>&point_h, std::vector<double>& point_r,double s,double t);

	double find_t(double& t_max);
	double find_s(double& t,double& s_max);

	double get_alphaS(double& dfds, double& s, double& s_max);
	double get_alphaT(double& dfdt, double& t, double& t_max);

	double function_f(double& s, double& t);

	double length(std::vector<double>& vec);
};