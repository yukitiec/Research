#pragma once

#include "stdafx.h"
#include "global_params.h"

class SpeedL {
private:
	// constant value setting
	const double dt_ = 0.002; //move 500 fps, every 2msec sending signal to UR
	const int SplitRate_ = 5; //divide move distance by splitRate
	const double Acceleration_ = 8.0; //set acceleration
	const double Max_speed_ = 0.25; //set velocity
	const double Max_omega_ = 0.5; // angular velocity
	const double Gain_p_speed_ = 2; // Gain for calculating velocity
	const double Gain_p_angle_ = 1.0; // Gain for calculating velocity

public:
	std::vector<std::vector<std::vector<int>>> joints_prev, joints_cur; //human previous joints' position
	std::vector<int> lw_prev, lw_cur; //previous and current left hand position
	std::vector<double> ur_cur, ur_target; //target tcp position
	std::vector<double> ur_joints_target; //joints of ur

	std::vector<std::vector<int>> targets; //{frame,x,y,z}

	std::queue<std::vector<double>> q_difference; //speed signal for moving (option if divide signal) 

	SpeedL() {
		std::cout << "construct SpeedL class" << std::endl;
	};
	~SpeedL() {};

	//robot control
	void main(std::queue<bool>& q_endTracking);

	//robot control
	void control();

	//calculate distance to the target
	void calcDist();
};