// robotControl.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "stdafx.h"
#include "global_params.h"
#include "speedj.h"

int main()
{
	SpeedJ ur;
	std::vector<std::vector<double>> posSaver;
	std::vector<double> pose_c, pose_difference;
	std::vector<std::vector<int>> targets; //{frame,x,y,z}
	for (int i = 0; i < 4; i++)
	{
		std::cout << "targets.size()=" << targets.size() << std::endl;
		std::vector<int> temp;
		temp.push_back(i + 1);
		if (i % 2 == 0) {
			temp.push_back(0);
			temp.push_back(0);
			temp.push_back(0);
		}
		else if (i % 2 == 1) {
			temp.push_back(0);
			temp.push_back(0);
			temp.push_back(-150);
		}
		targets.push_back(temp);
	}

	std::cout << "initializing" << std::endl;

	/* Real time robot control to catch */
	double diffJ; // variable for calculating distance to target
	std::vector<double> currentJ, targetJ; // variable for current joint angles and target joint angles
	std::vector<double> differenceJ; //pose diffenrence between current and target
	std::cout << "0" << std::endl;
	// 4 seconds control
	int count_target = 0;
	//std::vector<double> target = targets[count_target];
	std::cout << "targets.shape()=" << targets.size() << std::endl;
	std::vector<int> target = targets[count_target];
	std::cout << "00" << std::endl;
	std::cout << "start communication" << std::endl;
	//std::thread threadUR(robotControl);
	std::thread threadUR(&SpeedJ::main, ur, std::ref(q_endTracking));
	int counter_target = 0;
	for (unsigned int i = 0; i < 500; i++)
	{
		std::cout << "-------<<" << i << ">>---------" << std::endl;
		//auto start_l = std::chrono::high_resolution_clock::now();
		/* Initialization */
		//differenceJ.clear();
		pose_c = urDI->getActualTCPPose();
		posSaver.push_back(std::vector<double>{double(i + 1), pose_c[0], pose_c[1], pose_c[2], pose_c[3], pose_c[4], pose_c[5]});
		//std::cout << "target:" << target[0] << "," << target[1] << ","<< target[2] << ","<< target[3] << "," << target[4] << "," << target[5] << std::endl;
		if (i % 20 == 0)
		{
			target = targets[count_target % 4];
			count_target++;
			std::vector<std::vector<std::vector<int>>> humans_target;
			std::vector<std::vector<int>> joints_target;
			for (int i = 0; i < 6; i++)
				joints_target.push_back(target);
			humans_target.push_back(joints_target);
			queueJointsPositions.push(humans_target);
			std::this_thread::sleep_for(std::chrono::milliseconds(30));
			//std::cout << "////////////////////////// target has changed /////////////////////////" << std::endl;
		}
	}
	q_endTracking.push(true);
	threadUR.join();//wait for thread to finish
	// Open the file for writing
}
