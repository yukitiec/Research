#include "speedj.h"

void SpeedJ::main(std::queue<bool>& q_endTracking)
{
	/**
	* @brief get latest data from main thread and move UR with speedL
	* @param[in] q_endTracking signal for ending while loop
	*/

	std::vector<double> dMove;
	bool boolMove = false; //can move?
	auto start = std::chrono::high_resolution_clock::now();
	while (true)
	{
		if (!queueJointsPositions.empty()) break;
		//std::cout << "wait for target data" << std::endl;
	}
	std::cout << "start controlling UR" << std::endl;
	std::vector<std::vector<double>> posSaver;//storage: {frame,x,y,z,roll,pitch,yaw}
	int counter_iteration = 0;
	auto start_l = std::chrono::high_resolution_clock::now();
	while (true) // continue until finish
	{
		if (!q_endTracking.empty()) {
			urCtrl->stopJ(); //stop UR
			break;
		}
		else
		{
			//std::chrono::steady_clock::time_point t_start = urCtrl->initPeriod();
			//std::cout << "wait for next data, and queueJointsPositions.emty()=" << queueJointsPositions.empty() << std::endl;
			/* new Joints position available */
			if (!queueJointsPositions.empty())
			{
				//get data from queue
				std::cout << "1" << std::endl;
				joints_cur = queueJointsPositions.front();
				queueJointsPositions.pop();
				//update joints position
				if (joints_cur[0][4][0] >= 0) lw_cur = joints_cur[0][4];//get left wrist position
				else {//current detection was faileed
					if (!lw_prev.empty()) { //previous detection was succeeded
						if (lw_prev[0] >= 0) lw_cur = lw_prev; //adopt previous data
					}
				}
				std::cout << "2- lw_cur[0]=" << lw_cur[0] << std::endl;
				//first detection was failed
				if (counter_iteration == 0 || lw_cur[0] < 0) {
					joints_prev = joints_cur;
					if (lw_cur[0] >= 0) {
						lw_prev = lw_cur;//get left wrist position
						std::cout << "get previous data" << std::endl;
						counter_iteration++;
						std::cout << "3" << std::endl;
						continue;
					}
				}
				else {
					std::cout << "4" << std::endl;
					//robot operation process
					ur_cur = urDI->getActualTCPPose();
					ur_target = ur_cur;
					std::cout << "lw_prev=";
					for (int i = 0; i < lw_prev.size(); i++)
						std::cout << " " << lw_prev[i];
					std::cout << std::endl;
					std::cout << "lw_cur=";
					for (int i = 0; i < lw_cur.size(); i++)
						std::cout << " " << lw_cur[i];
					std::cout << std::endl;
					std::cout << "ur_cur.size()=" << ur_cur.size() << std::endl;

					posSaver.push_back(std::vector<double>{double(lw_cur[0]), ur_cur[0], ur_cur[1], ur_cur[2], ur_cur[3], ur_cur[4], ur_cur[5]});
					//calculate target position
					ur_target[0] = ur_cur[0] + (double)((lw_cur[1] - lw_prev[1]) / 100); //x_target = x_current + delta(x_wrist) [mm] -> [m]
					ur_target[1] = ur_cur[1] + (double)((lw_cur[2] - lw_prev[2]) / 100); //y_target = y_current + delta(y_wrist)
					ur_target[2] = ur_cur[2] + (double)((lw_cur[3] - lw_prev[3]) / 100); //z_target = z_current + delta(z_wrist)
					std::cout << "5" << std::endl;
					//from task-space to configure-space
					if (urCtrl->getInverseKinematicsHasSolution(ur_target))
					{
						ur_joints_target = urCtrl->getInverseKinematics(ur_target); //get target angle
					}
					//update previous lw position
					/*std::cout << "Joints=";
					for (int i = 0; i < ur_joints_target.size(); i++)
					{
						std::cout << ur_joints_target[i] << ", ";
					}
					std::cout << std::endl;
					*/
					lw_prev = lw_cur;
				}
			}
			//move UR
			if (!ur_joints_target.empty()) {
				calcDist();
				//urCtrl->waitPeriod(t_start);
				while (true) {
					auto end_l = std::chrono::high_resolution_clock::now();
					auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_l - start_l);
					double t_elapsed = duration.count();
					//std::cout << "Time taken: by calculation " << t_elapsed * 0.001 << " milliseconds, dt_=" << dt_*1000 << std::endl;

					if (t_elapsed >= dt_ * 1000 * 1000) { //dt_*1000
						break;
					}
				}
				auto end_l = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_l - start_l);
				std::cout << "Time taken: by calculation " << duration.count() * 0.001 << " milliseconds" << std::endl;
				start_l = std::chrono::high_resolution_clock::now();
			}
		}
	}
	//save data
	std::string file = "ur.csv";
	std::ofstream outputFile(file);
	if (!outputFile.is_open())
	{
		std::cerr << "Error: Could not open the file." << std::endl;
	}
	for (int i = 0; i < posSaver.size(); i++)
	{
		for (int j = 0; j < posSaver[i].size(); j++)
		{

			outputFile << posSaver[i][j];
			if (j != posSaver[i].size() - 1)
			{
				outputFile << ",";
			}
			else
				outputFile << "\n";
		}
	}
	// close file
	outputFile.close();
	if (!q_endTracking.empty()) q_endTracking.pop();
}

void SpeedJ::control() {
	/**
	* @brief divide move signal into SplitRate_
	*/
	std::vector<double> dMove; //robot velocity signal
	bool boolMove = false; //whether robot can move
	auto start = std::chrono::high_resolution_clock::now();
	while (true)
	{
		// get new data
		if (!q_difference.empty())
		{
			dMove = q_difference.front();
			dMove = { dMove[0] / SplitRate_,dMove[1] / SplitRate_,dMove[2] / SplitRate_,dMove[3] / SplitRate_,dMove[4] / SplitRate_,dMove[5] / SplitRate_ };
			boolMove = true;
			q_difference.pop();
			auto start = std::chrono::high_resolution_clock::now();
		}
		// move robot
		if (!dMove.empty() && boolMove)
		{
			int counter = 0;
			while (q_difference.empty() && counter < 5)
			{
				urCtrl->speedJ(dMove, Acceleration_, dt_);
				counter += 1;
			}
			boolMove = false;
			start = std::chrono::high_resolution_clock::now();
		}
		// wait for next motivation
		else
		{
			//nothing to do
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end - start;
		if (duration.count() > 10.0) break;
	}
	//stop 
	urCtrl->stopL();
	urCtrl->stopScript();
}

void SpeedJ::calcDist() {
	/**
	* @brief calculate distance between current pose and target pose
	*/

	//param settings
	std::vector<double> difference; //difference between current and target
	double diff, speed, omega; //each axis difference, gained speed, gained angular speed
	double norm_pos = 0.0; double norm_angle = 0.0; //init norm
	//end

	// calculate distance to target in joint space
	ur_cur = urDI->getActualQ(); //get current joints
	//base
	diff = ur_joints_target[0] - ur_cur[0];
	if (std::abs(diff) < 0.017) diff = 0.0; //under 1 degree motion is ignored
	difference.push_back(diff);
	//shouler
	diff = ur_joints_target[1] - ur_cur[1];
	if (std::abs(diff) < 0.017) diff = 0.0;
	difference.push_back(diff);
	//elbow
	diff = ur_joints_target[2] - ur_cur[2];
	if (std::abs(diff) < 0.017) diff = 0.0;
	difference.push_back(diff);
	//angle
	//wrist-pitch
	diff = ur_joints_target[3] - ur_cur[3];
	if (std::abs(diff) < 0.017) diff = 0.0; //under 1 degree, ignore
	difference.push_back(diff);
	//wrist-pitch
	diff = ur_target[4] - ur_cur[4];
	if (std::abs(diff) < 0.017) diff = 0.0;
	difference.push_back(diff);
	//wrist-yaw
	diff = ur_target[5] - ur_cur[5];
	if (std::abs(diff) < 0.005) diff = 0.0;
	difference.push_back(diff);
	std::cout << "Difference=";
	for (int i = 0; i < difference.size(); i++)
		std::cout << difference[i] << " ";
	std::cout << std::endl;
	//move UR
	urCtrl->speedJ(difference, Acceleration_, dt_);
	//q_difference.push(difference);
}