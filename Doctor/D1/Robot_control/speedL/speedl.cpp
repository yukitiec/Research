#include "speedl.h"

void SpeedL::main(std::queue<bool>& q_endTracking)
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
					ur_target[0] = ur_cur[0] + (double)((lw_cur[1] - lw_prev[1]) / 1000); //x_target = x_current + delta(x_wrist) [mm] -> [m]
					ur_target[1] = ur_cur[1] + (double)((lw_cur[2] - lw_prev[2]) / 1000); //y_target = y_current + delta(y_wrist)
					ur_target[2] = ur_cur[2] + (double)((lw_cur[3] - lw_prev[3]) / 1000); //z_target = z_current + delta(z_wrist)
					lw_prev = lw_cur;//get left wrist position
					std::cout << "5" << std::endl;
				}
			}
			//move UR
			if (!ur_target.empty()) {
				//std::cout << "joints="<<ur_joints_target[0] << std::endl;
				//start_l = std::chrono::high_resolution_clock::now();
				//std::cout << "dt_=" << dt_ << ", Max_speed_=" << Max_speed_ << ", Acceleration_=" << Acceleration_ << std::endl;
				//urCtrl->servoJ(ur_joints_target, Max_speed_, Acceleration_, 0.01, Lookahead_, Gain_p_); 
				//urCtrl->servoJ(ur_joints_target, 0.5, 0.5, 0.002, 0.1, 800);
				//calculate difference and move UR
				//urCtrl->stopL();
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

void SpeedL::control() {
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
				urCtrl->speedL(dMove, Acceleration_, dt_);
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

void SpeedL::calcDist() {
	/**
	* @brief calculate distance between current pose and target pose
	*/
	
	//param settings
	std::vector<double> difference; //difference between current and target
	double diff,speed,omega; //each axis difference, gained speed, gained angular speed
	double norm_pos = 0.0; double norm_angle = 0.0; //init norm
	//end
	
	// calculate distance to target
	ur_cur = urDI->getActualTCPPose();
	//x
	diff = ur_target[0] - ur_cur[0];
	if (std::abs(diff) < 0.001) diff = 0.0; //under 1mm motion is ignored
	difference.push_back(diff);
	norm_pos += std::pow(diff, 2);
	//y
	diff = ur_target[1] - ur_cur[1];
	if (std::abs(diff) < 0.001) diff = 0.0;
	difference.push_back(diff);
	norm_pos += std::pow(diff, 2);
	//z
	diff = ur_target[2] - ur_cur[2];
	if (std::abs(diff) < 0.001) diff = 0.0;
	difference.push_back(diff);
	norm_pos += std::pow(diff, 2);
	//angle
	//roll
	diff = ur_target[3] - ur_cur[3];
	if (std::abs(diff) < 0.017) diff = 0.0; //under 1 degree, ignore
	difference.push_back(diff);
	norm_angle += std::pow(diff, 2);
	//pitch
	diff = ur_target[4] - ur_cur[4];
	if (std::abs(diff) < 0.017) diff = 0.0;
	difference.push_back(diff);
	norm_angle += std::pow(diff, 2);
	//yaw
	diff = ur_target[5] - ur_cur[5];
	if (std::abs(diff) < 0.017) diff = 0.0;
	difference.push_back(diff);
	norm_angle += std::pow(diff, 2);
	norm_pos = std::pow(norm_pos, 0.5); //O(vel_norm) = 10e-3~10e-2[m]
	std::cout << "Velocity :" << norm_pos << std::endl;
	norm_angle = std::pow(norm_angle, 0.5);
	//std::cout << "angular velocity :" << norm_angle << std::endl;
	//speed = Gain_p_speed_ * norm_pos; //calculatenorm
	speed = Gain_p_speed_ * log10(100 * norm_pos);
	speed = Max_speed_/(1 + std::exp(-speed)); //sigmoid version
	std::cout << "speed~" << speed << std::endl;
	//speed = ((std::exp(speed) - 1) / (std::exp(speed) + 1)) * Max_speed_; //calculate velocity
	//std::cout << "velocity after converted :" << speed << std::endl;
	omega = Gain_p_angle_ * norm_angle; //calculatenorm
	omega = ((std::exp(omega) - 1) / (std::exp(omega) + 1)) * Max_omega_; //calculate velocity
	//std::cout << "angular velocity after converted :" << omega << std::endl;
	
	//calculate each axis speed
	//x
	if (std::abs(difference[0]) >= 0.001) //ignore if under 1mm
	{
		difference[0] = (speed * difference[0] / norm_pos);
		if (std::abs(difference[0]) < 0.001) difference[0] = 0.0;
	}
	// smaller than 1 mm 
	else difference[0] = 0.0;
	//y
	if (std::abs(difference[1]) >= 0.001)
	{
		difference[1] = (speed * difference[1] / norm_pos);
		if (std::abs(difference[0]) < 0.001) difference[1] = 0.0;
	}
	// smaller than 1 mm 
	else difference[1] = 0.0;
	//z
	if (std::abs(difference[2]) >= 0.001)
	{
		difference[2] = (speed * difference[2] / norm_pos);
		if (std::abs(difference[2]) < 0.001) difference[2] = 0.0;
	}
	// smaller than 1 mm 
	else difference[2] = 0.0;
	//angle
	//roll
	if (std::abs(difference[3]) >= 0.017)
	{
		difference[3] = (omega * difference[3] / norm_angle);
		if (std::abs(difference[3]) < 0.017) difference[3] = 0.0;
	}
	// smaller than 1 mm 
	else difference[3] = 0.0;
	//pitch
	if (std::abs(difference[4]) >= 0.017)
	{
		difference[4] = (omega * difference[4] / norm_angle);
		if (std::abs(difference[4]) < 0.0001) difference[4] = 0.0;
	}
	// smaller than 1 mm 
	else difference[4] = 0.0;
	//5
	if (std::abs(difference[5]) >= 0.017)
	{
		difference[5] = (omega * difference[5] / norm_angle);
		if (std::abs(difference[5]) < 0.0001) difference[5] = 0.0;
	}
	// smaller than 1 mm 
	else difference[5] = 0.0;
	//end calculate difference between current and target
	std::cout << "velocity=";
	for (int i = 0; i < difference.size(); i++)
		std::cout << difference[i] << " ";
	std::cout << std::endl;
	//move UR
	urCtrl->speedL(difference, Acceleration_, dt_);
	//q_difference.push(difference);
}