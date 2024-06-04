#pragma once

#include "stdafx.h"

class UR {
private:
	const double pi = 3.14159265358979323846;
	//UR geometric params
	// https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
	/**
	* DH-params
	* cv::Mat T_n2n+1 = (cv::Mat_<double>(4, 4) <<
    *    cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta),
    *    sin(theta),cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta),
    *    0.0, sin(alpha), cos(alpha), d,
    *    0.0, 0.0, 0.0, 1);
	*/
	//rotation angle [rad]
	double alpha1 = (pi / 2);
	double alpha2 = 0;
	double alpha3 = 0;
	double alpha4 = (pi / 2);
	double alpha5 = -(pi / 2);
	double alpha6 = 0; //nothing
	//link distance [m]
	double a1 = 0;


	double a2 = -0.4250;
	double a3 = -0.3922;
	double a4 = 0;
	double a5 = 0;
	double a6 = 0; //none
	//joint distance
	double d1 = 0.1625;
	double d2 = 0;
	double d3 = 0;
	double d4 = 0.1333;
	double d5 = 0.0997;
	double d6 = 0.0996;

	//matrix size definition
	const int row = 4;
	const int col = 4;

public:
	cv::Mat T01, T12, T23, T34, T45, T56; //homogeneous matrix
	cv::Mat T02, T03, T04, T05, T06; //homogeneous matrix
	std::vector<double> pose1, pose2, pose3, pose4, pose5, pose6; //each joint pose
	cv::Mat J01,J02, J03,J04,J05, J;//Jacobian matrix
	UR() {
		std::cout << "construct UR class" << std::endl;
	};

	//calculate transformation matrix. theta [rad]
	void mat01(double& theta1);
	void mat12(double& theta2);
	void mat23(double& theta3);
	void mat34(double& theta4);
	void mat45(double& theta5);
	void mat56(double& theta6);
	//calculate all homogeneous matrix
	void mat(std::vector<double>& joints);

	//calculate pose of each joint, joints : [base,shoulder,elbow,wrist1,wrist2,wrist3]
	void cal_pose1(std::vector<double>& joints,bool bool_mat=true);
	void cal_pose2(std::vector<double>& joints, bool bool_mat = true);
	void cal_pose3(std::vector<double>& joints, bool bool_mat = true);
	void cal_pose4(std::vector<double>& joints, bool bool_mat = true);
	void cal_pose5(std::vector<double>& joints, bool bool_mat = true);
	void cal_pose6(std::vector<double>& joints, bool bool_mat = true);
	void cal_poseAll(std::vector<double>& joints);

	void Jacobian01(std::vector<double>& joints);
	void Jacobian02(std::vector<double>& joints);
	void Jacobian03(std::vector<double>& joints);
	void Jacobian04(std::vector<double>& joints);
	void Jacobian05(std::vector<double>& joints);
	void Jacobian(std::vector<double>& joints);

	double determinant(cv::Mat& j);
};