#pragma once


#include "stdafx.h"
#include "ur.h"
#include "minimum_dist.h"
#include "tangent_vector.h"

//structure
struct CollisionState {
	double dmin = -1.0; //minimum distance
	cv::Mat mat_dmin; //minimum distance vector, unit vector
	double speed_obs = 0; //obstacle velocity
	std::vector<double> Vobs;//obstacle velocity
	std::vector<double> point_human;
	std::vector<double> point_robot;
	double theta_rob_Vobs = 0; //angle between robot and obstacle velocity
	double theta_rob_target = 0; //angle between robot and target
};

class IVPF {
	/**
	* @brief Improved Potential Velocity FIeld for UR collision-avoidance control
	* https://www.mdpi.com/1424-8220/23/7/3754
	*/
private:
	//unit ==> length [m], angle [rad]
	const double pi = 3.14159265358979323846;
	//types of method
	//which joints to consider. int method = 0 : elbow, wrist1, and ee. method = 1: consider all joints.
	const int method_joints = 0;
	const bool bool_considerMultiRepulsive = false;
	const int method_mix = 0;//0 : use attractive field all the time. 1: use either attractive or tangential field
	//attractive field
	const double rho_pos_g0 = 0.2; //rho_pos_g0 [m] : boundary point between quadratic and linear in potential field
	const double rho_angle_g0 = 0.1; //rho_angle_g0 [rad] : boundary point between quadratic and linear in potential fild
	const double zeta_pos = 1.25; //zeta_pos : attractive potential gain
	const double zeta_angle = 3.14;//zeta_rot ::attractive potential gain for angular velocity
	const double s = 0.2;//constant value for max velcotiy : zeta_pos*s, zeta_angle*s is the max value for translational and angular velocity
	//repulsive field
	const double k = 0.01; //repulsive potential gain
	const int m = 5;//order
	const int n = 10;//constant or def (rVobs)^n
	const double a = 2; //constant coefficient for theta_rob_target
	const double b = 5;//constant coefficient for theta_rob_Vobs
	const double r = 500; //1/Vref, Vref :: referential speed for judgin obstacle is dynamic or static
	const double Vref = 0.002;//2[mm/frame]
	//minimum radius
	const double radius = 0.15; //sperical radius = 0.2 [m]
	const double Vobs_max = ( 1.0/ 200.0); //max Vobs for repulsive field; Assume that human hand velocity is 0.65 [m/sec] and frequency 200 [Hz]
	const double rho_02_repulsive = 0.35; //max valid repulsive field 
	const double rho_01_repulsive = 0.25; //minimum valide repulsive field
	const double rho_tangent = 0.45;//tangential velocity
	//repulsive field
	const double delta_dist = 0.05; //0.05 m per step
	const double delta_theta = (-pi / 60);//3.0[degree] per step
	const double k_sigmaS_ = 1; //
	//human joints pairs : {ls(0),rs(1),le(2),re(3),lw(4),rw(5)} ->{{lw,le},{le,ls},{ls,rs},{rs,re},{re,rw}}
	const std::vector<std::vector<int>> pair_joints{ {2,4},{0,2},{0,1},{1,3},{3,5} };
	//default human velocity
	const double speed_default = 0.0;
	const double alpha_speed = 0.5; //speed change rate :: alpha*Vnew+(1-alpha)*Vold
	//epsilon for evaluating equality
	const double epsilon = 1e-6;
	//inverse Jacobian matrix :: lambda =  (k_const)/(tan(d_threshold+c)+k_const)
	const bool bool_avoidSingularity = true;
	const double k_const = 1;
	const double d_threshold = (pi / 6);
	//for setting appropriate c for deviating from singular points.
	const double c1_max = 2.0e-2;
	const double c1_min = 1.0e-2;
	const double c2_max = 1.0e-2;
	const double c2_min = 1.0e-3;
	const double Kp_det = 50.0;//Kp constraint for multiplying by determinant 
	//frame rate
	const double frame_rate = 50.0;
	const double angle_max = (pi * 95 / 100);
	const double angle_min = -(pi * 95 / 100);
	const double rate_angle_repulsive = 0.8;//how much rate angle constraints work
public:
	//make instance 
	UR ur_ivpf; //construct UR kinematics class
	MinimumDist minDist_ivpf; //calculate minimum distance between 2 segments of lines
	TangentVector tv_ivpf;//calculate tangent vector
	std::vector<double> determinants; //storage for saving determinants
	std::vector<double> target_previous;//previous target pose. for stabilizing system
	cv::Mat delta_target;//diffenrence of target pose

	//storage setting
	//current robot position
	std::vector<std::vector<double>> robot_current;//{px,py,pz,nx,ny,nz} 6*6 (base,shoulder,elbow,w1,w2,end-effector)
	//human state
	std::vector<std::vector<std::vector<double>>> human_current, human_previous; //num of human, {px,py,pz,vx,vy,vz}, 6*6 (ls,rs,le,re,lw,rw)

	//collision state
	std::vector<std::vector<std::vector<CollisionState>>> state_current, state_previous; // size=(num_human,3,5) -> each link (shoulder-elbow,elbow-wrist1,wrist1-ee) * (lw-le,le-ls,ls-rs,rs-re,re-rw))
	//repulsive field result
	cv::Mat q_shoulder_repulsive, q_elbow_repulsive, q_wrist1_repulsive, q_wrist2_repulsive, q_wrist_repulsive, q_ee_repulsive, q_repulsive_total;
	cv::Mat q_shoulder_tangent, q_elbow_tangent, q_wrist1_tangent, q_wrist2_tangent, q_wrist_tangent, q_ee_tangent, q_tangent_total;
	cv::Mat q_attractive, q_total, q_constraint;

	//constructor
	IVPF(std::vector<double>& joints_init) {
		ur_ivpf.cal_poseAll(joints_init);
		robot_current = std::vector<std::vector<double>>{ ur_ivpf.pose1,ur_ivpf.pose2,ur_ivpf.pose3,ur_ivpf.pose4,ur_ivpf.pose5,ur_ivpf.pose6 };
		//init human pose
		std::vector<std::vector<double>> human_init(6, std::vector<double>(6, 0.0));
		human_current.push_back(human_init);
		human_previous.push_back(human_init);
		//init collision state
		std::vector<std::vector<CollisionState>> state_init(3, std::vector<CollisionState>(5));
		state_current.push_back(state_init);
		state_previous.push_back(state_init);
		//finish initialization
		std::cout << "construct IVPF class" << std::endl;
	};

	//main function
	void main(std::vector<double>& target, std::vector<std::vector<std::vector<double>>>& pose_human, std::vector<double>& joints_robot);

	//robot joints angle constraint
	void angleConstraint(std::vector<double>& joints_robot);

	//repulsive weight
	double weight_rep(double& d_minimum);

	//calculate attractive field
	void attract(std::vector<double>& pose_current, std::vector<double>& joints_robot, std::vector<double>& pose_target);

	//calculate space vector k and rotational angle;
	void cal_pose_rot(std::vector<double>& current_pose, std::vector<double>& target_pose, std::vector<double>& k, double& theta);

	//repulsive field for elbow
	double repulsive_notEE(std::vector<double>& target, std::vector<double>& joints_robot, int idx_joint);

	//repulsive field for end-effector
	double repulsive_ee(std::vector<double>& target, std::vector<double>& joints_robot);

	//tangent field
	cv::Mat tangentEE(std::vector<double>& point_robot, std::vector<double>& point_human, std::vector<double>& vec_obs, std::vector<double>& pose_target, double& dmin, double& rho_rep);
	cv::Mat tangent_notEE(std::vector<double>& point_robot, std::vector<double>& point_human, std::vector<double>& vec_obs, double& dmin,int& idx_joint);

	double rho_setting(double& Vobs, std::vector<double>& vec_obs, std::vector<double>& point_robot, std::vector<double>& point_human);

	void invJacobian(cv::Mat& J, cv::Mat& invJ);

	double funcC(double& c);

	double dot(std::vector<double>& vec1, std::vector<double>& vec2);

	double length(std::vector<double>& vector);

	std::vector<double> subtractVec(std::vector<double>& vec1, std::vector<double>& vec2);

	std::vector<double> addVec(std::vector<double>& vec1, std::vector<double>& vec2);

	std::vector<double> multiplyVec(double a, std::vector<double>& vec);

	std::vector<double> divVec(double a, std::vector<double>& vec);
};