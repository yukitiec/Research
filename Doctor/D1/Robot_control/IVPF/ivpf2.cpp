#include "ivpf2.h"


//main function
void IVPF::main(std::vector<double>& target, std::vector<std::vector<std::vector<double>>>& pose_human, std::vector<double>& joints_robot) {
    /**
    * @brief calculate path planing
    * @param[in] target : target 3D positino {px,py,pz,nx,ny,nz}
    * @param[in] pose_human : (num of human, num of joints, position{x,y,z}), {ls(0),rs(1),le(2),re(3),lw(4),rw(5)}
    * @param[in] joints_robot : joints angle of robot {base,shoulder,elbow,wrist1,wrist2,wrist3}
    * @return dmin : minimum distance between robot and obstacles. if -1, no repulsive motion, else if positive, repulsive motion is necessary
    */

    //prepare basic info
    q_total = cv::Mat::zeros(6, 1, CV_64F);
    //calculate all the joints position of rob
    ur_ivpf.cal_poseAll(joints_robot);
    robot_current = std::vector<std::vector<double>>{ ur_ivpf.pose1,ur_ivpf.pose2,ur_ivpf.pose3,ur_ivpf.pose4,ur_ivpf.pose5,ur_ivpf.pose6 };
    //std::cout << "current robot position:: x=" << robot_current[5][0] << ", y=" << robot_current[5][1] << ", z=" << robot_current[5][2] << std::endl;
    //attractive field
    attract(robot_current.back(), joints_robot, target);//q_attractive
    angleConstraint(joints_robot);//joints angle constraint
    if (!pose_human.empty()) {//obstacles is in the environment
        for (int i = 0; i < pose_human.size(); i++) { //for each human

            /**  update human status  **/
            if (i >= human_previous.size()) {//new human -> initialization ***care about access collision in multi-thread***
                std::vector<std::vector<double>> human_init(6, std::vector<double>(6, 0.0)), joints_current;
                human_previous.push_back(human_init);
                for (int j = 0; j < human_init.size(); j++) {//add current data and calculate velocity
                    std::vector<double> joint;//{px,py,pz,vx,vy,vz}
                    joint = std::vector<double>{ pose_human[i][j][0],pose_human[i][j][1],pose_human[i][j][2], speed_default,speed_default, speed_default };
                    joints_current.push_back(joint);
                }
                human_current.push_back(joints_current);
                human_previous = human_current; //update data
                if (method_joints == 0) {//consider elbow, wrist1 and ee as significant joints
                    //init collision state
                    std::vector<std::vector<CollisionState>> state_init(3, std::vector<CollisionState>(5));
                    state_current.push_back(state_init);
                }
                else if (method_joints == 1) {//consider all joints. shoulder,elbow,wrist1,wrist2,ee
                    //init collision state
                    std::vector<std::vector<CollisionState>> state_init(5, std::vector<CollisionState>(5));
                    state_current.push_back(state_init);
                }
                //state_previous.push_back(state_init);
            }
            else {//existed human -> update position and velocity
                std::vector<std::vector<double>> joints_current;
                for (int j = 0; j < pose_human[i].size(); j++) {//add current data and calculate velocity for each joint
                    std::vector<double> joint;//{px,py,pz,vx,vy,vz}
                    if (human_previous[i][j][0] == 0)//init data
                        joint = std::vector<double>{ pose_human[i][j][0],pose_human[i][j][1],pose_human[i][j][2], speed_default,speed_default, speed_default };
                    else {//already updated -> compare with previous data
                        double speed_x = alpha_speed * (pose_human[i][j][0] - human_previous[i][j][0]) + (1 - alpha_speed) * human_previous[i][j][3];
                        double speed_y = alpha_speed * (pose_human[i][j][1] - human_previous[i][j][1]) + (1 - alpha_speed) * human_previous[i][j][4];
                        double speed_z = alpha_speed * (pose_human[i][j][2] - human_previous[i][j][2]) + (1 - alpha_speed) * human_previous[i][j][5];
                        joint = std::vector<double>{ pose_human[i][j][0],pose_human[i][j][1],pose_human[i][j][2], speed_x,speed_y,speed_z };
                    }
                    joints_current.push_back(joint);
                }
                human_current[i] = joints_current;
                human_previous = human_current;//update human state
            }
        }
        //std::cout << "3" << std::endl;
        /**  end updating human status  **/

        //repulsive field
        double d_minimum = 100;
        if (method_joints == 0) {//consider elbow, wrist1, and ee.
            //elbow
            double d_elbow = repulsive_notEE(target, joints_robot, 0);
            //wrist
            double d_wrist = repulsive_notEE(target, joints_robot, 1);
            //end-effector
            double d_ee = repulsive_ee(target, joints_robot);
            //linearly calculate repulsive velocity based on minimum distance
            double w_sum = 0;
            if (bool_considerMultiRepulsive) {//consider all repulsive field
                if (d_elbow > epsilon) {
                    w_sum += (1 / d_elbow);
                    if (d_elbow < d_minimum) d_minimum = d_elbow;
                }
                if (d_wrist > epsilon) {
                    w_sum += (1 / d_wrist);
                    if (d_wrist < d_minimum) d_minimum = d_wrist;
                }
                if (d_ee > epsilon) {
                    w_sum += (1 / d_ee);
                    if (d_ee < d_minimum) d_minimum = d_ee;
                }
                //prepare repulsive velocity
                q_repulsive_total = cv::Mat::zeros(6, 1, CV_64F);
                if (d_elbow > epsilon) {
                    for (int i = 0; i < q_elbow_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = (1 / d_elbow / w_sum) * q_elbow_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = (1 / d_elbow / w_sum) * q_elbow_tangent.at<double>(i);
                    }
                }
                if (d_wrist > epsilon) {
                    for (int i = 0; i < q_wrist_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = (1 / d_wrist / w_sum) * q_wrist_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = (1 / d_wrist / w_sum) * q_wrist_tangent.at<double>(i);
                    }
                }
                if (d_ee > epsilon) {
                    for (int i = 0; i < q_ee_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = (1 / d_ee / w_sum) * q_ee_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = (1 / d_ee / w_sum) * q_ee_tangent.at<double>(i);
                    }
                }
            }
            else {//consider only one repulsive field
                int idx_joint = 0;
                if (d_elbow > epsilon) {
                    if (d_elbow < d_minimum) {
                        d_minimum = d_elbow;
                    }
                }
                if (d_wrist > epsilon) {
                    if (d_wrist < d_minimum) {
                        d_minimum = d_wrist;
                        idx_joint = 1;
                    }
                }
                if (d_ee > epsilon) {
                    if (d_ee < d_minimum)
                    {
                        d_minimum = d_ee;
                        idx_joint = 2;
                    }
                }
                std::cout << "idx_jooint=" << idx_joint << std::endl;
                q_repulsive_total = cv::Mat::zeros(6, 1, CV_64F);
                if (idx_joint == 0) {
                    for (int i = 0; i < q_elbow_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = q_elbow_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = q_elbow_tangent.at<double>(i);
                    }
                }
                else if (idx_joint == 1) {
                    for (int i = 0; i < q_wrist_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = q_wrist_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = q_wrist_tangent.at<double>(i);
                    }
                }
                else if (idx_joint == 2) {
                    for (int i = 0; i < q_ee_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = q_ee_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = q_ee_tangent.at<double>(i);
                    }
                }
            }
        }
        else if (method_joints == 1) {//consider all joints, shoulder, elbow, wrist1,wrist2,ee.
            //shoulder
            double d_shoulder = repulsive_notEE(target, joints_robot, 0);
            //elbow
            double d_elbow = repulsive_notEE(target, joints_robot, 1);
            //wrist1
            double d_wrist1 = repulsive_notEE(target, joints_robot, 2);
            //wrist2
            double d_wrist2 = repulsive_notEE(target, joints_robot, 3);
            //ee
            double d_ee = repulsive_ee(target, joints_robot);
            //linearly calculate repulsive velocity based on minimum distance
            if (bool_considerMultiRepulsive) {
                double w_sum = 0;
                if (d_shoulder > epsilon) {
                    w_sum += (1 / d_shoulder);
                    if (d_shoulder < d_minimum) d_minimum = d_shoulder;
                }
                if (d_elbow > epsilon) {
                    w_sum += (1 / d_elbow);
                    if (d_elbow < d_minimum) d_minimum = d_elbow;
                }
                if (d_wrist1 > epsilon) {
                    w_sum += (1 / d_wrist1);
                    if (d_wrist1 < d_minimum) d_minimum = d_wrist1;
                }
                if (d_wrist2 > epsilon) {
                    w_sum += (1 / d_wrist2);
                    if (d_wrist2 < d_minimum) d_minimum = d_wrist2;
                }
                if (d_ee > epsilon) {
                    w_sum += (1 / d_ee);
                    if (d_ee < d_minimum) d_minimum = d_ee;
                }
                //prepare repulsive velocity
                q_repulsive_total = cv::Mat::zeros(6, 1, CV_64F);
                if (d_shoulder > epsilon) {
                    for (int i = 0; i < q_shoulder_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = (1 / d_shoulder / w_sum) * q_shoulder_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = (1 / d_shoulder / w_sum) * q_shoulder_tangent.at<double>(i);
                    }
                }
                if (d_elbow > epsilon) {
                    for (int i = 0; i < q_elbow_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = (1 / d_elbow / w_sum) * q_elbow_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = (1 / d_elbow / w_sum) * q_elbow_tangent.at<double>(i);
                    }
                }
                if (d_wrist1 > epsilon) {
                    for (int i = 0; i < q_wrist1_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = (1 / d_wrist1 / w_sum) * q_wrist1_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = (1 / d_wrist1 / w_sum) * q_wrist1_tangent.at<double>(i);
                    }
                }
                if (d_wrist2 > epsilon) {
                    for (int i = 0; i < q_wrist2_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = (1 / d_wrist2 / w_sum) * q_wrist2_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = (1 / d_wrist2 / w_sum) * q_wrist2_tangent.at<double>(i);
                    }
                }
                if (d_ee > epsilon) {
                    for (int i = 0; i < q_ee_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = (1 / d_ee / w_sum) * q_ee_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = (1 / d_ee / w_sum) * q_ee_tangent.at<double>(i);
                    }
                }
            }
            else {
                int idx_joint = 0;
                if (d_shoulder > epsilon) {
                    if (d_shoulder < d_minimum) d_minimum = d_shoulder;
                }
                if (d_elbow > epsilon) {
                    if (d_elbow < d_minimum) {
                        d_minimum = d_elbow;
                        idx_joint = 1;
                    }
                }
                if (d_wrist1 > epsilon) {
                    if (d_wrist1 < d_minimum) {
                        d_minimum = d_wrist1;
                        idx_joint = 2;
                    }
                }
                if (d_wrist2 > epsilon) {
                    if (d_wrist2 < d_minimum) {
                        d_minimum = d_wrist2;
                        idx_joint = 3;
                    }
                }
                if (d_ee > epsilon) {
                    if (d_ee < d_minimum) {
                        d_minimum = d_ee;
                        idx_joint = 4;
                    }
                }
                //prepare repulsive velocity
                q_repulsive_total = cv::Mat::zeros(6, 1, CV_64F);
                if (idx_joint == 0) {
                    for (int i = 0; i < q_shoulder_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = q_shoulder_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = q_shoulder_tangent.at<double>(i);
                    }
                }
                else if (idx_joint == 1) {
                    for (int i = 0; i < q_elbow_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = q_elbow_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = q_elbow_tangent.at<double>(i);
                    }
                }
                else if (idx_joint == 2) {
                    for (int i = 0; i < q_wrist1_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = q_wrist1_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = q_wrist1_tangent.at<double>(i);
                    }
                }
                else if (idx_joint == 3) {
                    for (int i = 0; i < q_wrist2_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = q_wrist2_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = q_wrist2_tangent.at<double>(i);
                    }
                }
                else if (idx_joint == 4) {
                    for (int i = 0; i < q_ee_repulsive.rows; i++) {
                        q_repulsive_total.at<double>(i) = q_ee_repulsive.at<double>(i);
                        q_tangent_total.at<double>(i) = q_ee_tangent.at<double>(i);
                    }
                }
            }
        }
        //total velocity
        if (d_minimum < 100) {//repulsive force is loaded
            double w_rep = weight_rep(d_minimum);
            double w_att = 1 - w_rep;
            q_total = q_attractive;// +q_constraint;

            //if (d_minimum <= (radius+0.05))
            //q_total = q_repulsive_total;
            //else
            //q_total = q_repulsive_total + q_attractive+q_constraint;

        }
        else
            q_total = q_attractive;//+ q_constraint;
    }
    else//obstacles aren't in environment
    {
        q_total = q_attractive;// + q_constraint;
    }
    //set previous target pose
    target_previous = target;
    //std::cout << "q_total=" << q_total << std::endl;
    //std::cout << "q_constraint=" << q_constraint << std::endl;
    //send robot signal
    for (int i = 0; i < joints_robot.size(); i++) {
        joints_robot[i] += (q_total.at<double>(i) / frame_rate);///frame_rate;
        //joints_robot[i] = std::max(std::min(joints_robot[i]+(q_total.at<double>(i)/frame_rate),pi),-pi);///frame_rate;
        //std::cout << i << "-th joint : " << joints_robot[i] << std::endl;
    }
}

void IVPF::angleConstraint(std::vector<double>& joints_robot) {
    /**
    * @brief robot joints angle constraing
    * @param[in] current robot joints angle
    */

    q_constraint = cv::Mat::zeros(6, 1, CV_64F);
    int count = 0;
    for (double& angle : joints_robot) {//for each joint angle
        if (angle >= rate_angle_repulsive * angle_max) {
            double theta_constraint = (pi / 2.0) * (angle - rate_angle_repulsive * angle_max) / ((1 - rate_angle_repulsive) * angle_max);
            //std::cout << "theta=" << theta_constraint << std::endl;
            double delta_angle = -pi * sin(theta_constraint);//negative action
            q_constraint.at<double>(count) += delta_angle;
        }
        else if (angle <= rate_angle_repulsive * angle_min) {
            double theta_constraint = (pi / 2.0) * (rate_angle_repulsive * angle_min - angle) / ((rate_angle_repulsive - 1) * angle_min);
            double delta_angle = pi * sin(theta_constraint);//positive action
            q_constraint.at<double>(count) += delta_angle;
        }
        count++;
    }
}


double IVPF::weight_rep(double& d_minimum) {
    /**
    * @brief calculate weight for repulsive field
    * @param[in] d_minimum : minimum distance between robot and obstacles
    */
    if (d_minimum <= radius)
        return 1.0;
    else {
        return std::exp(-(d_minimum - radius) / 0.1);
    }
}

void IVPF::attract(std::vector<double>& pose_current, std::vector<double>& joints_robot, std::vector<double>& pose_target) {
    /**
    * @brief define attract motion
    * @param[in] current_pose current pose {px,py,pz,nx,ny,nz}
    * @param[in] joints_robot : joints angle of robot {base,shoulder,elbow,wrist1,wrist2,wrist3}
    * @param[in] target_pose target pose {px,py,pz,nx,ny,nz}
    * @param[in] velocity : attractive joint velocity
    */

    std::vector<double> n_axis;//rotational vector
    double n_angle;//rotational angle
    //calculate delta rotational vector
    cal_pose_rot(pose_current, pose_target, n_axis, n_angle);
    //vecotor to the target 
    std::vector<double> vec2target{ pose_target[0] - pose_current[0], pose_target[1] - pose_current[1],pose_target[2] - pose_current[2],n_axis[0],n_axis[1],n_axis[2] };
    double rho = std::pow(vec2target[0] * vec2target[0] + vec2target[1] * vec2target[1] + vec2target[2] * vec2target[2], 0.5); //distatnce to the target
    double speed, omega;
    cv::Mat velocity = cv::Mat::zeros(6, 1, CV_64F);
    //translational speed
    if (rho > rho_pos_g0)
        speed = s * zeta_pos;
    else
        speed = zeta_pos * rho;
    //angular speed
    if (n_angle > rho_angle_g0)
        omega = s * zeta_angle;
    else
        omega = zeta_angle * n_angle;
    //std::cout << "speed=" << speed << ", omega=" << omega << std::endl;
    for (int i = 0; i < 6; i++) {
        if (i < 3) {//translation
            velocity.at<double>(i) = speed * (vec2target[i] / rho);
        }
        else {//angular
            velocity.at<double>(i) = omega * (vec2target[i] / n_angle);
        }
        //std::cout << "velocity-" << i << "=" << velocity.at<double>(i) << std::endl;;
    }

    //check target pose difference
    delta_target = cv::Mat::zeros(6, 1, CV_64F);
    if (!target_previous.empty()) {//previous target pose exists
        for (int i = 0; i < target_previous.size(); i++)
            delta_target.at<double>(i) = pose_target[i] - target_previous[i];//dx_d/dt
    }
    //convert end-effector velocity into joint-space
    ur_ivpf.Jacobian(joints_robot);
    cv::Mat J = ur_ivpf.J;
    cv::Mat JT;
    cv::transpose(J, JT);  // Compute J^T
    cv::Mat invJ;
    invJacobian(J, invJ); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
    q_attractive = invJ * delta_target + JT * velocity;//{JtJ+lambda*I}^(-1)Jt*dx_d/dt+KJt*deltaX;
    //q_attractive = invJ * velocity; //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
    determinants.push_back(ur_ivpf.determinant(J));
    //std::cout << "determinant=" << ur_ivpf.determinant(J) << std::endl;
    //std::cout << "q_attractive=" << q_attractive << std::endl;
}

void IVPF::cal_pose_rot(std::vector<double>& current_pose, std::vector<double>& target_pose, std::vector<double>& k, double& theta) {
    /**
    * @brief calculate space vector and rotation angle between current pose and target pose
    * @param[in] current_pose current pose {px,py,pz,nx,ny,nz}
    * @param[in] target_pose target pose {px,py,pz,nx,ny,nz}
    * @param[out] k rotational axis{nx,ny,nz}
    * @oaram[out] theta rotational angle [rad]
    */

    // Define the rotation vector
    cv::Mat rotation_vector1 = (cv::Mat_<double>(3, 1) << current_pose[3], current_pose[4], current_pose[5]); // current rotation vector
    cv::Mat rotation_vector2 = (cv::Mat_<double>(3, 1) << target_pose[3], target_pose[4], target_pose[5]); // target rotation vector

    // Declare the rotation matrix
    cv::Mat R1, R2;//current and target rotational matrix

    // Convert the rotation vector to rotation matrix using Rodrigues
    cv::Rodrigues(rotation_vector1, R1);
    cv::Rodrigues(rotation_vector2, R2);

    cv::Mat R1_inv;
    cv::invert(R1, R1_inv);

    cv::Mat R12 = R2 * R1_inv;

    //convert rotation matrix to rotation vector
    cv::Mat rotation_transform; //space vector -> k
    cv::Rodrigues(R12, rotation_transform);

    k = std::vector<double>{ rotation_transform.at<double>(0),rotation_transform.at<double>(1),rotation_transform.at<double>(2) };
    //rotational angle
    theta = cv::norm(rotation_transform);
}

//repulsive field for elbow
double IVPF::repulsive_notEE(std::vector<double>& target, std::vector<double>& joints_robot, int idx_joint) {
    /**
    * @brief calculate repulsive_field for other than end-effector
    * @param[in] target : target 3D positino {x,y,z}
    * @param[in] pose_human : (num of human, num of joints, position{x,y,z})
    * @param[in] joints_robot : joints angle of robot {base,shoulder,elbow,wrist1,wrist2,wrist3}
    * @param[in] idx_joint : 0: elbow, 1: wrist, 2:end-effector
    * @return dmin : minimum distance between robot and obstacles. if -1, no repulsive motion, else if positive, repulsive motion is necessary
    */
    //calculate all the joints position of robot 
    //ur_ivpf.cal_poseAll(joints_robot);
    //robot_current = std::vector<std::vector<double>>{ ur_ivpf.pose1,ur_ivpf.pose2,ur_ivpf.pose3,ur_ivpf.pose4,ur_ivpf.pose5,ur_ivpf.pose6 };

    //calculate minimum distance between shoulder-elbow line and human joints link
    double dist_minimum = rho_02_repulsive;
    std::vector<int> index_minimum{ -1,-1 };
    for (int i = 0; i < human_current.size(); i++) { //for each human

        /**  geometric relationship between robot link and human one  **/
        int count = 0;
        for (std::vector<int> pair : pair_joints) { //for each joint pair
            std::vector<double> point_human, point_robot;
            //for saving collisionState storage
            CollisionState tempState;
            //calculate minimum distance -> difference for each link elbow->(robot_current[1], robot_current[2]), wrist->(robot_current[2], robot_current[3]), end-effector->(robot_current[3], robot_current[5])
            double dmin;
            if (method_joints == 0) {//consider only elbow, wrist1 and ee.
                if (idx_joint == 0)//shoulder-elbow
                    dmin = minDist_ivpf.minDist_2lines(human_current[i][pair[0]], human_current[i][pair[1]], robot_current[1], robot_current[2], point_human, point_robot); //shoulder-elbow
                else if (idx_joint == 1)//elbow-wrist
                    dmin = minDist_ivpf.minDist_2lines(human_current[i][pair[0]], human_current[i][pair[1]], robot_current[2], robot_current[3], point_human, point_robot); //elbow-wrist
            }
            else if (method_joints == 1) {//consider all joints
                if (idx_joint == 0)//base-shoulder
                    dmin = minDist_ivpf.minDist_2lines(human_current[i][pair[0]], human_current[i][pair[1]], robot_current[0], robot_current[1], point_human, point_robot); //base-elbow
                else if (idx_joint == 1)//shoulder-elbow
                    dmin = minDist_ivpf.minDist_2lines(human_current[i][pair[0]], human_current[i][pair[1]], robot_current[1], robot_current[2], point_human, point_robot); //shoulder-elbow
                else if (idx_joint == 2)//elbow^-wrist1
                    dmin = minDist_ivpf.minDist_2lines(human_current[i][pair[0]], human_current[i][pair[1]], robot_current[2], robot_current[3], point_human, point_robot); //elbow-wrist1
                else if (idx_joint == 3)//wrist1-wrist2
                    dmin = minDist_ivpf.minDist_2lines(human_current[i][pair[0]], human_current[i][pair[1]], robot_current[3], robot_current[4], point_human, point_robot); //shoulder-elbow
            }
            //dmin
            tempState.dmin = dmin;
            if (dmin <= dist_minimum) { //minimum distance -> save dmin and index
                dist_minimum = dmin;
                index_minimum = std::vector<int>{ i,count }; //idx_human, joint
            }
            //calculate Vobs, theta_rob_Vobs,theta_rob_target
            //Vobs, speed_obs
            std::vector<double> Vobs;
            double dist_point_human = std::pow(std::pow(point_human[0] - human_current[i][pair[0]][0], 2) + std::pow(point_human[1] - human_current[i][pair[0]][1], 2) + std::pow(point_human[2] - human_current[i][pair[0]][2], 2), 0.5);
            double dist_link_human = std::pow(std::pow(human_current[i][pair[1]][0] - human_current[i][pair[0]][0], 2) + std::pow(human_current[i][pair[1]][1] - human_current[i][pair[0]][1], 2) + std::pow(human_current[i][pair[1]][2] - human_current[i][pair[0]][2], 2), 0.5);
            double speed_obstacle_x = (dist_point_human / dist_link_human) * human_current[i][pair[1]][3] + (1 - dist_point_human / dist_link_human) * human_current[i][pair[0]][3];
            double speed_obstacle_y = (dist_point_human / dist_link_human) * human_current[i][pair[1]][4] + (1 - dist_point_human / dist_link_human) * human_current[i][pair[0]][4];
            double speed_obstacle_z = (dist_point_human / dist_link_human) * human_current[i][pair[1]][5] + (1 - dist_point_human / dist_link_human) * human_current[i][pair[0]][5];
            Vobs = std::vector<double>{ speed_obstacle_x,speed_obstacle_y,speed_obstacle_z }; //{speed_x_childJoint,speed_y_childJoint,speed_z_childJoint}
            tempState.speed_obs = length(Vobs);
            tempState.Vobs = Vobs;
            //point_human,point_robot
            tempState.point_human = point_human;
            tempState.point_robot = point_robot;
            //theta_rob_Vobs
            std::vector<double> vec_rob2human = subtractVec(point_human, point_robot);
            double norm_rob2human = length(vec_rob2human);
            //mat_dmin
            cv::Mat mat_human2robot = (cv::Mat_<double>(6, 1) << (-vec_rob2human[0] / norm_rob2human), (-vec_rob2human[1] / norm_rob2human), (-vec_rob2human[2] / norm_rob2human), 0, 0, 0);
            tempState.mat_dmin = mat_human2robot;
            double norm_Vobs = length(Vobs);
            if (norm_rob2human > epsilon and norm_Vobs > epsilon) //both are not 0
                tempState.theta_rob_Vobs = std::abs(std::acos(dot(vec_rob2human, Vobs) / (norm_rob2human * norm_Vobs)));
            else //either is 0 -> adopt 0
                tempState.theta_rob_Vobs = 0;
            //change here for each robot link; elbow: state_current[i][0][count].Vobs, wrist: state_current[i][1][count].Vobs,end-effector:state_current[i][2][count].Vobs 
            if (method_joints == 0)//consider significant joint
                state_current[i][idx_joint][count] = tempState;
            else if (method_joints == 1)//consider all joints
                state_current[i][idx_joint][count] = tempState;
            //update iterator, count
            count++;
        }
    }
    /** end calculating basic info **/

    /** calculate repulsive motion **/
    // get index_minimum and minimum distance
    if (index_minimum[0] >= 0) //repulsive motion is valid
    {
        CollisionState minLink = state_current[index_minimum[0]][0][index_minimum[1]];
        double speed_obs = minLink.speed_obs;
        std::vector<double> vec_obs = minLink.Vobs;
        double rho0_repulsive = rho_setting(speed_obs,vec_obs,minLink.point_robot, minLink.point_human);
        //tangent field -> return velocity, (vx,vy,vz,0,0,0)
        double dmin = minLink.dmin;
        if (dmin <= rho_tangent) {//valid tangential field
            cv::Mat v_tangent = tangent_notEE(minLink.point_robot, minLink.point_human, vec_obs, dmin,idx_joint);
            //potential gain
            double force_rep = k * (1 / dmin - 1 / rho_tangent) / dmin / dmin;
            if (minLink.dmin <= rho0_repulsive) {//repulsive field is valid
                cv::Mat mat_dmin = minLink.mat_dmin;
                if (method_joints == 0) {//consider only significant joints
                    if (idx_joint == 0) {
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian02(joints_robot);
                        cv::Mat J = ur_ivpf.J02;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = JT * force_rep * mat_dmin; //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                    else if (idx_joint == 1)
                    {
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian03(joints_robot);
                        cv::Mat J = ur_ivpf.J03;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = JT * force_rep * mat_dmin; //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                }
                else if (method_joints == 1) {//consider all joints
                    if (idx_joint == 0) {//shoulder
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian01(joints_robot);
                        cv::Mat J = ur_ivpf.J01;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = JT * force_rep * mat_dmin; //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                    else if (idx_joint == 1)//elbow
                    {
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian02(joints_robot);
                        cv::Mat J = ur_ivpf.J02;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = JT * force_rep * mat_dmin; //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                    else if (idx_joint == 2)//wrist1
                    {
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian03(joints_robot);
                        cv::Mat J = ur_ivpf.J03;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = JT * force_rep * mat_dmin; //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                    else if (idx_joint == 3)//wrist2
                    {
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian04(joints_robot);
                        cv::Mat J = ur_ivpf.J04;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = JT * force_rep * mat_dmin; //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                }
            }
            else {//no valid repulsive field
                if (method_joints == 0) {//consider only significant joints
                    if (idx_joint == 0) {
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian02(joints_robot);
                        cv::Mat J = ur_ivpf.J02;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = cv::Mat::zeros(2, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                    else if (idx_joint == 1)
                    {
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian03(joints_robot);
                        cv::Mat J = ur_ivpf.J03;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = cv::Mat::zeros(3, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                }
                else if (method_joints == 1) {//consider all joints
                    if (idx_joint == 0) {//shoulder
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian01(joints_robot);
                        cv::Mat J = ur_ivpf.J01;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = cv::Mat::zeros(1, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                    else if (idx_joint == 1)//elbow
                    {
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian02(joints_robot);
                        cv::Mat J = ur_ivpf.J02;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = cv::Mat::zeros(2, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                    else if (idx_joint == 2)//wrist1
                    {
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian03(joints_robot);
                        cv::Mat J = ur_ivpf.J03;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = cv::Mat::zeros(3, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                    else if (idx_joint == 3)//wrist2
                    {
                        //convert into joint-space velocity
                        ur_ivpf.Jacobian04(joints_robot);
                        cv::Mat J = ur_ivpf.J04;
                        cv::Mat JT;
                        cv::transpose(J, JT); //J_elbow : {6,2},invJ_elbow (2,6)*(6,2)*(2,6) = (2,6)
                        q_elbow_repulsive = cv::Mat::zeros(4, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                        q_elbow_tangent = JT * force_rep * v_tangent;
                    }
                }
            }
            return dmin;
        }
        else {//no valid repulsive field
            if (method_joints == 0) {//consider only significant joints
                if (idx_joint == 0) {
                    q_elbow_repulsive = cv::Mat::zeros(2, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                    q_elbow_tangent = cv::Mat::zeros(2, 1, CV_64F);
                }
                else if (idx_joint == 1) {
                    q_wrist_repulsive = cv::Mat::zeros(3, 1, CV_64F);
                    q_wrist_tangent = cv::Mat::zeros(3, 1, CV_64F);
                }
            }
            else if (method_joints == 1) {//consider all joints
                if (idx_joint == 0) {//shoulder
                    q_shoulder_repulsive = cv::Mat::zeros(1, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                    q_shoulder_tangent = cv::Mat::zeros(1, 1, CV_64F);
                }
                else if (idx_joint == 1) {//elbow
                    q_elbow_repulsive = cv::Mat::zeros(2, 1, CV_64F);
                    q_elbow_tangent = cv::Mat::zeros(2, 1, CV_64F);
                }
                else if (idx_joint == 2) {
                    q_wrist1_repulsive = cv::Mat::zeros(3, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                    q_wrist1_tangent = cv::Mat::zeros(3, 1, CV_64F);
                }
                else if (idx_joint == 3) {
                    q_wrist2_repulsive = cv::Mat::zeros(4, 1, CV_64F);
                    q_wrist2_tangent = cv::Mat::zeros(4, 1, CV_64F);
                }
            }

            return (double)(-1);
        }
    }
    else {//no valid repulsive field -> reset q_elbow_repulsive
        if (method_joints == 0) {//consider only significant joints
            if (idx_joint == 0) {
                q_elbow_repulsive = cv::Mat::zeros(2, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                q_elbow_tangent = cv::Mat::zeros(2, 1, CV_64F);
            }
            else if (idx_joint == 1) {
                q_wrist_repulsive = cv::Mat::zeros(3, 1, CV_64F);
                q_wrist_tangent = cv::Mat::zeros(3, 1, CV_64F);
            }
        }
        else if (method_joints == 1) {//consider all joints
            if (idx_joint == 0) {//shoulder
                q_shoulder_repulsive = cv::Mat::zeros(1, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                q_shoulder_tangent = cv::Mat::zeros(1, 1, CV_64F);
            }
            else if (idx_joint == 1) {//elbow
                q_elbow_repulsive = cv::Mat::zeros(2, 1, CV_64F);
                q_elbow_tangent = cv::Mat::zeros(2, 1, CV_64F);
            }
            else if (idx_joint == 2) {
                q_wrist1_repulsive = cv::Mat::zeros(3, 1, CV_64F); //convert from cartesian-space to joint-space :: q :(2,1)=(2,6)*(6,1)
                q_wrist1_tangent = cv::Mat::zeros(3, 1, CV_64F);
            }
            else if (idx_joint == 3) {
                q_wrist2_repulsive = cv::Mat::zeros(4, 1, CV_64F);
                q_wrist2_tangent = cv::Mat::zeros(4, 1, CV_64F);
            }
        }
        return (double)(-1);
    }
    //update previous state
    //for (int i = 0; i < state_current.size(); i++) //for each human
    //    for (int j = 0; j < state_current.size(); j++) //for each pair
    //        state_previous[i][0][j] = state_current[i][0][j];
}

//repulsive field for elbow
double IVPF::repulsive_ee(std::vector<double>& target, std::vector<double>& joints_robot) {
    /**
    * @brief calculate repulsive_field for elbow
    * @param[in] target : target 3D positino {x,y,z}
    * @param[in] pose_human : (num of human, num of joints, position{x,y,z})
    * @param[in] joints_robot : joints angle of robot {base,shoulder,elbow,wrist1,wrist2,wrist3}
    * @return dmin : minimum distance between robot and obstacles. if -1, no repulsive motion, else if positive, repulsive motion is necessary
    */

    //calculate all the joints position of robot 
    //ur_ivpf.cal_poseAll(joints_robot);
    //robot_current = std::vector<std::vector<double>>{ ur_ivpf.pose1,ur_ivpf.pose2,ur_ivpf.pose3,ur_ivpf.pose4,ur_ivpf.pose5,ur_ivpf.pose6 };

    //calculate minimum distance between shoulder-elbow line and human joints link
    double dist_minimum = rho_02_repulsive;

    std::vector<int> index_minimum{ -1,-1 };
    for (int i = 0; i < human_current.size(); i++) { //for each human

        /**  geometric relationship between robot link and human one  **/
        int count = 0;
        for (std::vector<int> pair : pair_joints) { //for each joint pair
            std::vector<double> point_human, point_robot;
            //for saving collisionState storage
            CollisionState tempState;
            //calculate minimum distance -> difference for each link elbow->(robot_current[1], robot_current[2]), wrist->(robot_current[2], robot_current[3]), end-effector->(robot_current[3], robot_current[5])
            double dmin = minDist_ivpf.minDist_2lines(human_current[i][pair[0]], human_current[i][pair[1]], robot_current[3], robot_current[5], point_human, point_robot); //shoulder-elbow
            //dmin
            tempState.dmin = dmin;
            if (dmin <= dist_minimum) { //minimum distance -> save dmin and index
                dist_minimum = dmin;
                index_minimum = std::vector<int>{ i,count }; //idx_human, joint
            }
            //calculate Vobs, theta_rob_Vobs,theta_rob_target
            //Vobs, speed_obs
            std::vector<double> Vobs;
            double dist_point_human = std::pow(std::pow(point_human[0] - human_current[i][pair[0]][0], 2) + std::pow(point_human[1] - human_current[i][pair[0]][1], 2) + std::pow(point_human[2] - human_current[i][pair[0]][2], 2), 0.5);
            double dist_link_human = std::pow(std::pow(human_current[i][pair[1]][0] - human_current[i][pair[0]][0], 2) + std::pow(human_current[i][pair[1]][1] - human_current[i][pair[0]][1], 2) + std::pow(human_current[i][pair[1]][2] - human_current[i][pair[0]][2], 2), 0.5);
            double speed_obstacle_x = (dist_point_human / dist_link_human) * human_current[i][pair[1]][3] + (1 - dist_point_human / dist_link_human) * human_current[i][pair[0]][3];
            double speed_obstacle_y = (dist_point_human / dist_link_human) * human_current[i][pair[1]][4] + (1 - dist_point_human / dist_link_human) * human_current[i][pair[0]][4];
            double speed_obstacle_z = (dist_point_human / dist_link_human) * human_current[i][pair[1]][5] + (1 - dist_point_human / dist_link_human) * human_current[i][pair[0]][5];
            Vobs = std::vector<double>{ speed_obstacle_x,speed_obstacle_y,speed_obstacle_z }; //{speed_x_childJoint,speed_y_childJoint,speed_z_childJoint}
            tempState.speed_obs = length(Vobs);
            tempState.Vobs = Vobs;
            //point_human,point_robot
            tempState.point_human = point_human;
            tempState.point_robot = point_robot;
            //theta_rob_Vobs
            std::vector<double> vec_rob2human = subtractVec(point_human, point_robot);
            double norm_rob2human = length(vec_rob2human);
            double norm_Vobs = length(Vobs);
            //mat_dmin
            cv::Mat mat_human2robot = (cv::Mat_<double>(6, 1) << (-vec_rob2human[0] / norm_rob2human), (-vec_rob2human[1] / norm_rob2human), (-vec_rob2human[2] / norm_rob2human), 0, 0, 0);
            tempState.mat_dmin = mat_human2robot;
            if (norm_rob2human > epsilon and norm_Vobs > epsilon) //both are not 0
                tempState.theta_rob_Vobs = std::abs(std::acos(dot(vec_rob2human, Vobs) / (norm_rob2human * norm_Vobs)));
            else //either is 0 -> adopt 0
                tempState.theta_rob_Vobs = 0;
            //theta_rob_target
            //contact point
            std::vector<double> p_contact, vec_rob2contact;
            if (dmin > radius) {//robot is sufficient far from human
                vec_rob2contact = std::vector<double>{ vec_rob2human[0] * (dmin - radius) / dmin,vec_rob2human[1] * (dmin - radius) / dmin,vec_rob2human[2] * (dmin - radius) / dmin };
                p_contact = addVec(point_robot, vec_rob2contact);
            }
            else {//robot is too close -> have to go away instantaneously
                vec_rob2contact = std::vector<double>{ 0,0,0 };
                p_contact = point_robot;
            }
            std::vector<double> vec_contact2robot = multiplyVec(-1, vec_rob2contact);
            std::vector<double> vec_contact2target = subtractVec(target, p_contact);
            double norm_contact2robot = length(vec_contact2robot);
            double norm_contact2target = length(vec_contact2target);
            if (norm_contact2robot > epsilon and norm_contact2target > epsilon) //both are not 0
            {
                tempState.theta_rob_target = std::abs(std::acos(dot(vec_contact2robot, vec_contact2target) / (norm_contact2robot * norm_contact2target)));
            }
            else if (norm_contact2target < epsilon) //already target position
            {
                tempState.theta_rob_target = 0.0;
            }
            else //collision -> have to do repulsive motion
            {
                tempState.theta_rob_target = 3.1415926; //on the other side
            }
            //change here for each robot link; elbow: state_current[i][0][count].Vobs, wrist: state_current[i][1][count].Vobs,end-effector:state_current[i][2][count].Vobs 
            if (method_joints == 0)//consider only significant joints
                state_current[i][2][count] = tempState;
            else if (method_joints == 1)//consider all joints
                state_current[i][4][count] = tempState;
            //update iterator, count
            count++;
        }
    }
    /** end calculating basic info **/

    /** calculate repulsive motion **/
    // get index_minimum and minimum distance
    if (index_minimum[0] >= 0) //repulsive motion is valid
    {
        CollisionState minLink;
        if (method_joints==0)//consider only significant joints 
            minLink = state_current[index_minimum[0]][2][index_minimum[1]]; //index_minimum: {num_human, n_pair}. state_current[index_minimum[0]][option][index_minimum[1]]. option:: elbow=0, wist=1,ee=2
        else if (method_joints==1)
            minLink = state_current[index_minimum[0]][4][index_minimum[1]]; //index_minimum: {num_human, n_pair}. state_current[index_minimum[0]][option][index_minimum[1]]. option:: elbow=0, wist=1,ee=2
        double speed_obs = minLink.speed_obs;
        std::vector<double> vec_obs = minLink.Vobs;

        double rho0_repulsive = rho_setting(speed_obs, vec_obs,minLink.point_robot, minLink.point_human);
        //tangent field -> return velocity, (vx,vy,vz,0,0,0)
        if (minLink.dmin <= rho_tangent) {//valid tangential field
            cv::Mat v_tangent = tangentEE(minLink.point_robot, minLink.point_human, vec_obs, target, minLink.dmin, rho0_repulsive);
            //potential gain
            double dmin = minLink.dmin;
            double force_rep = k * (1 / dmin - 1 / rho_tangent) / dmin / dmin;
            //convert end-effector velocity into joint-space
            ur_ivpf.Jacobian(joints_robot);
            cv::Mat J = ur_ivpf.J;
            cv::Mat JT;
            cv::transpose(J, JT);  // Compute J^T
            q_ee_tangent = JT *force_rep* v_tangent;//{JtJ+lambda*I}^(-1)Jt*dx_d/dt+KJt*deltaX;
            if (minLink.dmin <= rho0_repulsive) {//repulsive field is valid
                cv::Mat mat_dmin = minLink.mat_dmin;
                q_ee_repulsive = JT*force_rep*mat_dmin; //(6,1)
            }
            else 
                q_ee_repulsive = cv::Mat::zeros(6, 1, CV_64F);
            return dmin;
        }
        else {
            q_ee_tangent = cv::Mat::zeros(6, 1, CV_64F);
            q_ee_repulsive = cv::Mat::zeros(6, 1, CV_64F);
            return (double)(-1);
        }
    }
    else {//no valid repulsive field -> reset q_elbow_repulsive
        q_ee_tangent = cv::Mat::zeros(6, 1, CV_64F);
        q_ee_repulsive = cv::Mat::zeros(6, 1, CV_64F);
        return (double)(-1);
    }
    //update previous state
    //for (int i = 0; i < state_current.size(); i++) //for each human
    //    for (int j = 0; j < state_current.size(); j++) //for each pair
    //        state_previous[i][2][j] = state_current[i][2][j];
}

double IVPF::rho_setting(double& Vobs, std::vector<double>& vec_obs,std::vector<double>& point_robot, std::vector<double>& point_human) {
    /**
    * @biref determine valid rarea for epulsive field
    * @param[in] Vobs obstacle speed
    * @param[in] vec_obs -> obstacle velocity
    * @param[in] point_robot closest point in robot. (x,y,z)
    * @param[in] point_human closest point in robot. (x,y,z)
    * @return radius of valid repulsive field
    */
    //robot to barrier(obstacle)
    std::vector<double> vec_b2r = std::vector<double>{ point_robot[0]-point_human[0],point_robot[1]-point_human[1],point_robot[2]-point_human[2] };
    double dot_B2RandVobs = dot(vec_b2r, vec_obs) / length(vec_b2r) / length(vec_obs);
    if (dot_B2RandVobs >= 0) {//move toward robots
        if (Vobs > Vobs_max)
            return rho_02_repulsive;
        else if (0 < Vobs and Vobs <= Vobs_max)
            return rho_01_repulsive + (rho_02_repulsive - rho_01_repulsive) * Vobs / Vobs_max;
        else
            return rho_01_repulsive;
    }
    else {//move away from robot
        return rho_01_repulsive;
    }
    
}

cv::Mat IVPF::tangentEE(std::vector<double>& point_robot,std::vector<double>& point_human,std::vector<double>& vec_obs,
    std::vector<double>& pose_target, double& dmin, double& rho_rep) {
    /**
    * @brief calculate tangential velocity
    * @param[in] point_robot closest point in robot. (x,y,z)
    * @param[in] point_human closest point in robot. (x,y,z)
    * @param[in] vec_obs -> obstacle velocity
    * @param[in] pose_target target pose. {px,py,pz,nx,ny,nz} -> target position
    * @param[in] dmin minimum distance
    * @param[in] rho_repulsive radius of repulsive field. -> repulsive circle, r_tan
    * @return tangential unit vector. (vx,vy,vz,0,0,0)
    */

    cv::Mat v_tangent = cv::Mat::zeros(6, 1, CV_64F);
    double speed_obs = length(vec_obs);//obstacle speed
    if (dmin > rho_rep) {//minimum distance is over radius of repulsive field. -> calculate tangential direction
        //calculate 2 tangential points
        // @param[in] k1 |vec_a|^2, k2 dot(vec_a, vec_b), k3 | vec_b | ^ 2, rtan radius of circle, return {n_solutions, { s,t }} coefficients
        //robot to barrier(obstacle)
        std::vector<double> vec_r2b = std::vector<double>{ point_human[0] - point_robot[0],point_human[1] - point_robot[1],point_human[2] - point_robot[2] };
        double k1 = dot(vec_r2b, vec_r2b);
        //robot to goal
        std::vector<double> vec_r2g = std::vector<double>{ pose_target[0] - point_robot[0],pose_target[1] - point_robot[1],pose_target[2] - point_robot[2] };
        double k2 = dot(vec_r2b, vec_r2g);
        double k3 = dot(vec_r2g, vec_r2g);
        bool bool_parallel = false;
        if (1 - k2 / std::pow(k1, 0.5) / std::pow(k3, 0.5) <= epsilon) {//2 vectors are parallel to each other
            bool_parallel = true;
            if (vec_r2b[0] > epsilon and vec_r2b[1] > epsilon and vec_r2b[2] > epsilon)
                vec_r2g = std::vector<double>{ 1.0 / vec_r2b[0],1.0 / vec_r2b[1],-2.0 / vec_r2b[2] };
            else
                vec_r2g[0] = vec_r2g[0] + 0.5;
        }
        //calculate 2 tangential points. vec_r2tangent = s*vec_r2b+t*vec_r2g;
        std::vector<std::vector<double>> coeffs = tv_ivpf.main(k1, k2, k3, rho_rep);
        if (coeffs.size() == 2) {//2 solutions are found
            std::vector<double> coeff1 = coeffs[0]; std::vector<double> coeff2 = coeffs[1];//{s,t}
            //vec_RT = s*vec_RB+t*vec_RG;
            std::vector<double> vec_r2T1 = std::vector<double>{ vec_r2b[0] * coeff1[0] + vec_r2g[0] * coeff1[1],vec_r2b[1] * coeff1[0] + vec_r2g[1] * coeff1[1],vec_r2b[2] * coeff1[0] + vec_r2g[2] * coeff1[1] };
            std::vector<double> vec_r2T2 = std::vector<double>{ vec_r2b[0] * coeff2[0] + vec_r2g[0] * coeff2[1],vec_r2b[1] * coeff2[0] + vec_r2g[1] * coeff2[1],vec_r2b[2] * coeff2[0] + vec_r2g[2] * coeff2[1] };
            //positional relationship ob tangential vector and goal position
            double dot_TandObs = std::abs(dot(vec_r2T1, vec_r2b) / (length(vec_r2T1) * length(vec_r2b)));
            double dot_GandObs = std::abs(dot(vec_r2g, vec_r2b) / (length(vec_r2g) * length(vec_r2b)));
            
            if (dot_TandObs <= dot_GandObs or bool_parallel) {//goal position is within tangential vector
                if (speed_obs < Vref) {//obstacle is static
                    double dot_T1andGoal = dot(vec_r2g, vec_r2T1) / (length(vec_r2g) * length(vec_r2T1));
                    double dot_T2andGoal = dot(vec_r2g, vec_r2T2) / (length(vec_r2g) * length(vec_r2T2));
                    if (dot_T1andGoal >= dot_T2andGoal) {//T1 is more likely to be toward goal
                        double norm_r2T1 = length(vec_r2T1);
                        for (int i = 0; i < 3; i++)
                            v_tangent.at<double>(i) = vec_r2T1[i] / norm_r2T1;
                    }
                    else {//T2 is more likely to be toward goal
                        double norm_r2T2 = length(vec_r2T2);
                        for (int i = 0; i < 3; i++)
                            v_tangent.at<double>(i) = vec_r2T2[i] / norm_r2T2;
                    }
                    return v_tangent;
                }
                else {//obstacle moves dynamically
                    double dot_T1andVobs = dot(vec_obs, vec_r2T1) / (length(vec_obs) * length(vec_r2T1));
                    double dot_T2andVobs = dot(vec_obs, vec_r2T2) / (length(vec_obs) * length(vec_r2T2));
                    if (dot_T1andVobs >= dot_T2andVobs) {//T2 is more likely to be in the other way of obstacle
                        double norm_r2T2 = length(vec_r2T2);
                        for (int i = 0; i < 3; i++)
                            v_tangent.at<double>(i) = vec_r2T2[i] / norm_r2T2;
                    }
                    else {//T1 is more likely to be in the other way of obstacle
                        double norm_r2T1 = length(vec_r2T1);
                        for (int i = 0; i < 3; i++)
                            v_tangent.at<double>(i) = vec_r2T1[i] / norm_r2T1;
                    }
                    return v_tangent;
                }
            }
            else {//goal position is out of tangential vector
                if (speed_obs > Vref) {//obstacle is moving dynamically
                    //barrier to goal
                    std::vector<double> vec_B2G = std::vector<double>{ pose_target[0] - point_human[0],pose_target[1] - point_human[1],pose_target[2] - point_human[2] };
                    std::vector<double> vec_B2R = std::vector<double>{ point_robot[0] - point_human[0],point_robot[1] - point_human[1],point_robot[2] - point_human[2] };
                    double dot_VobsandGoal = dot(vec_obs, vec_B2G) / length(vec_obs) / length(vec_B2G);
                    double dot_GoalandRobot = dot(vec_B2R, vec_B2G) / length(vec_B2R) / length(vec_B2G);
                    double dot_VobsandRobot = dot(vec_obs, vec_B2R) / length(vec_obs) / length(vec_B2R);
                    if (dot_VobsandGoal >= dot_GoalandRobot and dot_VobsandRobot >= dot_GoalandRobot) {//vector of Vobs is between Robot and goal
                        double dot_T1andVobs = dot(vec_obs, vec_r2T1) / (length(vec_obs) * length(vec_r2T1));
                        double dot_T2andVobs = dot(vec_obs, vec_r2T2) / (length(vec_obs) * length(vec_r2T2));
                        if (dot_T1andVobs >= dot_T2andVobs) {//T2 is more likely to be in the other way of obstacle
                            double norm_r2T2 = length(vec_r2T2);
                            for (int i = 0; i < 3; i++)
                                v_tangent.at<double>(i) = vec_r2T2[i] / norm_r2T2;
                        }
                        else {//T1 is more likely to be in the other way of obstacle
                            double norm_r2T1 = length(vec_r2T1);
                            for (int i = 0; i < 3; i++)
                                v_tangent.at<double>(i) = vec_r2T1[i] / norm_r2T1;
                        }
                    }
                    //else
                    //vector of Vobs is in other area of Robot and goal 
                    //no tangential velocity
                }
                //else {//obstacle is static
                //    //no tangential velocity
                //}
                return v_tangent;
            }
        }
        else if (coeffs.size() == 1) {//only one solution found -> x = -b/a -> point_robot is a tangential point -> calculate tangential vector 
            if (speed_obs < Vref) {//obstacle is static -> goal-based direction
                //obstacle to robot
                std::vector<double> vec_b2r = std::vector<double>{ point_robot[0]-point_human[0],point_robot[1]-point_human[1],point_robot[2]-point_human[2] };
                //obstacle to goal
                std::vector<double> vec_b2g = std::vector<double>{ pose_target[0] - point_human[0],pose_target[1] - point_human[1],pose_target[2] - point_human[2] };
                double k = dot(vec_b2g, vec_b2r) / length(vec_b2r) / length(vec_b2r);
                std::vector<double> vec_b2h = multiplyVec(k, vec_b2r);
                std::vector<double> vec_h2g = subtractVec(vec_b2g, vec_b2h);
                double norm_h2g = length(vec_h2g);
                if (norm_h2g > epsilon) {
                    for (int i = 0; i < 3; i++) {
                        v_tangent.at<double>(i) = vec_h2g[i]/norm_h2g;
                    }
                }
                return v_tangent;
            }
            else {//obstacle is dynamic -> Vobs-based direction
                //obstacle to robot
                std::vector<double> vec_b2r = std::vector<double>{ point_robot[0] - point_human[0],point_robot[1] - point_human[1],point_robot[2] - point_human[2] };
                double k = dot(vec_obs, vec_b2r) / length(vec_b2r) / length(vec_b2r);
                std::vector<double> vec_b2h = multiplyVec(k, vec_b2r);
                std::vector<double> vec_obs2h = subtractVec(vec_b2h,vec_obs);
                double norm_obs2h = length(vec_obs2h);
                if (norm_obs2h > epsilon) {
                    for (int i = 0; i < 3; i++) {
                        v_tangent.at<double>(i) = vec_obs2h[i] / norm_obs2h;
                    }
                }
                return v_tangent;
            }
        }
        //if there is no solution, nothing to do
    }
    else {//minimum distance is under radius of repulsive field. -> calculate vector
        if (speed_obs < Vref) {//obstacle is static -> goal-based direction
            //obstacle to robot
            std::vector<double> vec_b2r = std::vector<double>{ point_robot[0] - point_human[0],point_robot[1] - point_human[1],point_robot[2] - point_human[2] };
            //obstacle to goal
            std::vector<double> vec_b2g = std::vector<double>{ pose_target[0] - point_human[0],pose_target[1] - point_human[1],pose_target[2] - point_human[2] };
            double k = dot(vec_b2g, vec_b2r) / length(vec_b2r) / length(vec_b2r);
            std::vector<double> vec_b2h = multiplyVec(k, vec_b2r);
            std::vector<double> vec_h2g = subtractVec(vec_b2g, vec_b2h);
            double norm_h2g = length(vec_h2g);
            if (norm_h2g > epsilon) {
                for (int i = 0; i < 3; i++) {
                    v_tangent.at<double>(i) = vec_h2g[i] / norm_h2g;
                }
            }
            return v_tangent;
        }
        else {//obstacle is dynamic -> Vobs-based direction
            //obstacle to robot
            std::vector<double> vec_b2r = std::vector<double>{ point_robot[0] - point_human[0],point_robot[1] - point_human[1],point_robot[2] - point_human[2] };
            double k = dot(vec_obs, vec_b2r) / length(vec_b2r) / length(vec_b2r);
            std::vector<double> vec_b2h = multiplyVec(k, vec_b2r);
            std::vector<double> vec_obs2h = subtractVec(vec_b2h, vec_obs);
            double norm_obs2h = length(vec_obs2h);
            if (norm_obs2h > epsilon) {
                for (int i = 0; i < 3; i++) {
                    v_tangent.at<double>(i) = vec_obs2h[i] / norm_obs2h;
                }
            }
            return v_tangent;
        }
    }
}

cv::Mat IVPF::tangent_notEE(std::vector<double>& point_robot, std::vector<double>& point_human, std::vector<double>& vec_obs,double& dmin, int& idx_joint) {
    /**
    * @brief calculate tangential velocity for other than end-effector. consider only the obstacle velocity.
    * @param[in] point_robot closest point in robot. (x,y,z)
    * @param[in] point_human closest point in robot. (x,y,z)
    * @param[in] vec_obs -> obstacle velocity
    * @param[in] dmin minimum distance
    * @param[in] index_joint: index of joint. if
    * @return tangential unit vector. (vx,vy,vz,0,0,0)
    */

    cv::Mat v_tangent = cv::Mat::zeros(6, 1, CV_64F);
    double speed_obs = length(vec_obs);//obstacle speed
    
    if (speed_obs >= Vref) {//obstacle is moving dynamically -> Vobs-based direction
        //obstacle to robot
        std::vector<double> vec_b2r = std::vector<double>{ point_robot[0] - point_human[0],point_robot[1] - point_human[1],point_robot[2] - point_human[2] };
        double k = dot(vec_obs, vec_b2r) / length(vec_b2r) / length(vec_b2r);
        std::vector<double> vec_b2h = multiplyVec(k, vec_b2r);
        std::vector<double> vec_obs2h = subtractVec(vec_b2h, vec_obs);
        double norm_obs2h = length(vec_obs2h);
        if (norm_obs2h > epsilon) {
            for (int i = 0; i < 3; i++) {
                v_tangent.at<double>(i) = vec_obs2h[i] / norm_obs2h;
            }
        }
    }
    return v_tangent;
}

void IVPF::invJacobian(cv::Mat& J, cv::Mat& invJ) {
    /**
    * @brief calcuate Jacobian Matrix
    * @param[in] J: jacobian matrix
    * @param[out] invJ : inverse Jacobian matrix
    */
    if (bool_avoidSingularity) {//avoid singularity
        double c = ur_ivpf.determinant(J);
        double c_function = funcC(c);
        double angle_tan = std::min((pi / 2 - epsilon), (d_threshold + c_function));//clip by pi/2
        //double angle_tan = std::min((pi / 2 - epsilon), (d_threshold + (pi/2.0-d_threshold)*Kp_det * c));
        //double lambda = k_const / (std::tan(angle_tan) + k_const);
        double lambda = 1.0 / tan(angle_tan);//c1_max * tan(d_threshold)
        //std::cout << "lambda=" << lambda << std::endl;
        cv::Mat JT;
        cv::transpose(J, JT);  // Compute J^T
        cv::Mat JtJ = JT * J;  // Compute J * J^T
        int rows = JtJ.rows;
        cv::Mat matrix_avoidSingularity = cv::Mat::eye(rows, rows, CV_64F);
        matrix_avoidSingularity = lambda * matrix_avoidSingularity;
        cv::Mat A = JtJ + matrix_avoidSingularity;
        cv::invert(A, invJ);
        invJ = invJ * JT;
    }
    else {
        cv::Mat JT;
        cv::transpose(J, JT);  // Compute J^T
        cv::Mat JtJ = JT * J;  // Compute J * J^T
        cv::Mat A = JtJ;
        cv::Mat A_inv;
        cv::invert(A, A_inv);
        invJ = A_inv * JT;//pseud inverse matrix

    }


}

double IVPF::funcC(double& c) {
    /**
    * @brief adjust determinant, c, for deviating from singular points
    * @param[in] c determinang (det(JJT))^(0.5)
    */

    double point_mid = ((pi / 2.0) - d_threshold) / 2.0;//middle point. if d_threshold=pi/6, point_mid = pi/6 and end_point=pi/3
    if (c1_max < c)
        return ((pi / 2.0) - d_threshold);
    else if (c1_min <= c and c <= c1_max) {
        double angle_sine = (pi / 2.0) * ((c - c1_min) / (c1_max - c1_min));
        double val_return = point_mid * (sin(angle_sine) + 1);
        return val_return;
    }
    else if (c2_min <= c and c < c2_max) {
        double angle_sine = (pi / 2.0) * (1.0 - (c - c2_min) / (c2_max - c2_min));
        double val_return = point_mid * (1.0 - sin(angle_sine));
        return val_return;
    }
    else if (c < c2_min)
        return 0.0;
}

double IVPF::dot(std::vector<double>& vec1, std::vector<double>& vec2) {
    /**
    * @brief inner dot
    */
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

double IVPF::length(std::vector<double>& vector) {
    /**
    * @brief normalize vector
    * @brief vector: unit vector, norm : length
    */
    return std::pow(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2], 0.5); //calculate norm
}

std::vector<double> IVPF::subtractVec(std::vector<double>& vec1, std::vector<double>& vec2) {
    /**
    * @brief vec1 - vec2
    */
    std::vector<double> diff;
    int length1 = vec1.size();
    int length2 = vec2.size();
    int length;
    if (length1 <= length2) length = length1;
    else length = length2;
    for (int i = 0; i < length; i++)
        diff.push_back(vec1[i] - vec2[i]);
    return diff;
}

std::vector<double> IVPF::addVec(std::vector<double>& vec1, std::vector<double>& vec2) {
    /**
    * @brief vec1 + vec2
    */
    std::vector<double> diff;
    int length1 = vec1.size();
    int length2 = vec2.size();
    int length;
    if (length1 <= length2) length = length1;
    else length = length2;
    for (int i = 0; i < length; i++)
        diff.push_back(vec1[i] + vec2[i]);
    return diff;
}

std::vector<double> IVPF::multiplyVec(double a, std::vector<double>& vec) {
    /**
    * @brief  a*vec2
    */
    std::vector<double> diff;
    for (int i = 0; i < vec.size(); i++)
        diff.push_back(a * vec[i]);
    return diff;
}

std::vector<double> IVPF::divVec(double a, std::vector<double>& vec) {
    /**
    * @brief  a*vec2
    */
    std::vector<double> diff;
    for (int i = 0; i < vec.size(); i++)
        diff.push_back(vec[i] / a);
    return diff;
}