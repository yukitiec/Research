#include "ur.h"

void UR::mat01(double& theta1) {
    double c1 = cos(theta1);
    double s1 = sin(theta1);
    T01 = (cv::Mat_<double>(4, 4) <<
        c1, 0.0, s1, 0.0,
        s1, 0.0, -c1, 0,
        0.0, 1.0, 0.0, d1,
        0.0, 0.0, 0.0, 1);
}

void UR::mat12(double& theta2) {


    double c2 = cos(theta2);
    double s2 = sin(theta2);
    T12 = (cv::Mat_<double>(4, 4) <<
        c2, -s2, 0.0, a2*c2,
        s2, c2, 0.0, a2*s2,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1);
}

void UR::mat23(double& theta3) {
    double c3 = cos(theta3);
    double s3 = sin(theta3);
    T23 = (cv::Mat_<double>(4, 4) <<
        c3, -s3, 0.0, a3 * c3,
        s3, c3, 0.0, a3 * s3,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1);
}

void UR::mat34(double& theta4) {
    double c4 = cos(theta4);
    double s4 = sin(theta4);
    T34 = (cv::Mat_<double>(4, 4) <<
        c4, 0.0, s4, 0.0,
        s4, 0.0, -c4, 0,
        0.0, 1.0, 0.0, d4,
        0.0, 0.0, 0.0, 1);
}

void UR::mat45(double& theta5) {
    double c5 = cos(theta5);
    double s5 = sin(theta5);
    T45 = (cv::Mat_<double>(4, 4) <<
        c5, 0.0, -s5, 0.0,
        s5, 0.0, c5, 0,
        0.0, -1.0, 0.0, d5,
        0.0, 0.0, 0.0, 1);
}

void UR::mat56(double& theta6) {
    double c6 = cos(theta6);
    double s6 = sin(theta6);
    T56 = (cv::Mat_<double>(4, 4) <<
        c6,  -s6, 0.0, 0.0,
        s6,  c6, 0, 0.0,
        0.0, 0.0, 1.0, d6,
        0.0, 0.0, 0.0, 1);
}

void UR::mat(std::vector<double>& joints) {
    /**
    * calculate all joints homogeneous matrix
    */
    mat01(joints[0]);
    mat12(joints[1]);
    mat23(joints[2]);
    mat34(joints[3]);
    mat45(joints[4]);
    mat56(joints[5]);
}

void UR::cal_pose1(std::vector<double>& joints, bool bool_mat) {
    /**
    * base pose
    */
    if (bool_mat)
        mat(joints);//T01
    cv::Mat  R1 = T01(cv::Range(0, 3), cv::Range(0, 3));//rotation matrix
    //convert rotation matrix to rotation vecctor
    cv::Mat n1; //space vector -> k
    cv::Rodrigues(R1, n1);
    double px = T01.at<double>(0, 3);
    double py = T01.at<double>(1, 3);
    double pz = T01.at<double>(2, 3);
    double nx = n1.at<double>(0);
    double ny = n1.at<double>(1);
    double nz = n1.at<double>(2);
    pose1 = std::vector<double>{px,py,pz,nx,ny,nz}; //cv::Mat a, a.at<double>(row,col)
}

void UR::cal_pose2(std::vector<double>& joints, bool bool_mat) {
    /**
    * shoulder pose
    */
    if (bool_mat)
        mat(joints);
    // Method 1: Using cv::Mat::mul()
    T02 = T01 * T12;
    //cv::multiply(T01, T12, T02); //cv::multiply is element-wise multiplication
    cv::Mat  R2 = T02(cv::Range(0, 3), cv::Range(0, 3));//rotation matrix
    //convert rotation matrix to rotation vecctor
    cv::Mat n2; //space vector -> k
    cv::Rodrigues(R2, n2);
    double px = T02.at<double>(0, 3);
    double py = T02.at<double>(1, 3);
    double pz = T02.at<double>(2, 3);
    double nx = n2.at<double>(0);
    double ny = n2.at<double>(1);
    double nz = n2.at<double>(2);
    pose2 = std::vector<double>{ px,py,pz,nx,ny,nz }; //cv::Mat a, a.at<double>(row,col)
}

void UR::cal_pose3(std::vector<double>& joints, bool bool_mat) {
    /**
    * elbow pose
    */
    if (bool_mat)
        mat(joints);
    // Method 1: Using cv::Mat::mul()
    T03 = T01*T12*T23;
    cv::Mat  R3 = T03(cv::Range(0, 3), cv::Range(0, 3));//rotation matrix
    //convert rotation matrix to rotation vecctor
    cv::Mat n3; //space vector -> k
    cv::Rodrigues(R3, n3);
    double px = T03.at<double>(0, 3);
    double py = T03.at<double>(1, 3);
    double pz = T03.at<double>(2, 3);
    double nx = n3.at<double>(0);
    double ny = n3.at<double>(1);
    double nz = n3.at<double>(2);
    pose3 = std::vector<double>{ px,py,pz,nx,ny,nz }; //cv::Mat a, a.at<double>(row,col)
}

void UR::cal_pose4(std::vector<double>& joints, bool bool_mat) {
    /**
    * wrist1
    */
    if (bool_mat)
        mat(joints);
    // Method 1: Using cv::Mat::mul()
    T04 = T01 * T12 * T23 * T34;
    cv::Mat  R4 = T04(cv::Range(0, 3), cv::Range(0, 3));//rotation matrix
    //convert rotation matrix to rotation vecctor
    cv::Mat n4; //space vector -> k
    cv::Rodrigues(R4, n4);
    double px = T04.at<double>(0, 3);
    double py = T04.at<double>(1, 3);
    double pz = T04.at<double>(2, 3);
    double nx = n4.at<double>(0);
    double ny = n4.at<double>(1);
    double nz = n4.at<double>(2);
    pose4 = std::vector<double>{ px,py,pz,nx,ny,nz }; //cv::Mat a, a.at<double>(row,col)
}

void UR::cal_pose5(std::vector<double>& joints, bool bool_mat) {
    /**
    * wrist2
    */
    if (bool_mat)
        mat(joints);
    // Method 1: Using cv::Mat::mul()
    T05 = T01*T12*T23*T34*T45;
    cv::Mat  R5 = T05(cv::Range(0, 3), cv::Range(0, 3));//rotation matrix
    //convert rotation matrix to rotation vecctor
    cv::Mat n5; //space vector -> k
    cv::Rodrigues(R5, n5);
    double px = T05.at<double>(0, 3);
    double py = T05.at<double>(1, 3);
    double pz = T05.at<double>(2, 3);
    double nx = n5.at<double>(0);
    double ny = n5.at<double>(1);
    double nz = n5.at<double>(2);
    pose5 = std::vector<double>{ px,py,pz,nx,ny,nz }; //cv::Mat a, a.at<double>(row,col)
}

void UR::cal_pose6(std::vector<double>& joints, bool bool_mat) {
    /**
    * end effector
    */
    if (bool_mat)
        mat(joints);
    // Method 1: Using cv::Mat::mul()
    T06= T01*T12*T23*T34*T45*T56;
    cv::Mat  R6 = T06(cv::Range(0, 3), cv::Range(0, 3));//rotation matrix
    //std::cout << "matrix=" << T06 << std::endl;
    //convert rotation matrix to rotation vecctor
    cv::Mat n6; //space vector -> k
    cv::Rodrigues(R6, n6);
    double px = T06.at<double>(0, 3);
    double py = T06.at<double>(1, 3);
    double pz = T06.at<double>(2, 3);
    double nx = n6.at<double>(0);
    double ny = n6.at<double>(1);
    double nz = n6.at<double>(2);
    pose6 = std::vector<double>{ px,py,pz,nx,ny,nz }; //cv::Mat a, a.at<double>(row,col)
}

void UR::cal_poseAll(std::vector<double>& joints) {
    /**
    * all joints pose
    */
    mat(joints);
    //calculate all positions
    cal_pose1(joints,false);
    cal_pose2(joints, false);
    cal_pose3(joints, false);
    cal_pose4(joints, false);
    cal_pose5(joints, false);
    cal_pose6(joints, false);
}

void UR::Jacobian01(std::vector<double>& joints) {
    /**
    * @brief calculate Jacobian of shoulder as ee (end-effector)
    */
    double s1 = sin(joints[0]);
    double c1 = cos(joints[0]);
    double px = pose6[0];
    double py = pose6[1];
    double pz = pose6[2];
    J01 = (cv::Mat_<double>(6, 1) <<
        -py, 
        px, 
        0, 
        0, 
        0, 
        1);
}

void UR::Jacobian02(std::vector<double>& joints) {
    /**
    * @brief calculate Jacobian of elbow as ee (end-effector)
    */
    double s1 = sin(joints[0]);
    double c1 = cos(joints[0]);
    double px = pose6[0];
    double py = pose6[1];
    double pz = pose6[2];
    J02 = (cv::Mat_<double>(6, 2) <<
        -py, -c1 * (pz - d1),
        px, -s1 * (pz - d1),
        0, s1 * py + c1 * px,
        0,s1,
        0, -c1,
        1, 0);
}

void UR::Jacobian03(std::vector<double>& joints) {
    /**
    * @brief calculate Jacobian of elbow as ee (end-effector)
    */
    double s1 = sin(joints[0]);
    double c1 = cos(joints[0]);
    double s23 = sin(joints[1] + joints[2]);
    double c23 = cos(joints[1] + joints[2]);
    double s234 = sin(joints[1] + joints[2] + joints[3]); 
    double c234 = cos(joints[1] + joints[2] + joints[3]);
    double s5 = sin(joints[4]);
    double c5 = cos(joints[4]);
    double px = pose6[0];
    double py = pose6[1];
    double pz = pose6[2];
    J03 = (cv::Mat_<double>(6, 3) <<
        -py, -c1 * (pz - d1), c1 * (s234 * s5 * d6 + c234 * d5 - s23 * a3),
        px, -s1 * (pz - d1), s1 * (s234 * s5 * d6 + c234 * d5 - s23 * a3),
        0, s1 * py + c1 * px, -c234 * s5 * d6 + s234 * d5 + c23 * a3,
        0, s1,s1,
        0, -c1,-c1,
        1, 0,0);
}

void UR::Jacobian04(std::vector<double>& joints) {
    /**
    * @brief calculate Jacobian of wrist1 as ee (end-effector)
    */
    double s1 = sin(joints[0]);
    double c1 = cos(joints[0]);
    double s23 = sin(joints[1] + joints[2]);
    double c23 = cos(joints[1] + joints[2]);
    double s234 = sin(joints[1] + joints[2] + joints[3]);
    double c234 = cos(joints[1] + joints[2] + joints[3]);
    double s5 = sin(joints[4]);
    double c5 = cos(joints[4]);
    double r13 = T06.at<double>(0, 2);
    double r23 = T06.at<double>(1, 2);
    double r33 = T06.at<double>(2, 2);
    double px = pose6[0];
    double py = pose6[1];
    double pz = pose6[2];
    J04 = (cv::Mat_<double>(6, 4) <<
        -py, -c1 * (pz - d1), c1 * (s234 * s5 * d6 + c234 * d5 - s23 * a3), c1 * (s234 * s5 * d6 + c234 * d5), 
        px, -s1 * (pz - d1), s1 * (s234 * s5 * d6 + c234 * d5 - s23 * a3), s1 * (s234 * s5 * d6 + c234 * d5),
        0, s1 * py + c1 * px, -c234 * s5 * d6 + s234 * d5 + c23 * a3, -c234 * s5 * d6 + s234 * d5, 
        0, s1, s1, s1, 
        0, -c1, -c1, -c1,
        1, 0, 0, 0
        );
}

void UR::Jacobian05(std::vector<double>& joints) {
    /**
    * @brief calculate Jacobian of wrist2 as ee (end-effector)
    */
    double s1 = sin(joints[0]);
    double c1 = cos(joints[0]);
    double s23 = sin(joints[1] + joints[2]);
    double c23 = cos(joints[1] + joints[2]);
    double s234 = sin(joints[1] + joints[2] + joints[3]);
    double c234 = cos(joints[1] + joints[2] + joints[3]);
    double s5 = sin(joints[4]);
    double c5 = cos(joints[4]);
    double r13 = T06.at<double>(0, 2);
    double r23 = T06.at<double>(1, 2);
    double r33 = T06.at<double>(2, 2);
    double px = pose6[0];
    double py = pose6[1];
    double pz = pose6[2];
    J05 = (cv::Mat_<double>(6, 5) <<
        -py, -c1 * (pz - d1), c1 * (s234 * s5 * d6 + c234 * d5 - s23 * a3), c1 * (s234 * s5 * d6 + c234 * d5), -d6 * (s1 * s5 + c1 * c234 * c5), 
        px, -s1 * (pz - d1), s1 * (s234 * s5 * d6 + c234 * d5 - s23 * a3), s1 * (s234 * s5 * d6 + c234 * d5), d6 * (c1 * c5 - s1 * c234 * c5), 
        0, s1 * py + c1 * px, -c234 * s5 * d6 + s234 * d5 + c23 * a3, -c234 * s5 * d6 + s234 * d5, -c5 * s234 * d6, 
        0, s1, s1, s1, c1 * s234, 
        0, -c1, -c1, -c1, s1 * s234, 
        1, 0, 0, 0, -c234
        );
}

void UR::Jacobian(std::vector<double>& joints) {
    /**
    * @brief calculate Jacobian of end-effector as ee (end-effector)
    */
    double s1 = sin(joints[0]);
    double c1 = cos(joints[0]);
    double s23 = sin(joints[1] + joints[2]);
    double c23 = cos(joints[1] + joints[2]);
    double s234 = sin(joints[1] + joints[2] + joints[3]);
    double c234 = cos(joints[1] + joints[2] + joints[3]);
    double s5 = sin(joints[4]);
    double c5 = cos(joints[4]);
    double r13 = T06.at<double>(0, 2);
    double r23 = T06.at<double>(1, 2);
    double r33 = T06.at<double>(2, 2);
    double px = pose6[0];
    double py = pose6[1];
    double pz = pose6[2];
    J = (cv::Mat_<double>(6, 6) <<
        -py, -c1 * (pz - d1), c1 * (s234 * s5 * d6 + c234 * d5 - s23 * a3), c1 * (s234 * s5 * d6 + c234 * d5), -d6 * (s1 * s5 + c1 * c234 * c5), 0,
        px, -s1 * (pz - d1), s1 * (s234 * s5 * d6 + c234 * d5 - s23 * a3), s1 * (s234 * s5 * d6 + c234 * d5), d6 * (c1 * c5 - s1 * c234 * c5), 0,
        0, s1 * py + c1 * px, -c234 * s5 * d6 + s234 * d5 + c23 * a3, -c234 * s5 * d6 + s234 * d5, -c5 * s234 * d6, 0,
        0, s1, s1,s1,c1*s234,r13,
        0, -c1, -c1,-c1,s1*s234,r23,
        1, 0, 0,0,-c234,r33
       );
}

double UR::determinant(cv::Mat& j) {
    /**
    * @brief calculate determinant (j*j^T)^(0.5)
    */
    cv::Mat jT;
    cv::transpose(j, jT);  // Compute J^T
    cv::Mat jjt = j * jT;  // Compute J * J^T
    double determinant = std::pow(cv::determinant(jjt),0.5);
    return determinant;
}