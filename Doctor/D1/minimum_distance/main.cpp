// IVPF.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "stdafx.h"
#include "ivpf2.h"
#include "ur.h"

void saveDeterminant(std::string&, std::vector<double>&);
void saveData(std::string&, std::vector<std::vector<std::vector<double>>>&);
void saveData2(std::string&, std::vector<std::vector<double>>&);

int main()
{
    std::string file_joints = "joints.csv";
    std::string file_obs_closest = "obstacles_closest.csv";
    std::string file_robot_closest = "robots_closest.csv";
    std::string file_minimumDist = "minimum_distance.csv";
    UR ur_main;
    MinimumDist minDist_main;
    //calculate all the joints position of robot 
    std::vector<std::vector<std::vector<double>>> joints;//num sequence, num joints, {px,py,pz,nx,ny.nz}}
    std::vector<double> init{0,0,0,0,0,0};
    std::vector<double> obs1{ -0.2,-0.5,-0.3 }; std::vector<double> obs2{ 0.5,1.2,1.6 };
    std::vector<std::vector<std::vector<double>>> points_obs, points_robot;//num sequence,num joints,(x,y,z)
    std::vector<std::vector<double>> dists_minimum; //sequence, num joints
    for (int i = 0; i < 120; i++) {
        //std::cout << "i=" << i << std::endl;
        if (i < 20) init[0] += 0.1;
        else if (20 <= i and i < 40) init[1] += 0.1;
        else if (40 <= i and i < 60) init[2] += 0.1;
        else if (60 <= i and i < 80) init[3] += 0.1;
        else if (80 <= i and i < 100) init[4] += 0.1;
        else if (100 <= i and i < 120) init[5] += 0.1;
        ur_main.cal_poseAll(init);
        std::vector<std::vector<double>> robot_current = std::vector<std::vector<double>>{ ur_main.pose1,ur_main.pose2,ur_main.pose3,ur_main.pose4,ur_main.pose5,ur_main.pose6 };
        joints.push_back(robot_current);
        std::vector<std::vector<double>> temp_obs, temp_robot;
        std::vector<double> point_obs, point_robot;
        double dmin1 = minDist_main.minDist_2lines(obs1, obs2, robot_current[1], robot_current[2], point_obs, point_robot); //shoulder-elbow
        temp_obs.push_back(point_obs); temp_robot.push_back(point_robot);
        double dmin2 = minDist_main.minDist_2lines(obs1, obs2, robot_current[2], robot_current[3], point_obs, point_robot); //elbow-wrist
        temp_obs.push_back(point_obs); temp_robot.push_back(point_robot);
        double dmin3 = minDist_main.minDist_2lines(obs1, obs2, robot_current[3], robot_current[5], point_obs, point_robot); //elbow-wrist
        temp_obs.push_back(point_obs); temp_robot.push_back(point_robot);
        points_obs.push_back(temp_obs);
        points_robot.push_back(temp_robot);
        dists_minimum.push_back(std::vector<double>{dmin1, dmin2, dmin3});
    }
    saveData(file_joints, joints);
    saveData(file_obs_closest, points_obs);
    saveData(file_robot_closest, points_robot);
    saveData2(file_minimumDist, dists_minimum);
    /*
    std::vector<double> o1{ 1,2,0 }; std::vector<double> o2{ 3,2,0 };
    std::vector<double> r1{ 1,1,0 }; std::vector<double> r2{ 1,5,0 };
    std::vector<double> po, pr;
    double dmin = minDist_main.minDist_2lines(o1, o2, r1, r2, po,pr); //shoulder-elbow
    std::cout << "dmin=" << dmin << std::endl;
    std::cout << "po=";
    for (int i = 0; i < po.size(); i++)
        std::cout << po[i] << ",";
    std::cout << std::endl;
    std::cout << "pr=";
    for (int i = 0; i < pr.size(); i++)
        std::cout << pr[i] << ",";
    std::cout << std::endl;
    */

    
    //ivpf
    //target pose({px,py,pz,nx,ny,nz}),human_pose{n_human(1),n_joint(6),position(px,py,pz)}
    IVPF ivpf_main(init);
    std::vector<double> pose1{ 0,0,0,0, 0, 3.1415 }; std::vector<double> pose2{ 0,0,0,3.1415, 0, 0 };
    std::vector<double> k; double angle;
    cv::Mat rotation_transform; //space vector -> k
    cv::Mat_<double> rotationMatrix(3, 3);
    rotationMatrix << -0.52960, 0.74368, 0.40801,
        0.84753, 0.44413, 0.29059,
        0.0349, 0.4997, -0.8655;
    cv::Rodrigues(rotationMatrix, rotation_transform);
    //std::cout << "rotational vector=" << rotation_transform.at<double>(0) << ", " << rotation_transform.at<double>(1) << ", " << rotation_transform.at<double>(2) << std::endl;
    std::vector<double> pose_target{0.17667,-0.20033,0.61277,rotation_transform.at<double>(0),rotation_transform.at<double>(1),rotation_transform.at<double>(2) };
    std::vector<double> joints_ivpf{ 1.53,0.576,1.32,0.533,1.49,1.27 };
    std::vector<std::vector<std::vector<double>>> pose_human{ {{-0.2,0.1,0.9},{-0.2,-0.3,0.9},{0.2,-0.1,0.7},{0.1,-0.4,0.7},{0.1,-0.2,0.5},{0.3,-0.5,0.5}} };
    std::vector<double> pose_current{ 0,0,0,0,0,0 };
    std::vector<std::vector<double>> save_jointsAngle;
    //save storage
    std::vector<std::vector<double>> save_dists_minimum; //sequence, num joints
    //robot pose
    std::vector<std::vector<std::vector<double>>> save_joints;//num sequence, num joints, {px,py,pz,nx,ny.nz}}
    //pose_human -> num_joints, seq,position
    std::vector<std::vector<std::vector<double>>> save_joints_human;//num sequence, num joints, {px,py,pz,nx,ny.nz}}
    int count_iteration = 0;
    int count_moving = 700;
    double t_elapsed = 0;
    int iteration = 0;
    while (true) {
        double dist = std::pow(((pose_current[0] - pose_target[0])* (pose_current[0] - pose_target[0]) + (pose_current[1] - pose_target[1])* (pose_current[1] - pose_target[1]) + (pose_current[2] - pose_target[2])* (pose_current[2] - pose_target[2])), 0.5);
        //std::cout << "pose_current : " << pose_current[0] << "," << pose_current[1] << "," << pose_current[2] << ", pose_target : " << pose_target[0] << ", " << pose_target[2] << ", " << pose_target[2] << std::endl;
        std::cout << "countIteration=" << count_iteration << ", distance to target=" << dist << std::endl;
        if (count_iteration >= 3000 || dist < 0.01) break; //reach target or max iteration
        auto start = std::chrono::high_resolution_clock::now();
        ivpf_main.main(pose_target, pose_human, joints_ivpf);
        auto end = std::chrono::high_resolution_clock::now();
        // Calculate the duration in microseconds
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        t_elapsed += (double)(duration.count());
        std::cout << "process time is " << duration.count() << "microseconds" << std::endl;
        iteration++;
        //robot joints pose
        save_joints.push_back(ivpf_main.robot_current);//{px,py,pz,nx,ny,nz} 6*6 (base,shoulder,elbow,w1,w2,end-effector)
        //save joints angle
        save_jointsAngle.push_back(joints_ivpf);
        pose_current = ivpf_main.robot_current[5];
        //minimum distances -> 3*5
        std::vector<double> tmp_dists;
        for (int i = 0; i < ivpf_main.state_current[0].size(); i++) {
            for (int j = 0; j < ivpf_main.state_current[0][i].size(); j++) {
                tmp_dists.push_back(ivpf_main.state_current[0][i][j].dmin);
            }
        }
        save_dists_minimum.push_back(tmp_dists);
        //human_pose
        save_joints_human.push_back(pose_human[0]);
        //update human pose
        pose_human[0][4][0] += 1 / count_moving; pose_human[0][2][0] += 1 / count_moving; pose_human[0][0][0] += 0.5 / count_moving;
        pose_human[0][5][1] += 0.2 / count_moving; pose_human[0][3][1] += 0.2 / count_moving;
        //increment iterator
        count_iteration++;
    }
    std::cout << "process speed = " << iteration / t_elapsed * 1000000 << " Hz" << std::endl;
    
    //save data into csv file
    std::string file_joints_ivpf = "C:/Users/kawaw/cpp/IVPF/analysis/joints_ivpf.csv";
    std::string file_jointsAngle_ivpf = "C:/Users/kawaw/cpp/IVPF/analysis/joints_angle.csv";
    std::string file_minimumDist_ivpf = "C:/Users/kawaw/cpp/IVPF/analysis/minimum_distance_ivpf.csv";
    std::string file_human_ivpf = "C:/Users/kawaw/cpp/IVPF/analysis/human_pose.csv";
    std::string file_determinant = "C:/Users/kawaw/cpp/IVPF/analysis/determinant.csv";
    saveDeterminant(file_determinant, ivpf_main.determinants);
    saveData(file_joints_ivpf, save_joints);
    saveData2(file_jointsAngle_ivpf, save_jointsAngle);
    saveData(file_human_ivpf, save_joints_human);
    saveData2(file_minimumDist_ivpf, save_dists_minimum);

    
    // Define the rotation vector
    ivpf_main.cal_pose_rot(pose1, pose2, k, angle);

    std::vector<double> joints2{ 1,1,1,1,1,1 };
    UR ur;
    ur.cal_pose6(joints2);
    // Print the rotation matrix
    std::cout << "k:" << std::endl;
    for (double ele : k)
        std::cout << ele << " ";
    std::cout << std::endl;
    std::cout << " Rotation angle:\n" << angle << std::endl;

    const double pi = 3.14159265358979323846;
    std::vector<double> joints3{(pi*184.05/180.0),(pi * -147.97/ 180.0),(pi * 91.87 / 180.0), (pi * -120.23 / 180.0),(pi * 98.9 / 180.0),(pi * 37.09 / 180.0) };
    ur.cal_poseAll(joints3);
    std::vector<std::vector<double>> robot_current = std::vector<std::vector<double>>{ ur.pose1,ur.pose2,ur.pose3,ur.pose4,ur.pose5,ur.pose6 };
    std::cout << "robot end-effector pose";
    for (int i = 0; i < robot_current[5].size(); i++)
        std::cout << robot_current[5][i] << ",";
    std::cout << std::endl;
    


}

void saveDeterminant(std::string& fileName, std::vector<double>& d) {
    /**
    * @brief save determinant in csv file
    * @param[in] d list of determinants
    */
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    int count = 1;
    //std::cout << "Class saver :: Contensts ::" << std::endl;
    for (int i = 0; i < d.size(); i++)//sequence
    {
        outputFile << d[i];
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}
void saveData(std::string& fileName, std::vector<std::vector<std::vector<double>>>& joints) {
    /**
    * @brief save joints data in csv file
    * @param[in] joints : num sequence, num joints, {px,py,pz,nx,ny.nz}
    */
    //save data into csv file
    // Open the file for writing
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    int count = 1;
    //std::cout << "Class saver :: Contensts ::" << std::endl;
    for (int i = 0; i < joints.size(); i++)//sequence
    {
        for (int j = 0; j < joints[i].size(); j++)//joints
        {
            for (int k = 0; k < joints[i][j].size(); k++) {
                outputFile << joints[i][j][k];
                if (j != joints[i].size() - 1 or k != joints[i][j].size()-1)
                    outputFile << ",";
                
            }
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void saveData2(std::string& fileName, std::vector<std::vector<double>>& joints) {
    /**
    * @brief save joints data in csv file
    * @param[in] joints : num sequence, num joints, {px,py,pz,nx,ny.nz}
    */
    //save data into csv file
    // Open the file for writing
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    int count = 1;
    //std::cout << "Class saver :: Contensts ::" << std::endl;
    for (int i = 0; i < joints.size(); i++)//sequence
    {
        for (int j = 0; j < joints[i].size(); j++)//joints
        {
            outputFile << joints[i][j];
            if (j != joints[i].size() - 1)
                outputFile << ",";
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}