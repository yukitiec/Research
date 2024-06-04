// practice.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "stdafx.h"
#include "test.h"
#include "hungarian.h"
#include "hung_test.h"
#include "ur.h"
#include "tangent_vector.h"

void show_data(Rect*);
void dlt(std::vector<cv::Point2f>&, std::vector<cv::Point2f>&, std::vector<cv::Point3f>&);

const int Scale = 2.0;
const float scale_search = 2.0;
const bool bool_crop = true;
const int threshold = 50;

//structure
struct CollisionState {
    double dmin=0; //minimum distance
    double theta_rob_Vobs=0; //angle between robot and obstacle velocity
    double theta_rob_target=0; //angle between robot and target
};

int main()
{
    TangentVector tv_main;
    double k1 = 4.0; double k2 = 5.0; double k3 = 10.0; double rtan = 2.0;
    std::vector<std::vector<double>> coeffs = tv_main.main(k1, k2, k3, rtan);
    for (int i=0;i<coeffs.size();i++)
        tv_main.checkSolution(k1, k2, k3, rtan, coeffs[i][0], coeffs[i][1]);
    std::vector<cv::Point2f> points_left{
        cv::Point2f(1.0f, 2.0f),cv::Point2f(1.0f, 2.0f),cv::Point2f(1.0f, 2.0f)
    };
    std::vector<cv::Point2f> points_right{
        cv::Point2f(1.0f, 2.0f),cv::Point2f(1.0f, 2.0f),cv::Point2f(1.0f, 2.0f)
    };
    std::vector<cv::Point3f> results;
    dlt(points_left, points_right, results);
    for (int i = 0; i < results.size(); i++) {
        std::cout << "result=" << results[i].x << ", " << results[i].y << ", " << results[i].z << std::endl;
    }
    std::vector<int> q = { 2, 3, 1, 5, 7 };

    // Use std::find to check if 1 is in the vector
    if (std::find(q.begin(), q.end(), 1) != q.end()) {
        std::cout << "1 is in the vector." << std::endl;
    }
    else {
        std::cout << "1 is not in the vector." << std::endl;
    }

    cv::Mat matrix2 = cv::Mat::zeros(3, 1, CV_64F);
    for (int i = 0; i < matrix2.rows; i++)
        std::cout << matrix2.at<double>(i) << ",";
    std::cout << std::endl;
    std::vector<std::vector<CollisionState>> b(3, std::vector<CollisionState>(5));
    for (int i = 0; i < b.size(); i++) {
        std::cout << i << "-th joint=";
        for (int j = 0; j < b[i].size(); j++) {
            std::cout << b[i][j].dmin << ", ";
        }
        std::cout << std::endl;
    }
    const int m = 3;
    const double n = 0.3;
    std::cout << m * n << std::endl;
    std::vector<double> joints{ 5.27,3.316,1.0297,3.4732,2.0943,1.5707 };
    UR ur;
    auto start = std::chrono::high_resolution_clock::now();
    ur.cal_poseAll(joints);
    // Print the rotation matrix
    //std::cout
    //robot pose info
    std::vector<std::vector<double>> robot_previous;
    // Record the end time
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "process time=" << duration.count() << " microseconds" << std::endl;
    robot_previous = std::vector<std::vector<double>>{ ur.pose1,ur.pose2,ur.pose3,ur.pose4,ur.pose5,ur.pose6 };
    std::cout << "UR pose=" << std::endl;
    for (int i = 0; i < robot_previous.size(); i++) {
        std::cout << i << "-th joint=";
        for (int j = 0; j < robot_previous[i].size(); j++) {
            std::cout << robot_previous[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<double>> a(6, std::vector<double>(6, 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            std::cout << a[i][j] << ",";
        }
        std::cout << std::endl;
    }
    // Define a 4x4 matrix
    cv::Mat matrix0 = (cv::Mat_<double>(4, 4) <<
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16);

    // Extract the first 3x3 submatrix
    cv::Mat rotationMatrix = matrix0(cv::Range(0, 3), cv::Range(0, 3));
    std::cout << "1,2 = " << matrix0.at<double>(1, 2) << std::endl;

    // Print the submatrix
    std::cout << "Submatrix:" << std::endl;

    std::cout << rotationMatrix << std::endl;

    // Define two 4x4 matrices
    cv::Mat mat1 = (cv::Mat_<double>(4, 4) <<
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16);

    cv::Mat mat2 = (cv::Mat_<double>(4, 4) <<
        1, 1, 0, 0,
        0, 1, 0, 0,
        0, 2, 1, 0,
        0, 0, 3, 1);

    // Method 1: Using cv::Mat::mul()
    cv::Mat result1 = mat1 * mat2*mat2;
    std::cout << result1 << std::endl;

    // Method 2: Using overloaded operator *
    cv::Mat result2 = mat1 * mat2;

    // Print the results
    std::cout << "Result using cv::Mat::mul():" << std::endl;
    std::cout << result1 << std::endl << std::endl;

    Hung_test test;
    test.main();

    // Create a 3x3 matrix
    Eigen::Matrix3d matrix;
    matrix << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;

    // Accessing each element
    for (int i = 0; i < matrix.rows(); ++i) {

        for (int j = 0; j < matrix.cols(); ++j) {
            std::cout << "Element at (" << i << ", " << j << "): " << matrix(i, j) << std::endl;
        }
    }

    Rect rect1 = { "float",1,1,3,5 };
    std::cout << "rect1 type : " << rect1.type << ", left=" << rect1.left << ", top=" << rect1.top << ", width=" << rect1.width << ", height=" << rect1.height << std::endl;
    show_data(&rect1);
    cv::Rect search;
    // declares all required variables
    cv::Rect roi;
    cv::Mat frame;
    cv::Mat templateImg;
    // set input video
    std::string video = "ball_box_0119_left.mp4";
    cv::VideoCapture cap(video);

    // perform the tracking process
    std::printf("Start the tracking process, press ESC to quit.\n");
    int counter = 0;
    for (;; ) {
        break;
        // get frame from the video
        //auto start = std::chrono::high_resolution_clock::now();
        cap >> frame;
        // stop the program if no more images
        if (frame.rows == 0 || frame.cols == 0)
            break;
        if (counter <= 150)
        {
            std::cout << counter << std::endl;
            counter++;
            continue;
        }
        //preprocess
        cv::resize(frame, frame, cv::Size((int)(frame.cols / Scale), (int)(frame.rows / Scale))); //cv::Size size(width, height)
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        auto start = std::chrono::high_resolution_clock::now();
        //std::cout << "resize image" << std::endl;
        if (counter == 151) //select ROI
            roi = selectROI("select ROI", frame);

        //cropp image for high-speed
        int left_search = std::min(frame.cols, std::max(0, (int)(roi.x - roi.width * (scale_search / 2))));
        int top_search = std::min(frame.rows, std::max(0, (int)(roi.y - roi.height * (scale_search / 2))));
        int right_search = std::min(frame.cols, std::max(left_search, (int)(roi.x + roi.width * (1 + scale_search / 2))));
        int bottom_search = std::min(frame.rows, std::max(top_search, (int)(roi.y + roi.height * (1 + scale_search / 2))));
        cv::Mat croppedFrame;
        if (bool_crop && right_search > left_search && bottom_search > top_search)
        {
            croppedFrame = gray(cv::Rect(left_search, top_search, (right_search - left_search), (bottom_search - top_search)));
            roi.x -= left_search; //world -> local
            roi.y -= top_search; //world -> local
        }
        else
            croppedFrame = gray.clone();

        // First frame, give the groundtruth to the tracker
        if (counter == 151) {
            templateImg = croppedFrame(roi);
            cv::Mat template_binary;
            cv::threshold(templateImg, template_binary, threshold, 255, cv::THRESH_BINARY);
            cv::imwrite("templateImg.png", templateImg);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(template_binary, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE); //wanna only outer contour -> cv::RETR_EXTERNAL

            // Draw contours on a black background
            cv::Mat contoursImage = cv::Mat::zeros(templateImg.size(), CV_8UC3);
            cv::drawContours(template_binary, contours, -1, cv::Scalar(255), 2);
            cv::imwrite("contour.png", template_binary);
        }
        break;
        // Update
    }
    return 0;
}


void show_data(Rect* r){
    std::cout << "type=" << r->type;
    std::cout << "centerX=" << r->left + r->width / 2 << ", centerY=" << r->top + r->height / 2 << std::endl;
}


void dlt(std::vector<cv::Point2f>& points_left, std::vector<cv::Point2f>& points_right, std::vector<cv::Point3f>& results)
{
    /**
    * @brief calculate 3D points with DLT method
    * @param[in] points_left, points_right {n_data,(xCenter,yCenter)}
    * @param[out] reuslts 3D points storage. shape is like (n_data, (x,y,z))
    */

    const cv::Mat cameraMatrix_left = (cv::Mat_<double>(3, 3) << 754.66874569, 0, 255.393104, // fx: focal length in x, cx: principal point x
        0, 754.64708568, 335.6848201,                           // fy: focal length in y, cy: principal point y
        0, 0, 1                                // 1: scaling factor
        );
    const cv::Mat cameraMatrix_right = (cv::Mat_<double>(3, 3) << 802.62616415, 0, 286.48516862, // fx: focal length in x, cx: principal point x
        0, 802.15806832, 293.54957668,                           // fy: focal length in y, cy: principal point y
        0, 0, 1                                // 1: scaling factor
        );
    const cv::Mat distCoeffs_left = (cv::Mat_<double>(1, 5) << -0.00661832, -0.19633213, 0.00759942, -0.01391234, 0.73355661);
    const cv::Mat distCoeffs_right = (cv::Mat_<double>(1, 5) << 0.00586444, -0.18180071, 0.00489287, -0.00392576, 1.20394993);
    const cv::Mat projectMatrix_left = (cv::Mat_<double>(3, 4) << 375.5, 0, 249.76, 0, // fx: focal length in x, cx: principal point x
        0, 375.5, 231.0285, 0,                           // fy: focal length in y, cy: principal point y
        0, 0, 1, 0                                // 1: scaling factor
        );
    const cv::Mat projectMatrix_right = (cv::Mat_<double>(3, 4) << 375.5, 0, 249.76, -280, // fx: focal length in x, cx: principal point x
        0, 375.5, 231.028, 0,                           // fy: focal length in y, cy: principal point y
        0, 0, 1, 0                               // 1: scaling factor
        );

    cv::Mat points_left_mat(points_left);
    cv::Mat undistorted_points_left_mat;
    cv::Mat points_right_mat(points_right);
    cv::Mat undistorted_points_right_mat;
    for (int i = 0; i < points_left_mat.rows; i++) {
        for (int j = 0; j < points_left_mat.cols; j++) {
            std::cout << points_left_mat.at<float>(i, j) << ",";
        }
        std::cout << std::endl;
    }
    // Undistort the points
    cv::undistortPoints(points_left_mat, undistorted_points_left_mat, cameraMatrix_left, distCoeffs_left);
    cv::undistortPoints(points_right_mat, undistorted_points_right_mat, cameraMatrix_right, distCoeffs_right);
    for (int i = 0; i < undistorted_points_left_mat.rows; i++) {
        for (int j = 0; j < undistorted_points_left_mat.cols; j++) {
            std::cout << undistorted_points_left_mat.at<float>(i, j) << ",";
        }
        std::cout << std::endl;
    }
    // Output matrix for the 3D points
    cv::Mat triangulated_points_mat;

    // Triangulate points
    cv::triangulatePoints(projectMatrix_left, projectMatrix_right, undistorted_points_left_mat, undistorted_points_right_mat, triangulated_points_mat);
    for (int i = 0; i < triangulated_points_mat.rows; i++) {
        for (int j = 0; j < triangulated_points_mat.cols; j++) {
            std::cout << triangulated_points_mat.at<float>(i, j) << ",";
        }
        std::cout << std::endl;
    }
    // Convert homogeneous coordinates to 3D points
    cv::convertPointsFromHomogeneous(triangulated_points_mat.t(), triangulated_points_mat);

    for (int i = 0; i < triangulated_points_mat.rows; i++) {
        for (int j = 0; j < triangulated_points_mat.cols; j++) {
            std::cout << triangulated_points_mat.at<float>(i, j) << ",";
        }
        std::cout << std::endl;
    }
    
    // Access triangulated 3D points
    results = triangulated_points_mat;
}
// プログラムの実行: Ctrl + F5 または [デバッグ] > [デバッグなしで開始] メニュー
// プログラムのデバッグ: F5 または [デバッグ] > [デバッグの開始] メニュー

// 作業を開始するためのヒント: 
//    1. ソリューション エクスプローラー ウィンドウを使用してファイルを追加/管理します 
//   2. チーム エクスプローラー ウィンドウを使用してソース管理に接続します
//   3. 出力ウィンドウを使用して、ビルド出力とその他のメッセージを表示します
//   4. エラー一覧ウィンドウを使用してエラーを表示します
//   5. [プロジェクト] > [新しい項目の追加] と移動して新しいコード ファイルを作成するか、[プロジェクト] > [既存の項目の追加] と移動して既存のコード ファイルをプロジェクトに追加します
//   6. 後ほどこのプロジェクトを再び開く場合、[ファイル] > [開く] > [プロジェクト] と移動して .sln ファイルを選択します
