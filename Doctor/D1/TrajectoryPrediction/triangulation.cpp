#include "triangulation.h"

void Triangulation::main()
{
    /**
    * @brief triangulate 3D points. procedure is as following; matching objects in 2 cameras -> triangulate 3D points 
    * -> convert points from camera to robot base coordinate -> save data in the list -> predict trajectory -> estimate catching candidate point
    */
    Utility utTri;//constructor
    Matching match; //matching algorithm
    std::vector<std::vector<std::vector<double>>> data_3d(numObjects); //{num of objects, sequential, { frameIndex, X,Y,Z }}
    /*while (true)
    {
        if (!seqData_left.empty() || !seqData_right.empty()) break;
    }
    std::cout << "start calculating 3D position" << std::endl;

    int counterIteration = 0;
    int counterFinish = 0;
    int counter = 0;//counter before delete new data
    while (true) // continue until finish
    {
        counterIteration++;
        if (queueFrame.empty())
        {
            if (counterFinish == 10) break;
            counterFinish++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "Sequence :: By finish : remain count is " << (10 - counterFinish) << std::endl;
            continue;
        }
        else
        {
            counterFinish = 0;
            //new detection data available -> get seqData_left, seqData_right -> matching -> triangulate 3D points -> transform from camera to robot base coordinates
            if (!queueUpdateLabels_left.empty() && !queueUpdateLabels_right.empty())
            {
                counter = 0; //reset
                auto start = std::chrono::high_resolution_clock::now();
                std::cout << "start 3d positioning" << std::endl;
                std::vector<std::vector<int>> matchingIndexes; //list for matching indexes : {n_pairs, (idx_left,idx_right)}
                //matching when number of labels increase -> seqData_left.size()>num_obj_left || seqData_right.size()>num_obj_right;
                
                std::cout << "update matching" << std::endl;;
                //matching objects in 2 images 
                match.main(seqData_left, seqData_right, oY_left, oY_right, matchingIndexes); 
                //sort data in index-left ascending way, 0,1,2,3,4,,,
                if (!matchingIndexes.empty()) sortData(matchingIndexes); 
                std::cout << "start 3D positioning" << std::endl;
                //triangulate 3D points
                triangulation(seqData_left, seqData_right, matchingIndexes, data_3d);
                queue_tri2predict.push(true);
                ///queue3DData.push(data_3d);
                queueMatchingIndexes.push(matchingIndexes);
                std::cout << "calculate 3d position" << std::endl;
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "time taken by 3d positioning=" << duration.count() << " microseconds" << std::endl;
            }
            /* at least one data can't be available -> delete data *//*
            else
            {
                if (!queueUpdateLabels_left.empty() || !queueUpdateLabels_right.empty())
                {
                    //std::cout << "both data can't be availble :: left " << !queueTriangulation_left.empty() << ", right=" << queueTriangulation_right.empty() << std::endl;
                    if ((!queueUpdateLabels_left.empty() || !queueUpdateLabels_right.empty()) && counter == 100)
                    {
                        if (!queueUpdateLabels_left.empty()) queueUpdateLabels_left.pop();
                        if (!queueUpdateLabels_right.empty()) queueUpdateLabels_right.pop();
                        counter = 0;
                    }
                    else
                    {
                        std::this_thread::sleep_for(std::chrono::microseconds(200));
                        counter++;
                    }
                }
            }
        }
    }
    std::cout << "***triangulation data***" << std::endl;
    utTri.save3d(data_3d, file_3d);*/
}

bool Triangulation::compareVectors(const std::vector<int>& a, const std::vector<int>& b)
{
    /**
    * @brief compare 2 data with first element. here first element is index of the left object
    */
    return a[0] < b[0];
}

void Triangulation::sortData(std::vector<std::vector<int>>& data)
{
    /**
    * @brief sort Data to arrange the data in index-left ascending order. like {{1,~},{2,~},{3,~}...}, data style is like {{idx_left,idx_right},....}
    * @param[in] data format is like {{idx_left,idx_right},....}
    */
    std::sort(data.begin(), data.end(), [this](const std::vector<int>& a, const std::vector<int>& b) {
        return compareVectors(a, b);
        });
}

void Triangulation::triangulation(std::vector<std::vector<std::vector<double>>>& data_left, std::vector<std::vector<std::vector<double>>>& data_right, 
    std::vector<std::vector<int>>& matchingIndexes, std::vector<std::vector<std::vector<double>>>& data_3d)
{
    /**
    * @brief triangulate 3D points based on sequential data (seqData_left/right) and matching pairs from Matching.main()
    * @param[in] data_left/data_right shape is like {n_objects, sequence, (frame,label,left,top,width,height)}
    * @param[in] matchingIndexes shape is like {n_pairs, (index_left,index_right)}
    * @param[out] data_3d shape is like {num of objects, sequential, { frameIndex, X,Y,Z }}
    */

    //for all matching data
    for (std::vector<int>& matchIndex : matchingIndexes) //triangulate 3D points in the index-left ascending way
    {
        int index = matchIndex[0]; //left object's index
        //calculate objects
        std::vector<std::vector<double>> left = data_left[matchIndex[0]]; //{num frames, {frameIndex, classLabel,left,top,width,height}}
        std::vector<std::vector<double>> right = data_right[matchIndex[1]]; //{num frames, {frameIndex, classLabel,left,top,width,height}}
        //std::cout << "1" << std::endl;
        calulate3Ds(index, left, right, data_3d); //{num of objects, sequential, {frameIndex, X,Y,Z}}
    }
}

void Triangulation::calulate3Ds(int& index, std::vector<std::vector<double>>& left, std::vector<std::vector<double>>& right, std::vector<std::vector<std::vector<double>>>& data_3d)
{
    /**
    * @brief execute calculating 3D positions
    * @param[in] left, right  each object sequential data. shape is like {num frames, {frameIndex, classLabel,left,top,width,height}}
    * @param[out] data_3d storage for saving data. shape is like {num of objects, sequential, { frameIndex, X,Y,Z }}
    */

    std::vector<std::vector<int>> temp;
    int left_frame_start = left[0][0];
    int right_frame_start = right[0][0];
    int num_frames_left = left.size();
    int num_frames_right = right.size();
    //std::cout << "1-1" << std::endl;
    //check whether previous data is in data_3d -> if exist, check last frame Index
    std::vector<cv::Point3d> results;
    std::vector<cv::Point2d> pts_left, pts_right;
    std::vector<double> frames;
   
    if (!data_3d[index].empty())
    {
        int last_frameIndex = (data_3d[index].back())[0]; //get last frame index
        //std::cout << "last_frameIndex=" << last_frameIndex << std::endl;
        //std::cout << "1-2" << std::endl;
        //calculate 3d position for all left data
        //gather frame-matched data 
        int it_left = 0; int it_right = 0;
        cv::Point2d point_tmp;
        while (true)
        {
            if (it_left == num_frames_left || it_right == num_frames_right) break;
            //std::cout << "it_left=" << it_left << ", it_right=" << it_right << ", left[it_left][0]=" << left[it_left][0] << ", right[it_right][0]=" << right[it_right][0] << std::endl;
            //std::cout << "left.size()=" << left.size() << ", right.size()=" << right.size() << std::endl;
            if (left[it_left][0] == right[it_right][0]) //same frame
            {
                if (left[it_left][0] != -1) {//not lost
                    if (left[it_left][0] > last_frameIndex) {//new data
                        //std::cout << "2" << std::endl;
                        std::vector<int> result;
                        point_tmp.x = left[it_left][2] + left[it_left][4] / 2.0;
                        point_tmp.y = left[it_left][3] + left[it_left][5] / 2.0;
                        pts_left.push_back(point_tmp);
                        point_tmp.x = right[it_right][2] + right[it_right][4] / 2.0;
                        point_tmp.y = right[it_right][3] + right[it_right][5] / 2.0;
                        pts_right.push_back(point_tmp);
                        frames.push_back(left[it_left][0]);
                        it_left++;
                        it_right++;
                        //std::cout << "3" << std::endl;
                    }
                    else {
                        it_left++;
                        it_right++;
                    }
                }
                else {
                    it_left++;
                    it_right++;
                }
            }
            else if (left[it_left][0] > right[it_right][0]) it_right++;//right frame is older than left one
            else if (left[it_left][0] < right[it_right][0]) it_left++;//left frame is older than right one
        }
        //triangulate with gathered data, pts_left and pts_right
        if (!pts_left.empty())
        {
            //std::cout << "3" << std::endl;
            if (method_triangulate == 0) {//DLT (Direct Linear Translation)
                dlt(pts_left, pts_right, results);
            }
            else if (method_triangulate == 1) {//stereo triangulation
                stereo3D(pts_left, pts_right, results);
            }
            //push results to data_3d :: {frameIndex, x,y,z}
            if (!results.empty())
            {
                int counter = 0;
                std::vector<double> newData;
                for (cv::Point3d& result : results)
                {
                    //compare latest data with the last one -> if judged as noise, not add
                    double difference = std::pow((data_3d[index].back()[1] - result.x) * (data_3d[index].back()[1] - result.x) + (data_3d[index].back()[2] - result.y) * (data_3d[index].back()[2] - result.y) + (data_3d[index].back()[3] - result.z) * (data_3d[index].back()[3] - result.z), 0.5);
                    if ((frames[counter] - data_3d[index].back()[0])<=counter_valid && difference >= threshold_difference_perFrame) continue;//judge as noise -> not append. Condition is that valid_counter sequential data, frame difference is not far, and distance difference is too big
                    newData.push_back(frames[counter]);
                    newData.push_back(result.x);
                    newData.push_back(result.y);
                    newData.push_back(result.z);
                    data_3d.at(index).push_back(newData);
                    counter++;
                }
            }
            
        }
    }
    //no previous data
    else
    {
        //std::cout << "2" << std::endl;
        //calculate 3d position for all left data
        std::vector<std::vector<int>> temp_3d;
        int it_left = 0; int it_right = 0;
        cv::Point2d point_tmp;
        while (true)
        {
            if (it_left == num_frames_left || it_right == num_frames_right) break;
            //std::cout << "3" << std::endl;
            if (left[it_left][0] == right[it_right][0]) //new data
            {
                if (left[it_left][0] != -1) {//new data
                    std::vector<int> result;
                    point_tmp.x = left[it_left][2] + left[it_left][4] / 2.0;
                    point_tmp.y = left[it_left][3] + left[it_left][5] / 2.0;
                    pts_left.push_back(point_tmp);
                    point_tmp.x = right[it_right][2] + right[it_right][4] / 2.0;
                    point_tmp.y = right[it_right][3] + right[it_right][5] / 2.0;
                    pts_right.push_back(point_tmp);
                    frames.push_back(left[it_left][0]);
                    it_left++;
                    it_right++;
                }
                else {//lost
                    it_left++;
                    it_right++;
                }
            }
            else if (left[it_left][0] > right[it_right][0]) it_right++;
            else if (left[it_left][0] < right[it_right][0]) it_left++;
        }
        //triangulate 
        if (!pts_left.empty())
        {
            //std::cout << "4" << std::endl;
            if (method_triangulate == 0) {//DLT method
                dlt(pts_left, pts_right, results);
            }
            else if (method_triangulate == 1) {//stereo triangulation
                stereo3D(pts_left, pts_right, results);
            }
            //push results to data_3d :: {frameIndex, x,y,z}
            if (!results.empty())
            {
                int counter = 0;
                std::vector<double> newData;
                for (cv::Point3d& result : results)
                {
                    newData.push_back(frames[counter]);
                    newData.push_back(result.x);
                    newData.push_back(result.y);
                    newData.push_back(result.z);
                    data_3d.at(index).push_back(newData);
                    counter++;
                }
            }
            //std::cout << "5" << std::endl;
        }
    }
}

void Triangulation::dlt(std::vector<cv::Point2d>& points_left, std::vector<cv::Point2d>& points_right, std::vector<cv::Point3d>& results)
{
    /**
    * @brief calculate 3D points with DLT method
    * @param[in] points_left, points_right {n_data,(xCenter,yCenter)}
    * @param[out] reuslts 3D points storage. shape is like (n_data, (x,y,z))
    */
    cv::Mat points_left_mat(points_left);
    cv::Mat undistorted_points_left_mat;
    cv::Mat points_right_mat(points_right);
    cv::Mat undistorted_points_right_mat;

    // Undistort the points
    cv::undistortPoints(points_left_mat, undistorted_points_left_mat, cameraMatrix_left, distCoeffs_left);
    cv::undistortPoints(points_right_mat, undistorted_points_right_mat, cameraMatrix_right, distCoeffs_right);

    // Output matrix for the 3D points
    cv::Mat triangulated_points_mat;

    // Triangulate points
    cv::triangulatePoints(projectMatrix_left, projectMatrix_right, undistorted_points_left_mat, undistorted_points_right_mat, triangulated_points_mat);

    // Convert homogeneous coordinates to 3D points
    cv::convertPointsFromHomogeneous(triangulated_points_mat.t(), triangulated_points_mat);
    
    // Access triangulated 3D points
    results = triangulated_points_mat;
    //convert from camera coordinate to robot base coordinate
    for (int i = 0; i < results.size(); i++) {
        double x = results[i].x; double y = results[i].y; double z = results[i].z;
        results[i].x = static_cast<double>(transform_cam2base.at<double>(0,0) * x + transform_cam2base.at<double>(0,1) * y+ transform_cam2base.at<double>(0, 2) * z + transform_cam2base.at<double>(0, 3));
        results[i].y = static_cast<double>(transform_cam2base.at<double>(1, 0) * x + transform_cam2base.at<double>(1, 1) * y + transform_cam2base.at<double>(1, 2) * z + transform_cam2base.at<double>(1, 3));
        results[i].z = static_cast<double>(transform_cam2base.at<double>(2, 0) * x + transform_cam2base.at<double>(2, 1) * y + transform_cam2base.at<double>(2, 2) * z + transform_cam2base.at<double>(2, 3));
    }
}

void Triangulation::stereo3D(std::vector<cv::Point2d>& left, std::vector<cv::Point2d>& right, std::vector<cv::Point3d>& results)
{
    /**
    * @brief calculate 3D points with stereo method
    * @param[in] points_left, points_right {n_data,(xCenter,yCenter)}
    * @param[out] reuslts 3D points storage. shape is like (n_data, (x,y,z))
    */
    int size_left = left.size(); int size_right = right.size(); int size;
    if (size_left <= size_right) size = size_left;
    else size = size_right;

    cv::Point3d result;
    for (int i = 0; i < size; i++) {
        double xl = left[i].x; double xr = right[i].x; double yl = left[i].y; double yr = right[i].y;
        double disparity = xl - xr;
        double X = (double)(BASELINE / disparity) * (xl - oX_left - (fSkew / fY) * (yl - oY_left));
        double Y = (double)(BASELINE * (fX / fY) * (yl - oY_left) / disparity);
        double Z = (double)(fX * BASELINE / disparity);
        /* convert Camera coordinate to robot base coordinate */
        X = static_cast<int>(transform_cam2base.at<double>(0, 0) * X + transform_cam2base.at<double>(0, 1) * Y + transform_cam2base.at<double>(0, 2) * Z + transform_cam2base.at<double>(0, 3));
        Y = static_cast<int>(transform_cam2base.at<double>(1, 0) * X + transform_cam2base.at<double>(1, 1) * Y + transform_cam2base.at<double>(1, 2) * Z + transform_cam2base.at<double>(1, 3));
        Z = static_cast<int>(transform_cam2base.at<double>(2, 0) * X + transform_cam2base.at<double>(2, 1) * Y + transform_cam2base.at<double>(2, 2) * Z + transform_cam2base.at<double>(2, 3));
        result.x = X; result.y = Y; result.z = Z;
        results.push_back(result);
    }
}