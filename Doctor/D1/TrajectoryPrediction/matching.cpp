#include "matching.h"


void Matching::main(std::vector<std::vector<std::vector<double>>>& seqData_left, std::vector<std::vector<std::vector<double>>>& seqData_right,
    const double& oY_left, const double& oY_right, std::vector<std::vector<int>>& matching)
{
    /**
    * @brief match objects in 2 cameras with hungarian algorithm in stereo vision
    * @param[in] seqData_left,seqData_right : {n_objects, sequence, (frame,label,left,top,width,height)}
    * @param[in] oY_left,oY_right : optical point in each camera
    * @param[out] matching pairs of index in seqData {n_
    
    pairs, (idx_left, idx_right)}
    */
    //diffenrence in y-axis between each camera
    Delta_oy = oY_right - oY_left;
    std::vector<std::vector<double>> ball_left, ball_right, box_left, box_right; //storage for data {n_objects,(frame,label,left,top,width,height)}
    std::vector<int> ball_index_left, box_index_left, ball_index_right, box_index_right; //storage for index
    //extract position and index, and then sort -> 300 microseconds for 2 
    frame_left = arrangeData(seqData_left, ball_left, box_left, ball_index_left, box_index_left); //sort data
    frame_right = arrangeData(seqData_right, ball_right, box_right, ball_index_right, box_index_right); //sort data, retain indexes 
    if (bool_hungarian) {//use Hungarian algorithm
        if (frame_left == frame_right) {
            frame_latest = frame_left;
            matchingHung(ball_left, ball_right, ball_index_left, ball_index_right, matching);
            matchingHung(box_left, box_right, box_index_left, box_index_right, matching);
        }
    }
    else if (!bool_hungarian) { //don't use Hungarian algorithm
        //matching data in y value
        //ball
        int num_left = ball_index_left.size();
        int num_right = ball_index_right.size();
        auto start_time2 = std::chrono::high_resolution_clock::now();
        // more objects detected in left camera -> 79 microseconds for 2
        matchingObj(ball_left, ball_right, ball_index_left, ball_index_right, oY_left, oY_right, matching);
        //box
        matchingObj(box_left, box_right, box_index_left, box_index_right, oY_left, oY_right, matching);
    }

    if (debug)
    {
        for (int i = 0; i < matching.size(); i++)
            std::cout << i << "-th matching :: left : " << matching[i][0] << ", right: " << matching[i][1] << std::endl;
    }
}

int Matching::arrangeData(std::vector<std::vector<std::vector<double>>>& seqData, std::vector<std::vector<double>>& data_ball, std::vector<std::vector<double>>& data_box,
    std::vector<int>& index_ball, std::vector<int>& index_box)
{
    /**
    * @brief arrange data to convert data based on labels
    * @param[in] seqData : {n_objects, sequence, (frame,label,left,top,width,height)}
    * @param[out] data_ball,data_box : latest data for ball and box {n_objects,(frame,label,left,top,width,height)}
    * @param[out] index_ball,index_box : index list of successful data
    * @return latest frame
    */
    int frame_latest=0;
    for (int i = 0; i < seqData.size(); i++)//for each object
    {
        if (seqData[i].back()[0] != -1) //tracking is successful
        {
            if (seqData[i].back()[1] == 0) {//ball
                data_ball.push_back(seqData[i].back());
                index_ball.push_back(i);
            }
            else if (seqData[i].back()[1] == 1) {//box
                data_box.push_back(seqData[i].back());
                index_box.push_back(i);
            }
            if (seqData[i].back()[0] > frame_latest)
                frame_latest = seqData[i].back()[0];
        }
    }
    if (!bool_hungarian) {
        // sort data in y-coordinate in ascending order
        sortData(data_ball, index_ball);
        sortData(data_box, index_box);
    }
    return frame_latest;
}

void Matching::sortData(std::vector<std::vector<double>>& data, std::vector<int>& classes)
{
    /**
    * @brief sort data in y-coordinate ascending order
    * @param[out] data {n objects, (frame,label,left,top,width,height)}
    * @param[out] classes inddex list
    */

    // Create an index array to remember the original order
    std::vector<size_t> index(data.size());
    for (size_t i = 0; i < index.size(); ++i)
    {
        index[i] = i;
    }
    // Sort data1 based on centerX values and apply the same order to data2 : {frameIndex,label,left,top,width,height}
    std::sort(index.begin(), index.end(), [&](size_t a, size_t b)
        { return (data[a][3] + data[a][5] / 2) >= (data[b][3] + data[b][5] / 2); });

    std::vector<std::vector<double>> sortedData(data.size());
    std::vector<int> sortedClasses(classes.size());

    for (size_t i = 0; i < index.size(); ++i)
    {
        sortedData[i] = data[index[i]];
        sortedClasses[i] = classes[index[i]];
    }

    data = sortedData;
    classes = sortedClasses;
}

void Matching::matchingObj(std::vector<std::vector<double>>& ball_left, std::vector<std::vector<double>>& ball_right, std::vector<int>& ball_index_left, std::vector<int>& ball_index_right,
    const float& oY_left, const float& oY_right, std::vector<std::vector<int>>& matching)
{
    /**
    * matching objects in both images with label, bbox info. and y-position
    * Args:
    *   ball_left, ball_riht : {frameIndex, label,left,top,width,height}
    * Return:
    *   matching : {{index_left, index_right}}
    */
    int dif_min = dif_threshold;
    int matchIndex_right;
    int i = 0;
    int startIndex = 0; //from which index to start comparison
    //for each object
    while (i < ball_left.size() && startIndex < ball_right.size())
    {
        int j = 0;
        bool boolMatch = false;
        dif_min = dif_threshold;
        std::cout << "startIndex = " << startIndex << std::endl;
        //continue right object y-value is under threshold
        while (true)
        {
            //if right detection is too low -> stop searching
            if (((ball_left[i][3] - ball_right[startIndex + j][3]) > dif_threshold) || (startIndex + j < ball_right.size())) break;
            //bbox info. criteria
            if (((float)ball_left[i][4] * MAX_ROI_DIF >= (float)ball_right[startIndex + j][4]) && ((float)ball_left[i][5] * MAX_ROI_DIF >= (float)ball_right[startIndex + j][5]) &&
                ((float)ball_left[i][4] * MIN_ROI_DIF <= (float)ball_right[startIndex + j][4]) && ((float)ball_left[i][5] * MIN_ROI_DIF <= (float)ball_right[startIndex + j][5]))
            {
                std::cout << "dif_in_2imgs=" << std::abs(((float)(ball_left[i][3] + ball_left[i][5] / 2) - oY_left) - ((float)(ball_right[startIndex + j][3] + ball_right[startIndex + j][5] / 2) - oY_right)) << std::endl;
                int dif = std::abs(((float)(ball_left[i][3] + ball_left[i][5] / 2) - oY_left) - ((float)(ball_right[startIndex + j][3] + ball_right[startIndex + j][5] / 2) - oY_right));
                if (dif < dif_min)
                {
                    dif_min = dif;
                    matchIndex_right = startIndex + j;
                    boolMatch = true;
                }
            }
            j++;
        }
        std::cout << "matching objects found? : " << boolMatch << std::endl;
        //match index is the last value
        if (boolMatch && (matchIndex_right == (startIndex + j - 1))) startIndex += j;
        else startIndex += std::max(j - 1, 0);
        /* matching successful*/
        if (boolMatch)
        {
            matching.push_back({ ball_index_left[i],ball_index_right[matchIndex_right] }); //save matching pair
            //delete selected data
            ball_index_left.erase(ball_index_left.begin() + i);
            ball_left.erase(ball_left.begin() + i);
            //ball_index_right.erase(ball_index_right.begin() + matchIndex_right);
            //ball_right.erase(ball_right.begin() + matchIndex_right);
        }
        // can't find matching object
        else
            i++;
    }
}


void Matching::matchingHung(std::vector<std::vector<double>>& ball_left, std::vector<std::vector<double>>& ball_right, 
    std::vector<int>& ball_index_left, std::vector<int>& ball_index_right,std::vector<std::vector<int>>& matching) {
    /**
    * @brief match objects in 2 cameras with Hungarian algorithm
    * @param[in] ball_left,ball_right {n objects, (frame,label,left,top,width,height)}
    * @param[in]ball_index_left, ball_index_right index list of seqData {idx1,idx2,...
    * @param[out] matching {{index_left, index_right}}
    */

    //prepare cost matrix
    std::vector<std::vector<double>> costMatrix_x, costMatrix_y, costMatrix_size, costMatrix; //for storing matching data
    std::vector<double> cost_x, cost_y, cost_size;//for calculating max and min
    //matching object with hungarian algorithm
    double delta_x, delta_y, delta_size;
    if (!ball_left.empty() && !ball_right.empty()) {//both are not empty
        int i = 0;
        while (true) {//for each object in left cam
            if (i >= ball_left.size()) break;
            if (ball_left[i][0] != frame_latest){//not succeeded in latest data -> delete ball_index_left
                ball_left.erase(ball_left.begin() + i);
                ball_index_left.erase(ball_index_left.begin() + i);
                continue; 
            }
            //std::cout << "ball_left.size()=" << ball_left.size() << ", ball_right.size()=" << ball_right.size() << std::endl;
            std::vector<double> cost_x_tmp, cost_y_tmp, cost_size_tmp;
            int j = 0;
            while (true) {//for each object in right
                if (j >= ball_right.size()) break;
                if (ball_right[j][0] != frame_latest) {//not succeeded in latest data -> delte ball_index_right
                    ball_right.erase(ball_right.begin() + j);
                    ball_index_right.erase(ball_index_right.begin() + j);
                    continue;
                }
                //prepare basic info
                double centerX_left = ball_left[i][2] + ball_left[i][4] / 2;//left
                double centerY_left = ball_left[i][3] + ball_left[i][5] / 2;
                double centerX_right = ball_right[j][2] + ball_right[j][4] / 2;//right
                double centerY_right = ball_right[j][3] + ball_right[j][5] / 2;
                double w_left = ball_left[i][4];
                double h_left = ball_left[i][5];
                double w_right = ball_right[j][4];
                double h_right = ball_right[j][5];
                //diffenrence in x axis
                //right x coordinates go over left x coordinate
                if (centerX_right >= centerX_left) delta_x = 1.0;
                else {//candidate
                    delta_x = 1.0 / (1.0 + std::exp(-1 * slope_x * (centerX_right - (centerY_left - mu_x))));
                    if (delta_x < epsilon) delta_x = 0.0;
                }
                //std::cout << "delta_x=" << delta_x << std::endl;
                //difference in y axis
                delta_y = std::abs(centerY_right - (Delta_oy + centerY_left));
                //diffenrence in size
                delta_size = (std::abs(w_left - w_right) + std::abs(h_left - h_right)) * (1 + std::abs((h_left / w_left) - (h_right / w_right)));
                //save data into storage
                cost_x_tmp.push_back(delta_x);
                cost_y_tmp.push_back(delta_y);
                cost_size_tmp.push_back(delta_size);
                cost_x.push_back(delta_x);
                cost_y.push_back(delta_y);
                cost_size.push_back(delta_size);
                j++;//increment
            }
            //save data in each costMatrix
            costMatrix_x.push_back(cost_x_tmp);
            costMatrix_y.push_back(cost_y_tmp);
            costMatrix_size.push_back(cost_size_tmp);
            i++;//increment
        }//finish calculate cost

        if (!cost_x.empty()) {//candidates exist
            //normalize cost in x, y and size->calculate max and min in each cost
            // Calculate the maximum element
            //x
            //auto maxIt_x = std::max_element(cost_x.begin(), cost_x.end());
            //double max_x = *maxIt_x;
            //auto minIt_x = std::min_element(cost_x.begin(), cost_x.end());
            //double min_x = *minIt_x;
            //y
            auto maxIt_y = std::max_element(cost_y.begin(), cost_y.end());
            double max_y = *maxIt_y;
            auto minIt_y = std::min_element(cost_y.begin(), cost_y.end());
            double min_y = *minIt_y;
            //size
            auto maxIt_size = std::max_element(cost_size.begin(), cost_size.end());
            double max_size = *maxIt_size;
            auto minIt_size = std::min_element(cost_size.begin(), cost_size.end());
            double min_size = *minIt_size;
            std::cout << "costMatrix_x, size=" << costMatrix_x.size() << std::endl;;
            for (int row = 0; row < costMatrix_x.size(); row++)
                for (int col = 0; col < costMatrix_x[row].size(); col++)
                    std::cout << costMatrix_x[row][col] << ",";
                std::cout<< std::endl;
            //combine all cost 
            for (int i = 0; i < costMatrix_x.size(); i++) {//for each object in left cam
                std::vector<double> cost_tmp;
                for (int j = 0; j < costMatrix_x[i].size(); j++) {//for each object in right cam
                    double c_x = 0; double c_y = 0;  double c_size = 0; double c;
                    //if (max_x-min_x>1e-6)
                    //c_x = (costMatrix_x[i][j] - min_x) / (max_x - min_x);
                    c_x = costMatrix_x[i][j];//already normalized
                    if (max_y-min_y>1e-6)
                        c_y = (costMatrix_y[i][j] - min_y) / (max_y - min_y);
                    if (max_size-min_size>1e-6)
                        c_size = (costMatrix_size[i][j] - min_size) / (max_size - min_size);
                    c = lambda_x * c_x + lambda_y * c_y + lambda_size * c_size;
                    //c = lambda_x * costMatrix_x[i][j] + lambda_y * costMatrix_y[i][j] + lambda_size * costMatrix_size[i][j];
                    cost_tmp.push_back(c);
                }
                costMatrix.push_back(cost_tmp);
            }//finish calculating all costMatrix
            //std::cout << "costMatrix=" << std::endl;
            //for (int i = 0; i < costMatrix.size(); i++) {
            //    for (int j = 0; j < costMatrix[i].size(); j++) {
            //        std::cout << costMatrix[i][j] << " ";
            //    }
            //    std::cout << std::endl;
            //}
            //hungarian algorithm -> calculate optimal combination
            std::vector<int> assignment;//assignment
            double cost = HungAlgo.Solve(costMatrix, assignment);
            for (unsigned int x = 0; x < assignment.size(); x++) { //for each candidates in a left camera
                int index_match = assignment[x];
                if ((index_match >= 0) && (costMatrix_y[x][assignment[x]] <= threshold_ydiff) && (costMatrix_x[x][assignment[x]]<1.0)) {//matching is successful
                    int idx_left = ball_index_left[x];
                    int idx_right = ball_index_right[assignment[x]];
                    matching.push_back({ idx_left,idx_right });
                    //std::cout << "matching is successful:: diff_y=" << costMatrix_y[x][assignment[x]] << ", diff_x=" << costMatrix_x[x][assignment[x]] << ", diff_size=" << costMatrix_x[x][assignment[x]] << std::endl;
                }
            }
        }
    }
}