// trajectory_prediction.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//
#include "stdafx.h"
#include "hungarian.h"
#include "matching.h"
#include "triangulation.h"
#include "utility.h"
#include "prediction.h"

const std::string file_bbox_left = "bbox_left_3thrown_gt.csv";
const std::string file_bbox_right = "bbox_right_3thrown_gt.csv";
const std::string file_label_left = "label_left_3thrown_gt.csv";
const std::string file_label_right = "label_right_3thrown_gt.csv";

const double Rmse_identity = 50; //20 pixel for idntifying the same objects

std::vector<std::vector<std::vector<double>>> makeSeqData(const std::string,  std::string, std::vector<std::vector<std::vector<double>>>& seqData_left);
std::vector<std::vector<double>> readCSV(const std::string&);//read csv file 
double calculateRMSE(std::vector<double>&, std::vector<double>&);
void checkSeqData(std::vector<std::vector<std::vector<double>>>&, std::string);
std::vector<std::vector<std::vector<double>>> permute(std::vector<std::vector<std::vector<double>>>&);
void saveMatching(std::vector<std::vector<std::vector<int>>>&, std::string&);

int main()
{
    Matching matching_main; //construct Matching class
    Triangulation tri_main; //construct Triangulation class
    int numObjects = 100; //prepare 100 objects' storage
    std::vector<std::vector<std::vector<double>>> data_3d(numObjects); //{num of objects, sequential, { frameIndex, X,Y,Z }}
    Utility utility_main;//utility
    Prediction pred_main;//prediction
    std::vector<std::vector<std::vector<double>>> targetPoint(numObjects), targetPoint_save(numObjects);//target points. {n_objects, sequence, {frame_target,x,y,z}}, targetPoint_save{counter,frame_target,x,y,z}
    std::string file_left = "seqData_left.csv"; std::string file_right = "seqData_right.csv";
    //make sequential data :: {num_objects,sequence,position(frame,label,left,top,width,height)}
    std::vector<std::vector<std::vector<double>>> seq_left, seq_right;//{sequence, n_objects, (frame,label,left,top,width,height)}
    std::vector<std::vector<std::vector<double>>> seqData_left = makeSeqData(file_bbox_left, file_left,seq_left);
    std::vector<std::vector<std::vector<double>>> seqData_right= makeSeqData(file_bbox_right, file_right,seq_right);
    std::vector<std::vector<std::vector<double>>> seq_cum_left, seq_cum_right;
    double last_frame=0;
    for (int i = 0; i < seqData_left.size(); i++) {//for each object
        if (seqData_left[i].back()[0] > last_frame)
            last_frame = seqData_left[i].back()[0];
    }
    for (int i = 0; i < seqData_right.size(); i++) {//for each object
        if (seqData_right[i].back()[0] > last_frame)
            last_frame = seqData_right[i].back()[0];
    }
    int counter_left = 0; int counter_right = 0;
    double oY_left = 335.6848201;
    double oY_right = 293.54957668;
    std::vector<double> frames;
    std::vector<std::vector<std::vector<int>>> matchings;
    double t_elapsed = 0.0;
    int count_process = 0;
    //matching starts
    for (int i = 1; i < last_frame; i++) {
        std::vector<std::vector<int>> matching;
        if (counter_left < seq_left.size()) {
            if (seq_left[counter_left][0][0] == i) {//i-th frame exists
                if (seq_cum_left.empty()) {//first
                    seq_cum_left.push_back(seq_left[counter_left]);
                }
                else {//not first
                    for (int j = 0; j < seq_left[counter_left].size(); j++) {//for each object
                        if (j < seq_cum_left.size()) {//existed objects
                            seq_cum_left[j].push_back(seq_left[counter_left][j]);
                        }
                        else {//new objects
                            seq_cum_left.push_back({ seq_left[counter_left][j] });
                        }
                    }
                }
                counter_left++;
            }
        }
        if (counter_right < seq_right.size()) {
            if (seq_right[counter_right][0][0] == i) {//i-th frame exists
                if (seq_cum_right.empty()) {//first
                    seq_cum_right.push_back(seq_right[counter_right]);
                }
                else {//not first
                    for (int j = 0; j < seq_right[counter_right].size(); j++) {//for each object
                        if (j < seq_cum_right.size()) {//existed objects
                            seq_cum_right[j].push_back(seq_right[counter_right][j]);
                        }
                        else {//new objects
                            seq_cum_right.push_back({ seq_right[counter_right][j] });
                        }
                    }
                }
                counter_right++;
            }
        }
        if (!seq_cum_left.empty() && !seq_cum_right.empty()) {
            auto start = std::chrono::high_resolution_clock::now();
            matching_main.main(seq_cum_left, seq_cum_right, oY_left, oY_right, matching);
            if (!matching.empty()) tri_main.sortData(matching);
            //std::cout << "start 3D positioning" << std::endl;
            //triangulate 3D points
            tri_main.triangulation(seq_cum_left, seq_cum_right, matching, data_3d);
            //predict trajectory
            int it = 0;
            double depth_target = 1300;//[mm]

            for (std::vector<std::vector<double>>& data : data_3d)//for each objects
            {
                if (data.size() >= 15)
                {
                    pred_main.predictTargets(it, depth_target, data, targetPoint);//output is {frame_target, x_target,y_target,depth_target}
                }
                it++;
            }
            //prediction
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "process time=" << duration.count() << " microseconds" << std::endl;
            t_elapsed += duration.count();
            count_process++;
            if (!matching.empty()) {
                int counter_idx_left = 0;
                for (int j = 0; j < matching.size(); j++) {//matchings {n_idx_left,sequence,(frame,idx_right)}
                    std::cout << "matching : idx_left=" << matching[j][0] << ", idx_right=" << matching[j][1] << std::endl;
                    if (matching[j][0] > counter_idx_left) {//not find counter_idx_left 
                        if (matchings.size() < (counter_idx_left + 1)) //never find -> make an empty instance
                            matchings.push_back(std::vector<std::vector<int>>{});
                        counter_idx_left++;
                    }
                    else if (matching[j][0] == counter_idx_left) {//find counter_idx_left
                        if (matchings.size() < (counter_idx_left + 1)) {//first
                            std::vector<std::vector<int>> a{ {(int)i,matching[j][1]} };
                            matchings.push_back(a);
                        }
                        else //not first
                            matchings[counter_idx_left].push_back({ (int)i,matching[j][1] });
                        counter_idx_left++;
                    }
                }
            }
            //save target points into targetPoints_save
            double alpha = 0.2;//adopt new value
            for (int j = 0; j < targetPoint.size(); j++) {//for each object
                if (targetPoint[j].size() > targetPoint_save[j].size()) {//for each iteration
                    for (int k = targetPoint_save[j].size(); k < targetPoint[j].size(); k++) {//for each sequential target data
                        if (targetPoint_save[j].empty())//first data
                            targetPoint_save[j].push_back({ (double)i,targetPoint[j][k][0],targetPoint[j][k][1],targetPoint[j][k][2],targetPoint[j][k][3] });
                        else//update
                            targetPoint_save[j].push_back({ (double)i,((1-alpha)*targetPoint_save[j].back()[1]+alpha*targetPoint[j][k][0]),((1 - alpha) * targetPoint_save[j].back()[2] + alpha * targetPoint[j][k][1]),((1 - alpha) * targetPoint_save[j].back()[3] + alpha * targetPoint[j][k][2]),((1 - alpha) * targetPoint_save[j].back()[4] + alpha * targetPoint[j][k][3]) });
                    }
                }
            }
        }
    }
    //save matching results
    std::string file_matching = "matching.csv";
    saveMatching(matchings, file_matching);
    std::string file_3d = "C:/Users/kawaw/cpp/trajectory_prediction/analysis/3d/3d.csv";
    utility_main.save3d(data_3d, file_3d);
    std::string file_target = "C:/Users/kawaw/cpp/trajectory_prediction/analysis/3d/target.csv";
    utility_main.saveTarget(targetPoint_save, file_target);
    std::cout << "matching process speed = " << 1000000 * (count_process / t_elapsed) << " Hz" << std::endl;

    std::vector<std::vector<double>> data_test{ {1,1,1,1},{2,2,4,3},{3,3,9,5},{4,4,16,7}, {5,5,25,9},{6,6,36,11} };
    std::vector<double> result;
    pred_main.linearRegression(data_test, result);
    pred_main.linearRegressionZ(data_test, result);
    pred_main.curveFitting(data_test, result);
    return 0;
}

std::vector<std::vector<std::vector<double>>> makeSeqData(const std::string file_bbox_left,std::string file_output, std::vector<std::vector<std::vector<double>>>& seqData_left) {
    std::vector<std::vector<double>> csvData_left = readCSV(file_bbox_left); //load csv file
    //prepare sequential data
    //find -> label is 0, not find -> label is -1
    std::vector<std::vector<double>> data_init;
    for (int i = 0; i < (int)(csvData_left[0].size() / 5); i++)
        data_init.push_back({ csvData_left[0][i * 5],(double)0,csvData_left[0][i * 5 + 1],csvData_left[0][i * 5 + 2],csvData_left[0][i * 5 + 3],csvData_left[0][i * 5 + 4] });
    seqData_left.push_back(data_init);
    //Hungarian algorithm to make sequence data 
    HungarianAlgorithm HungAlgo;
    for (int i = 1; i < csvData_left.size(); i++) {//for each time step
        //std::cout << "csvData_left.size()=" << csvData_left.size() <<", i="<<i << std::endl;
        //calculate cost matrix
        std::vector<std::vector<double>> costMatrix; //for storing matching data
        std::vector<std::vector<double>> data_next; //next data
        std::vector<std::vector<double>> data_last = seqData_left.back(); //last data
        for (int j = 0; j < (int)(csvData_left[i].size() / 5); j++) //set next data
            data_next.push_back({ csvData_left[i][j * 5],(double)0,csvData_left[i][j * 5 + 1],csvData_left[i][j * 5 + 2],csvData_left[i][j * 5 + 3],csvData_left[i][j * 5 + 4] });
        std::vector<int> index_list; //index list for identifying the past data
        for (int j = 0; j < data_last.size(); j++) {//for each existed data
            std::vector<double> costMatrix_temp;
            if (data_last[j][0] != -1) {
                index_list.push_back(j);
                for (int k = 0; k < data_next.size(); k++) { //for each latest data
                    double rmse = calculateRMSE(data_last[j], data_next[k]);
                    costMatrix_temp.push_back(rmse); //add rmse data
                }
                costMatrix.push_back(costMatrix_temp);
            }
        }
        std::vector<int> assignment;//assignment
        double cost = HungAlgo.Solve(costMatrix, assignment);
        //assign
        int counter = 0; //counter for compensating lost data
        std::vector<std::vector<double>> data_new, data_latest; //next data
        for (unsigned int x = 0; x < assignment.size(); x++) { //for each candidates
            int index_match = assignment[x];
            while (counter < index_list[x]) {//find lost data -> compensate with {-1,-1,-1,-1,-1,-1}
                data_new.push_back(std::vector<double>{-1, -1, -1, -1, -1, -1});
                counter++;
            }
            if ((index_match >= 0) && costMatrix[x][assignment[x]] <= Rmse_identity) {//the same objects found
                data_new.push_back(data_next[assignment[x]]); //update with next data
            }
            else if ((index_match >= 0) && costMatrix[x][assignment[x]] > Rmse_identity) {//not found the same data
                data_new.push_back(std::vector<double>{-1, -1, -1, -1, -1, -1});//compensate with -1
                data_latest.push_back(data_next[assignment[x]]);//push latest data for pushing back later
            }
            else if (index_match < 0) {
                data_new.push_back(std::vector<double>{-1, -1, -1, -1, -1, -1});//compensate with -1
            }
            counter++;
        }
        //if data_last.size()<data_new.size() -> last element is -1 -> add -1
        while (data_last.size() > data_new.size()) {
            data_new.push_back(std::vector<double>{-1, -1, -1, -1, -1, -1});
        }
        //if data_next size is larger than data_last -> have to cover it
        if (data_next.size() > data_last.size()) {
            for (int j = 0; j < data_next.size(); j++) {
                if (std::find(assignment.begin(), assignment.end(), j) == assignment.end()) {//not found in assignment -> add data to data_latest
                    data_latest.push_back(data_next[j]);
                }
            }
        }
        //add data_latest to data_next
        if (!data_latest.empty()) {
            for (int j = 0; j < data_latest.size(); j++)
                data_new.push_back(data_latest[j]);
        }
        seqData_left.push_back(data_new);
    }
    std::cout << "seqData.size()=" << seqData_left.size() << std::endl;
    std::vector<std::vector<std::vector<double>>> seqData = permute(seqData_left);
    std::cout << "seqData.size()=" << seqData.size() << std::endl;
    //convert seqData shape from (seq,data,position) to (data,seq,position).
    checkSeqData(seqData, file_output);
    return seqData;
}

std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(lineStream, cell, ',')) {
            try {
                // Convert the string cell to double
                double value = std::stod(cell);
                row.push_back(value);
            }
            catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number in CSV: " << cell << std::endl;
                row.push_back(0); // Handle error: substitute with 0 or handle differently
            }
            catch (const std::out_of_range& e) {
                std::cerr << "Number out of range in CSV: " << cell << std::endl;
                row.push_back(0); // Handle error: substitute with 0 or handle differently
            }
        }

        data.push_back(row);
    }

    file.close();
    return data;
}

double calculateRMSE(std::vector<double>& box1, std::vector<double>& box2)
{
    /**
    * @brief calculate RMSE
    * @param[in] box1 {frame,label,left,top,width,height}
    */
    double centerX_1 = box1[2] + box1[4] / 2;
    double centerY_1 = box1[3] + box1[5] / 2;
    double centerX_2 = box2[2] + box2[4] / 2;
    double centerY_2 = box2[3] + box2[5] / 2;
    double dx = std::pow(centerX_1 - centerX_2, 2);
    double dy = std::pow(centerY_1 - centerY_2, 2);
    double rmse = std::pow(dx + dy, 0.5);
    return rmse;
}

void checkSeqData(std::vector<std::vector<std::vector<double>>>& dataLeft, std::string fileName)
{
    // Open the file for writing
    /* bbox data */
    std::ofstream outputFile(fileName);
    std::vector<int> frameIndexes;
    frameIndexes.reserve(2000);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    //std::cout << "file open :: save sequence data in csv file :: data size=" << dataLeft.size() << std::endl;
    for (int i = 0; i < dataLeft.size(); i++) //num objects
    {
        //std::cout << (i + 1) << "-th objects : " << std::endl;
        for (int j = 0; j < dataLeft[i].size(); j++) //num sequence
        {
            //std::cout << j << ":: frameIndex=" << dataLeft[i][j][0] << "class label=" << dataLeft[i][j][1] << ", left=" << dataLeft[i][j][2] << ", top=" << dataLeft[i][j][3] << " " << ", width=" << dataLeft[i][j][4] << ", height=" << dataLeft[i][j][5] << std::endl;
            auto it = std::find(frameIndexes.begin(), frameIndexes.end(), dataLeft[i][j][0]);
            /* new frame index */
            if (it == frameIndexes.end()) frameIndexes.push_back(dataLeft[i][j][0]);
            outputFile << dataLeft[i][j][0];
            outputFile << ",";
            outputFile << dataLeft[i][j][1];
            outputFile << ",";
            outputFile << dataLeft[i][j][2];
            outputFile << ",";
            outputFile << dataLeft[i][j][3];
            outputFile << ",";
            outputFile << dataLeft[i][j][4];
            outputFile << ",";
            outputFile << dataLeft[i][j][5];
            if (j != dataLeft[i].size() - 1)
            {
                outputFile << ",";
            }
        }
        outputFile << "\n";
        //std::cout << std::endl;
    }
    // close file
    outputFile.close();
}

std::vector<std::vector<std::vector<double>>> permute(std::vector<std::vector<std::vector<double>>>& a) {
   
    // Initialize the permuted vector with dimensions (D1, D0, D2)
    std::vector<std::vector<std::vector<double>>> b;
    if (!a.empty())
    {
        for (int i = 0; i < a.size(); i++) {//for each time step
            if (!a[i].empty()) {

                for (int j = 0; j < a[i].size(); j++) {//for each object
                    if (i == 0) {//first time
                        b.push_back({ a[i][j] });
                    }
                    else {
                        if (j < b.size()) {//existed tracker
                            b[j].push_back(a[i][j]);
                        }
                        else {//new data
                            b.push_back({ a[i][j] });
                        }
                    }
                }
            }
        }
    }

    return b;
}

void saveMatching(std::vector<std::vector<std::vector<int>>>& matchings, std::string& fileName) {
    /**
    * @brief save matching result
    * @param[in] matchings {n_idx_left, sequence,(frame,idx_right)}
    * @param[in] file_matching file name
    */

    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    for (int i = 0; i < matchings.size(); i++) {//for each idx_left
        if (matchings.empty()) {//no data
            outputFile << "\n";
        }
        else {
            for (int j = 0; j < matchings[i].size(); j++) {//for each sequence
                outputFile << matchings[i][j][0];//frame
                outputFile << ",";
                outputFile << matchings[i][j][1];//idx_right
                if (j != matchings[i].size() - 1)
                {
                    outputFile << ",";
                }
            }
            outputFile << "\n";
        }
    }
    // close file
    outputFile.close();
}