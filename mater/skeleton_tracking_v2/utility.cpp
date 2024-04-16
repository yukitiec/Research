#include "stdafx.h"
#include "utility.h"

void Utility::pushImg(std::array<cv::Mat1b, 2>& frame, int& frameIndex)
{
    // std::unique_lock<std::mutex> lock(mtxImg);
    queueFrame.push(frame);
    queueFrameIndex.push(frameIndex);
}

void Utility::getImages(std::array<cv::Mat1b, 2>& frame, int& frameIndex)
{
    //std::unique_lock<std::mutex> lock(mtxImg); // exclude other accesses
    frame = queueFrame.front();
    // cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    frameIndex = queueFrameIndex.front();
    queueFrame.pop();
    queueFrameIndex.pop();
}

void Utility::saveYolo(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file)
{
    // Open the file for writing
    std::ofstream outputFile(file);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    /* write posSaver data to csv file */
    //std::cout << "estimated position :: YOLO :: " << std::endl;
    /*sequence*/
    for (int i = 0; i < posSaver.size(); i++)
    {
        //std::cout << i << "-th sequence data ::: " << std::endl;
        /*num of humans*/
        for (int j = 0; j < posSaver[i].size(); j++)
        {
            //std::cout << j << "-th human detection:::" << std::endl;
            /*num of joints*/
            for (int k = 0; k < posSaver[i][j].size(); k++)
            {
                //std::cout << k << "-th joint :: frameIndex=" << posSaver[i][j][k][0] << ", xCenter=" << posSaver[i][j][k][1] << ", yCenter=" << posSaver[i][j][k][2] << std::endl;
                outputFile << posSaver[i][j][k][0];
                outputFile << ",";
                outputFile << posSaver[i][j][k][1];
                outputFile << ",";
                outputFile << posSaver[i][j][k][2];
                outputFile << ",";
                outputFile << posSaver[i][j][k][3];
                outputFile << ",";
                outputFile << posSaver[i][j][k][4];
                if (k != posSaver[i][j].size() - 1)
                {
                    outputFile << ",";
                }
            }
            outputFile << "\n";
        }
    }
    // Close the file
    outputFile.close();
}

void Utility::save(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file)
{
    // Open the file for writing
    std::ofstream outputFile(file);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    //std::cout << "estimated position :: Optical Flow :: " << std::endl;
    /*sequence*/
    for (int i = 0; i < posSaver.size(); i++)
    {
        //std::cout << i << "-th sequence data ::: " << std::endl;
        /*num of humans*/
        for (int j = 0; j < posSaver[i].size(); j++)
        {
            //std::cout << j << "-th human detection:::" << std::endl;
            /*num of joints*/
            for (int k = 0; k < posSaver[i][j].size(); k++)
            {
                //std::cout << k << "-th joint :: frameIndex=" << posSaver[i][j][k][0] << ", xCenter=" << posSaver[i][j][k][1] << ", yCenter=" << posSaver[i][j][k][2] << std::endl;
                outputFile << posSaver[i][j][k][0];
                outputFile << ",";
                outputFile << posSaver[i][j][k][1];
                outputFile << ",";
                outputFile << posSaver[i][j][k][2];
                outputFile << ",";
                outputFile << posSaver[i][j][k][3];
                outputFile << ",";
                outputFile << posSaver[i][j][k][4];
                if (k != posSaver[i][j].size() - 1)
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

void Utility::save3d(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file)
{
    // Open the file for writing
    std::ofstream outputFile(file);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    //std::cout << "estimated position :: Optical Flow :: " << std::endl;
    /*sequence*/
    for (int i = 0; i < posSaver.size(); i++)
    {
        //std::cout << i << "-th sequence data ::: " << std::endl;
        /*num of humans*/
        for (int j = 0; j < posSaver[i].size(); j++)
        {
            //std::cout << j << "-th human detection:::" << std::endl;
            /*num of joints*/
            for (int k = 0; k < posSaver[i][j].size(); k++)
            {
                //std::cout << k << "-th joint :: frameIndex=" << posSaver[i][j][k][0] << ", xCenter=" << posSaver[i][j][k][1] << ", yCenter=" << posSaver[i][j][k][2] << std::endl;
                outputFile << posSaver[i][j][k][0];
                outputFile << ",";
                outputFile << posSaver[i][j][k][1];
                outputFile << ",";
                outputFile << posSaver[i][j][k][2];
                outputFile << ",";
                outputFile << posSaver[i][j][k][3];
                if (k != posSaver[i][j].size() - 1)
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