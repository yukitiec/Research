#include "prediction.h"

void Prediction::main()
{
    /**
    * @brief predict trajectory based on 3D sequential data
    */
    Utility utPre;//constructor
    double depth_target;
    /*
    while (true)
    {
        if (!data_3d.empty()) break;//get data_3d from triangulation.h
    }
    std::cout << "start predicting target positions" << std::endl;

    int counterIteration = 0;
    int counterFinish = 0;
    int counter = 0;//counter before delete new data
    std::vector<std::vector<std::vector<double>>> targetPoint(numObjects);//{num of objects, sequence, {targetFrame, targetX, targetY, targetZ}
    while (true) // continue until finish
    {
        counterIteration++;
        if (queueFrame.empty() && queue_tri2predict.empty())
        {
            if (counterFinish == 10) break;
            counterFinish++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "By finish : remain count is " << (10 - counterFinish) << std::endl;
            continue;
        }
        else
        {
            counterFinish = 0;
            /* new detection data available *//*
            if (!queue_tri2predict.empty() && !queue3DData.empty())
            {
                /* for predict *//*
                counter = 0;//reset
                std::vector<std::vector<std::vector<double>>> data_3d = queue3DData.front();
                queue3DData.pop();
                queue_tri2predict.pop();
                std::cout << "Prediction :: get 3d positions" << std::endl;
                auto start = std::chrono::high_resolution_clock::now();
                std::cout << "start 3d prediction" << std::endl;
                int it = 0;
                for (std::vector<std::vector<double>>& data : data_3d)
                {
                    if (data.size() >= 3)
                    {
                        predictTargets(it, depth_target,data, targetPoint);
                    }
                    it++;
                }

            }
            /* at least one data can't be available -> delete data *//*
            else
            {
                if (!queue_tri2predict.empty() || !queue3DData.empty())
                {
                    //std::cout << "both data can't be availble :: left " << !queueTriangulation_left.empty() << ", right=" << queueTriangulation_right.empty() << std::endl;
                    if ((!queue_tri2predict.empty() || !queue3DData.empty()) && counter == 100)
                    {
                        if (!queue_tri2predict.empty()) queue_tri2predict.pop();
                        if (!queue3DData.empty()) queue3DData.pop();
                        counter = 0;
                    }
                    else
                    {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                        counter++;
                    }
                }
            }
        }
    }
    std::cout << "***target position***" << std::endl;
    utPre.save3d(targetPoint, file_target);
    */
    
}

void Prediction::predictTargets(int& index, double& depth_target,std::vector<std::vector<double>>& data, std::vector<std::vector<std::vector<double>>>& targets3D)
{
    /**
    * @brief predict 3D trajectory
    * @param[in] index object index whose update is over 3
    * @param[out] depth_target target depth
    * @param[in] data sequential data. shape is like {sequence, {x,y,z}}
    * @param[out] targets3D storage for target 3D points. shape is like {n_objects, sequence, {x,y,z}}
    */
    /* trajectoryPrediction */
    linearRegression(data, coefX);
    linearRegressionZ(data, coefZ);
    curveFitting(data, coefY);
    /* objects move */
    if (coefZ[0] < 0) // moving forward to camera
    {
        double frameTarget = (double)((int)((depth_target - coefZ[1]) / coefZ[0]));
        double xTarget = (double)(coefX[0] * frameTarget + coefX[1]);
        double yTarget = (double)(coefY[0] * frameTarget * frameTarget + coefY[1] * frameTarget + coefY[2]);
        //not empty
        if (!targets3D.empty()) targets3D.at(index).push_back({ frameTarget, xTarget, yTarget, depth_target }); // push_back target position
        //empty
        else targets3D.at(index) = { {frameTarget, xTarget, yTarget, depth_target} };
        //std::cout << "target is : ( frameTarget :  " << frameTarget << ", xTarget : " << xTarget << ", yTarget : " << yTarget << ", depthTarget : " << depth_target << std::endl;
    }
}

void Prediction::linearRegression(std::vector<std::vector<double>>& data, std::vector<double>& result_x)
{
    /*
     * @brief linear regression
     * y = ax + b
     * a = (sigma(xy)-n*mean_x*mean_y)/(sigma(x^2)-n*mean_x^2)
     * b = mean_y - a*mean_x
     * 
     * @param[in] data sequential data. shape is like {sequence, {frame,x,y,z}}
     * @param[in] result_x(std::vector<double>&) : vector for saving result x
     */
    double sumt = 0, sumx = 0, sumtx = 0, sumtt = 0; // for calculating coefficients
    double mean_t, mean_x;
    int length = data.size(); // length of data

    for (int i = 1; i < (int)N_POINTS_PREDICT + 1; i++)
    {
        sumt += data[length - i][0];
        sumx += data[length - i][1];
        sumtx += data[length - i][0] * data[length - i][1];
        sumtt += data[length - i][0] * data[length - i][0];
    }
    double slope_x, intercept_x;
    if (std::abs(N_POINTS_PREDICT*sumtt-sumt*sumt) > 0.0001)
    {
        slope_x = (N_POINTS_PREDICT*sumtx-sumt*sumx)/(N_POINTS_PREDICT*sumtt-sumt*sumt);
        intercept_x = (sumx-slope_x*sumt)/N_POINTS_PREDICT;
    }
    else
    {
        slope_x = 0;
        intercept_x = 0;
    }
    result_x = { slope_x, intercept_x };
    //std::cout << "\n\nX :: The best fit value of curve is : x = " << slope_x << " t + " << intercept_x << ".\n\n"<< std::endl;
}

void Prediction::linearRegressionZ(std::vector<std::vector<double>>& data, std::vector<double>& result_z)
{
    /**
     * @brief linear regression
     * y = ax + b
     * a = (sigma(xy)-n*mean_x*mean_y)/(sigma(x^2)-n*mean_x^2)
     * b = mean_y - a*mean_x
     * 
     * @param[in] data sequential data {sequence, {frame, x,y,z}}
     * @param[in] result_x(std::vector<double>&) : vector for saving result x
     */
    double sumt = 0, sumz = 0, sumtt = 0, sumtz = 0; // for calculating coefficients
    double mean_t, mean_z;
    int length = data.size(); // length of data

    for (int i = 1; i < (int)N_POINTS_PREDICT + 1; i++)
    {
        sumt += data[length - i][0];
        sumz += data[length - i][3];
        sumtt += data[length - i][0] * data[length - i][0];
        sumtz += data[length - i][0] * data[length - i][3];
    }
    //std::cout << "Linear regression" << std::endl;
    mean_t = static_cast<double>(sumt) / static_cast<double>(N_POINTS_PREDICT);
    mean_z = static_cast<double>(sumz) / static_cast<double>(N_POINTS_PREDICT);
    double slope_z, intercept_z;
    if (std::abs(sumtt - N_POINTS_PREDICT * mean_t * mean_t) > 0.0001)
    {
        slope_z = (N_POINTS_PREDICT * sumtz - sumt * sumz) / (N_POINTS_PREDICT * sumtt - sumt * sumt);
        intercept_z = (sumz - slope_z * sumt) / N_POINTS_PREDICT;
    }
    else
    {
        slope_z = 0;
        intercept_z = 0;
    }
    result_z = { slope_z, intercept_z };
    //std::cout << "\n\nZ :: The best fit value of curve is : z = " << slope_z << " t + " << intercept_z << ".\n\n"<< std::endl;
}

void Prediction::curveFitting(std::vector<std::vector<double>>& data, std::vector<double>& result)
{
    /**
     * @brief curve fitting with parabora
     * y = a*x^2+b*x+c
     *
     * 
     * @param[in] data sequential data {sequence, (frameIndex,x,y,z)}
     * @param[out] result(std::vector<double>&) : vector for saving result
     */

     // argments analysis
    int length = data.size(); // length of data
    // Initialize sums
    double sumX = 0, sumY = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
    double sumXY = 0, sumX2Y = 0;

    for (int i = 1; i < (int)N_POINTS_PREDICT+1; ++i) {
        double x = data[length-i][0];
        double y = data[length-i][2];
        double x2 = x * x;
        double x3 = x2 * x;
        double x4 = x3 * x;

        sumX += x;
        sumY += y;
        sumX2 += x2;
        sumX3 += x3;
        sumX4 += x4;
        sumXY += x * y;
        sumX2Y += x2 * y;
    }

    // Construct matrices
    Eigen::Matrix3d A;
    Eigen::Vector3d B;

    A << N_POINTS_PREDICT, sumX, sumX2,
        sumX, sumX2, sumX3,
        sumX2, sumX3, sumX4;

    B << sumY,
        sumXY,
        sumX2Y;

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> dec(A);
    Eigen::Vector3d coeffs = dec.solve(B);
    //Eigen::Vector3d coeffs = A.colPivHouseholderQr().solve(B);
    double a, b, c;
    //std::cout << "dec.rank()=" << dec.rank() << ", dec.info()=" << (dec.info()==Eigen::Success) << std::endl;
    //if (dec.rank() < 3 || dec.info() != Eigen::Success) {//check if the matrix is of full rank and calculation is successful
    //    c = 0;
    //    b = 0;
    //    a = 0;
    //}
    //else {
    c = coeffs[0];
    b = coeffs[1];
    a = coeffs[2];
    if (std::abs(c) < 1e-6) c = 0.0;
    if (std::abs(b) < 1e-6) b = 0.0;
    if (std::abs(a) < 1e-6) a = 0.0;
    //}

    result = { a, b, c };

    //std::cout << "y = " << a << "x^2 + " << b << "x + " << c << std::endl;
}

void Prediction::trajectoryPredict2D(std::vector<std::vector<std::vector<double>>>& dataLeft, std::vector<std::vector<double>>& coefficientsX, std::vector<std::vector<double>>& coefficientsY, std::vector<int>& classesLatest)
{
    /**
    * @brief predict 2D trajectory. This is for matching objects in 2 cameras.
    * @param[in] dataLeft sequential data. shape is like {n_objects, sequence, {frame,label,left,top,width,height}}
    * @param[out] coefficientsX coefficients of trajectory in x axis. shape is like {n_objects, (a,b)}. y = ax+b
    * @param[out] coefficientsY coefficients of trajectory in y axis. shape is like {n_objects, (a,b,c)}. y = ax^2+bx+c
    * @param[out] classesLatest list for class indexes of latest data
    */

    int counterData = 0;
    for (const std::vector<std::vector<double>>& data : dataLeft)
    {
        /* get 3 and more time-step datas -> can predict trajectory */
        if (data.size() >= 3)
        {
            /* get latest 3 time-step data */
            std::vector<std::vector<double>> tempData;
            std::vector<double> coefX, coefY;
            // Use reverse iterators to access the last three elements
            auto rbegin = data.rbegin(); // Iterator to the last element
            auto rend = data.rend();     // Iterator one past the end
            /* Here data is latest to old -> but not have bad effect to trajectory prediction */
            for (auto it = rbegin; it != rend && std::distance(rend, it) < 3; ++it)
            {
                const std::vector<double>& element = *it;
                tempData.push_back(element);
            }
            /* trajectory prediction in X and Y */
            linearRegression(tempData, coefX);
            curveFitting(tempData, coefY);
            coefficientsX.push_back(coefX);
            coefficientsY.push_back(coefY);
        }
        /* get less than 3 data -> can't predict trajectory -> x : have to make the size equal to classesLatest
         *  -> add specific value to coefficientsX and coefficientsY, not change the size of classesLatest for maintaining size consisitency between dataLeft and data Right
         */
        else
        {
            coefficientsX.push_back({ 0.0, 0.0 });
            coefficientsY.push_back({ 0.0, 0.0, 0.0 });
            // classesLatest.erase(classesLatest.begin() + counterData); //erase class
            // counterData++;
            /* can't predict trajectory */
        }
    }
}

double Prediction::calculateME(std::vector<double>& coefXLeft, std::vector<double>& coefYLeft, std::vector<double>& coefXRight, std::vector<double>& coefYRight)
{
    double me = 0.0; // mean error
    for (int i = 0; i < coefYLeft.size(); i++)
    {
        me = me + (coefYLeft[i] - coefYRight[i]);
    }
    me = me + coefXLeft[0] - coefXRight[0];
    return me;
}

void Prediction::dataMatching(std::vector<std::vector<double>>& coefficientsXLeft, std::vector<std::vector<double>>& coefficientsXRight,
    std::vector<std::vector<double>>& coefficientsYLeft, std::vector<std::vector<double>>& coefficientsYRight,
    std::vector<int>& classesLatestLeft, std::vector<int>& classesLatestRight,
    std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<std::vector<int>>>& dataRight,
    std::vector<std::vector<std::vector<std::vector<int>>>>& dataFor3D)
{
    double minVal = 20;
    int minIndexRight;
    if (!coefficientsXLeft.empty() && !coefficientsXRight.empty())
    {
        /* calculate metrics based on left img data */
        for (int i = 0; i < coefficientsXLeft.size(); i++)
        {
            /* deal with moving objects -> at least one coefficient should be more than 0 */
            if (coefficientsYLeft[i][0] != 0)
            {
                for (int j = 0; j < coefficientsXRight.size(); j++)
                {
                    /* deal with moving objects -> at least one coefficient should be more than 0 */
                    if (coefficientsYRight[i][0] != 0)
                    {
                        /* if class label is same */
                        if (classesLatestLeft[i] == classesLatestRight[j])
                        {
                            /* calculate metrics */
                            double me = calculateME(coefficientsXLeft[i], coefficientsYLeft[i], coefficientsXRight[j], coefficientsXRight[j]);
                            /* minimum value is updated */
                            if (me < minVal)
                            {
                                minVal = me;
                                minIndexRight = j; // most reliable matching index in Right img
                            }
                        }
                        /* maybe fixed objects detected */
                        else
                        {
                            /* ignore */
                        }
                    }
                }
                /* matcing object found */
                if (minVal < 20)
                {
                    dataFor3D.push_back({ dataLeft[i], dataRight[minIndexRight] }); // match objects and push_back to dataFor3D
                }
            }
            /* maybe fixed objects detected */
            else
            {
                /* ignore */
            }
        }
    }
}

void Prediction::predict3DTargets(double& depth_target,std::vector<std::vector<std::vector<std::vector<double>>>>& datasFor3D, std::vector<std::vector<double>>& targets3D)
{
    int indexL, indexR, xLeft, xRight, yLeft, yRight;
    double fX = cameraMatrix.at<double>(0, 0);
    double fY = cameraMatrix.at<double>(1, 1);
    double fSkew = cameraMatrix.at<double>(0, 1);
    double oX = cameraMatrix.at<double>(0, 2);
    double oY = cameraMatrix.at<double>(1, 2);
    /* iteration of calculating 3d position for each matched objects */
    for (std::vector<std::vector<std::vector<double>>>& dataFor3D : datasFor3D)
    {
        std::vector<std::vector<double>> dataL = dataFor3D[0];
        std::vector<std::vector<double>> dataR = dataFor3D[1];
        /* get 3 and more time-step datas -> calculate 3D position */
        int numDataL = dataL.size();
        int numDataR = dataR.size();
        std::vector<std::vector<double>> data3D; //[mm]
        // calculate 3D position
        int counter = 0; // counter for counting matching frame index
        int counterIteration = 0;
        bool boolPredict = false; // if 3 datas are available
        while (counterIteration < std::min(numDataL, numDataR))
        {
            counterIteration++;
            if (counter > 3)
            {
                boolPredict = true;
                break;
            }
            indexL = dataL[numDataL - counter][0];
            indexR = dataR[numDataR - counter][0];
            if (indexL == indexR)
            {
                xLeft = dataL[numDataL - counter][1];
                xRight = dataR[numDataR - counter][1];
                yLeft = dataL[numDataL - counter][2];
                yRight = dataR[numDataR - counter][2];
                double disparity = (double)(xLeft - xRight);
                double X = (double)(BASELINE / disparity) * (xLeft - oX - (fSkew / fY) * (yLeft - oY));
                double Y = (double)(BASELINE * (fX / fY) * (yLeft - oY) / disparity);
                double Z = (double)(fX * BASELINE / disparity);
                data3D.push_back({ (double)indexL, X, Y, Z });
                counter++;
            }
        }
        if (boolPredict)
        {
            /* trajectoryPrediction */
            std::vector<double> coefX, coefY, coefZ;
            linearRegression(data3D, coefX);
            linearRegressionZ(data3D, coefZ);
            curveFitting(data3D, coefY);
            /* objects move */
            if (coefZ[0] < 0) // moving forward to camera
            {
                double frameTarget = (double)((depth_target - coefZ[1]) / coefZ[0]);
                double xTarget = (double)(coefX[0] * frameTarget + coefX[1]);
                double yTarget = (double)(coefY[0] * frameTarget * frameTarget + coefY[1] * frameTarget + coefY[2]);
                targets3D.push_back({ frameTarget, xTarget, yTarget, depth_target }); // push_back target position
                std::cout << "target is : ( frameTarget :  " << frameTarget << ", xTarget : " << xTarget << ", yTarget : " << yTarget << ", depthTarget : " << depth_target << std::endl;
            }
        }
    }
}