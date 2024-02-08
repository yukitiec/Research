#pragma once

#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include "stdafx.h"

extern const int COUNTER_LOST;
extern const int FPS;

class KalmanFilter2D {
private:
    Eigen::Vector<double, 6> state_; // State estimate [x, y, vx, vy, ax, ay]
    Eigen::Matrix<double, 6, 6> P_;     // Estimate error covariance
    Eigen::Matrix<double, 6, 6> Q_;     // Process noise covariance
    Eigen::Matrix<double, 2, 2> R_;     // Measurement noise covariance
    Eigen::Matrix<double, 6, 6> A_;     // State transition matrix
    Eigen::Matrix<double, 2, 6> H_;     // Measurement matrix
    Eigen::Matrix<double, 6, 2> K_;     // Kalman gain
    int counter_notUpdate = 0; //count not update time
    double dt_ = 3.0 / (double)FPS; // frame interval between Yolo inference
public:
    int counter_update = 0; // number of update

    KalmanFilter2D(double initial_x, double initial_y, double initial_vx, double initial_vy, double init_ax, double init_ay,
        double process_noise_pos, double process_noise_vel, double process_noise_acc, double measurement_noise) {
        // Initial state: [x, y, vx, vy, ax, ay]
        state_ << initial_x, initial_y, initial_vx, initial_vy, init_ax, init_ay; //column vector

        // Initial estimate error covariance
        P_ = Eigen::MatrixXd::Identity(6, 6);

        // Process noise covariance
        Q_ << process_noise_pos, 0, 0, 0, 0, 0,
            0, process_noise_pos, 0, 0, 0, 0,
            0, 0, process_noise_vel, 0, 0, 0,
            0, 0, 0, process_noise_vel, 0, 0,
            0, 0, 0, 0, process_noise_acc, 0,
            0, 0, 0, 0, 0, process_noise_acc;

        // Measurement noise covariance
        R_ = Eigen::MatrixXd::Identity(2, 2) * measurement_noise;
    }

    // Prediction step
    void predict(Eigen::Vector<double, 6>& prediction, double dframe, std::vector<std::vector<double>>& seqData) {
        dt_ = dframe / (double)FPS;
        // State transition matrix A for constant acceleration model
        A_ << 1, 0, dt_, 0, 0.5 * dt_ * dt_, 0,
            0, 1, 0, dt_, 0, 0.5 * dt_,
            0, 0, 1, 0, dt_, 0,
            0, 0, 0, 1, 0, dt_,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1;

        // Predict the next state
        state_ = A_ * state_;
        prediction = state_;

        // Update the estimate error covariance
        P_ = A_ * P_ * A_.transpose() + Q_;
        counter_notUpdate++;
        if (counter_notUpdate == COUNTER_LOST)
        {
            seqData.push_back({ -1,-1,-1,-1,-1,-1 });
            prediction << -1, -1, -1, -1, -1, -1;
            counter_update = 0;
        }

    }

    // Update step
    void update(const Eigen::Vector2d& measurement) {
        // Measurement matrix H (we are measuring the position in x and y)
        H_ << 1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0; //2*6 vector

        // Kalman gain
        K_ = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + R_).inverse(); //6*2 matrix

        // Update the state estimate
        state_ = state_ + K_ * (measurement - H_ * state_); //6*1 + (6*2)*(2*1) (2*6)*(6*1)

        // Update the estimate error covariance
        P_ = (Eigen::MatrixXd::Identity(6, 6) - K_ * H_) * P_;
        counter_notUpdate = 0;
        counter_update++; //increment update counter
    }

    Eigen::Vector<double, 6> getState() const {
        return state_;
    }
};

#endif
