#include <opencv2/opencv.hpp>
#include <iostream>
#include "yolo_detector.hpp"

class KalmanTracker {

public:
	static int next_id_;
	int id;
	int missed_frames;
	Detection last_d;

	KalmanTracker(const Detection& d);

	// Update the full internal state: x and P
	// The caller (Hungarian algorithm) only needs the bounding box to compute IoU against YOLO detections
	cv::Rect predict();

	// Correct the measurement H
	void update(const Detection& d);

private:
	cv::Mat x;  // State Vector 8x1 (belief)
	cv::Mat P;  // Covariance 8x8 (uncertainty)
	cv::Mat F;  // State Transition 8x8
	cv::Mat Q;  // Process Noise 8x8

	cv::Mat z;  // Measurement Vector 4x1
	cv::Mat K;  // Kalman Gain 8x4
	cv::Mat H;  // Measurement Tansition 4x8
	cv::Mat R;  // Measurement Noise 4x4

};