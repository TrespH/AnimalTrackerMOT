#include <opencv2/opencv.hpp>
#include <iostream>
#include "yolo_detector.hpp"

class KalmanTracker {

public:
	static int next_id_;
	int id;
	int missed_frames;

	KalmanTracker(const Detection& d);

	// Update the full internal state: x and P
	// The caller (Hungarian algorithm) only needs the bounding box to compute IoU against YOLO detections
	cv::Rect predict();


	// Correct the measurement H
	void update(const Detection& d);

private:
	cv::Mat x;  // State vector 8x1 (belief
	cv::Mat P;  // Covariance 8x8 (uncertainty)
	cv::Mat F;  // State 8x8
	cv::Mat Q;  // State Noise 8x8

	cv::Mat z;  // Measurement
	cv::Mat K;  // Kalman gain 8x4
	cv::Mat H;  // Measurement 4x8
	cv::Mat R;  // Measurement Noise 4x4

};