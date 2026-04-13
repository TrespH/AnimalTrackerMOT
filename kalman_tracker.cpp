#include <opencv2/opencv.hpp>
#include <iostream>
#include "yolo_detector.hpp"
#include "kalman_tracker.hpp"


int KalmanTracker::next_id_ = 0;


KalmanTracker::KalmanTracker(const Detection& d) : id(++next_id_), missed_frames(0) {

	// Initialize status x
	x = cv::Mat(8, 1, CV_32F);
	x.at<float>(0) = d.bbox.x + d.bbox.width / 2.0f; // cx
	x.at<float>(1) = d.bbox.y + d.bbox.height / 2.0f; // cy
	x.at<float>(2) = d.bbox.width; // w
	x.at<float>(3) = d.bbox.height; // h
	x.at<float>(4) = 0.0f; // vx
	x.at<float>(5) = 0.0f; // vy
	x.at<float>(6) = 0.0f; // vw
	x.at<float>(7) = 0.0f; // vh

	// Initialize covariance P
	// The standard SORT initialization is:
	// High uncertainty on position (trust YOLO somewhat but not completely): diagonal value ~10
	// Very high uncertainty on velocity (no velocity measurement yet): diagonal value ~100 or 1000
	P = cv::Mat::eye(8, 8, CV_32F);
	for (int i = 0; i < 4; i++) P.at<float>(i, i) = 10.0f;
	for (int i = 4; i < 8; i++) P.at<float>(i, i) = 1000.0f;

	// Initalize  F (identity block for positions and velocities, and +1 where position couples to velocity (top-right block)
	F = cv::Mat::eye(8, 8, CV_32F);
	for (int i = 0; i < 4; i++) F.at<float>(i, i + 4) = 1.0f;

	// Initialize measurement H (identity block on the position entries)
	H = cv::Mat::zeros(4, 8, CV_32F);
	for (int i = 0; i < 4; i++) H.at<float>(i, i) = 1.0f;

	// Initialize Q (small process noise) and R (moderate mesurement noise)
	Q = 0.01 * cv::Mat::eye(8, 8, CV_32F);
	R = 10 * cv::Mat::eye(4, 4, CV_32F);
}


cv::Rect KalmanTracker::predict() {
	// Every time we predict without an update, the object has gone one more frame unmatched
	missed_frames++;

	// Predict step
	x = F * x;
	P = F * P * F.t() + Q;

	// Convert state into bbox to return
	int cx = x.at<float>(0);
	int cy = x.at<float>(1);
	int w = x.at<float>(2);
	int h = x.at<float>(3);

	cv::Rect bbox((cx - w / 2), (cy - h / 2), w, h);
	return bbox;
}


void KalmanTracker::update(const Detection& d) {

	// Reset missed number of frames, as we just received a measurement
	missed_frames = 0;

	// Initialize measurement z from last detection
	z = cv::Mat(4, 1, CV_32F);
	z.at<float>(0) = d.bbox.x + d.bbox.width / 2.0f; // cx
	z.at<float>(1) = d.bbox.y + d.bbox.height / 2.0f; // cy
	z.at<float>(2) = d.bbox.width; // w
	z.at<float>(3) = d.bbox.height; // h

	// Kalman Gain (8x4)
	K = cv::Mat(8, 4, CV_32F);
	K = P * H.t() * (H * P * H.t() + R).inv();

	// Correct state and covariance
	x = x + K * (z - H * x);
	P = (cv::Mat::eye(8, 8, CV_32F) - K * H) * P;
}
