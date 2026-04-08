#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <unordered_set>

struct Detection {
	int class_id;
	std::string class_label;
	float confidence;
	cv::Rect bbox;
};

class YOLODetector {
public:
	YOLODetector(const std::string& model_path, const std::string& names_path, const std::unordered_set<int>& allowed_classes = {}, float conf_thresh = 0.45f, float nms_thresh = 0.50f, bool verbose = false);

	std::vector<Detection> detect(const cv::Mat& image);

	void draw(cv::Mat& image, const std::vector<Detection>& detections) const;

private:
	cv::dnn::Net net_;
	std::vector<std::string> class_names_;
	std::unordered_set<int> allowed_classes_;
	std::vector<cv::Scalar> colors_;
	float conf_thresh_;
	float nms_thresh_;
	bool verbose_;
};