#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include "yolo_detector.hpp"

constexpr float YOLO_INPUT_SIZE = 640.0f;
constexpr int NUM_CLASSES = 80;
constexpr int NUM_CANDIDATES = 8400;


YOLODetector::YOLODetector(const std::string& model_path, const std::string& names_path, const std::unordered_set<int>& allowed_classes, float conf_thresh, float nms_thresh, bool verbose)
	: allowed_classes_(allowed_classes), conf_thresh_(conf_thresh), nms_thresh_(nms_thresh), verbose_(verbose) {

	// Reserve space for class names and colors
	class_names_.reserve(NUM_CLASSES);
	colors_.reserve(NUM_CLASSES);
	
	// Load model from the ONNX downloaded
	net_ = cv::dnn::readNetFromONNX(model_path);

	// Import (COCO) class names
	std::ifstream names_file;
	names_file.open(names_path);

	std::string name;
	int i = 0;
	while (std::getline(names_file, name))
		if (!name.empty()) {
			if (!allowed_classes_.empty() && allowed_classes_.count(i) == 0)
				class_names_.push_back(""); // Placeholder for skipped class to keep indices aligned; will be ignored in detect() since class_id won't be in allowed_classes_
			else {
				class_names_.push_back(name);
				if (verbose_)
					std::cout << "Loaded class " << i << ": " << name << std::endl;
			}
			i++;
		}

	names_file.close();

	// Initialize class colors (sample Hue on class id)
	for (int i = 0; i < class_names_.size(); i++) {
		if (class_names_[i].empty()) {
			colors_.push_back(cv::Scalar(0, 0, 0)); // Placeholder color for skipped class
			continue; // Skip placeholders for disallowed classes
		}

		int hue = (i * 180 / class_names_.size()) % 180;  // Set hue in range [0, 179] deterministically and evenly across classes
		cv::Scalar hsvColor(hue, 255, 255);  // Set Saturation and Value to max for visibility
		cv::Mat hsvMat(1, 1, CV_8UC3, hsvColor);  // Wrap color in a 1x1 3-Channels Matrix
		cv::Mat bgrMat;
		cv::cvtColor(hsvMat, bgrMat, cv::COLOR_HSV2BGR);  // Convert HSV to BGR
		cv::Scalar bgrScalar = (bgrMat.at<cv::Vec3b>(0, 0));
		colors_.push_back(bgrScalar);
		if (verbose_)
			std::cout << "Assigned color for class '" << class_names_[i] << "': " << bgrScalar << std::endl;
	}

}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image) {
	std::vector<Detection> detections;
	int out_size (NUM_CLASSES + 4);

	cv::Mat blob;
	// Preprocessing: normalize; resize so the smaller side is YOLO_INPUT_SIZE, keeping aspect ratio; stretch to YOLO_INPUT_SIZE x YOLO_INPUT_SIZE (YOLOv8 is robust enough to handle aspect-ratio distortion)
	cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), 0, true, false);

	// Inference
	net_.setInput(blob);
	cv::Mat inference = net_.forward(); // size (1, out_size, NUM_CANDIDATES) for yolov8n
	inference = inference.reshape(0, out_size); // reshape to (out_size, NUM_CANDIDATES) for easier indexing

	// Set scales
	float x_scale = (float)image.cols / YOLO_INPUT_SIZE;
	float y_scale = (float)image.rows / YOLO_INPUT_SIZE;

	// Iterate over all candidates
	for (int i = 0; i < NUM_CANDIDATES; i++) {

		// Find the best class score among indices 4..83
		float maxLabel = 0;
		int bestClass = 0;
		for (int j = 4; j < out_size; j++) {
			if (!allowed_classes_.empty() && allowed_classes_.count(j - 4) == 0)
				continue; // Skip if class is not in allowed_classes_

			float label = inference.at<float>(j, i);
			if (label > maxLabel) {
				maxLabel = label;
				bestClass = j - 4;
			}
		}

		// If best score < conf_thresh_, skip
		if (maxLabel < conf_thresh_)
			continue;

		if (verbose_)
			std::cout << "Candidate " << i << ": Best class = " << class_names_[bestClass] << " with confidence = " << maxLabel << std::endl;

		// Convert cx,cy,w,h to cv::Rect, scale by x_scale/y_scale
		int cx = inference.at<float>(0, i);
		int cy = inference.at<float>(1, i);
		int w = inference.at<float>(2, i);
		int h = inference.at<float>(3, i);

		cx *= x_scale;
		cy *= y_scale;
		w *= x_scale;
		h *= y_scale;

		cv::Rect bbox((int)(cx - w / 2), (int)(cy - h / 2), (int)w, (int)h);

		// Push a Detection into detections
		Detection d{ bestClass, class_names_[bestClass], maxLabel, bbox };
		detections.push_back(d);
	}
	return detections;
}

void YOLODetector::draw(cv::Mat& image, const std::vector<Detection>& detections) const{
	for (const Detection& d : detections) {
		cv::Scalar color = colors_.at(d.class_id);
		cv::rectangle(image, d.bbox, color, 2);
		cv::putText(image, d.class_label, d.bbox.tl(), cv::FONT_HERSHEY_PLAIN, 2, color, 1);
	}
}