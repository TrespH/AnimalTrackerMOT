#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include "yolo_detector.hpp"

constexpr float YOLO_INPUT_SIZE = 640.0f;
constexpr int NUM_CLASSES = 80;
constexpr int NUM_CANDIDATES = 8400;


YOLODetector::YOLODetector(const std::string& model_path, const std::string& names_path, const std::unordered_set<int>& allowed_classes, float conf_thresh, float nms_thresh, bool use_gpu, bool verbose)
	: allowed_classes_(allowed_classes), conf_thresh_(conf_thresh), nms_thresh_(nms_thresh), verbose_(verbose) {

	// Reserve space for class names and colors
	class_names_.reserve(NUM_CLASSES);
	colors_.reserve(NUM_CLASSES);

	// Load model from the ONNX downloaded
	net_ = cv::dnn::readNetFromONNX(model_path);

	// Set CUDA environment if requested
	if (use_gpu) {
		net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA); // DNN_TARGET_CUDA or DNN_TARGET_CUDA_FP16
		net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); // DNN_TARGET_CUDA or DNN_TARGET_CUDA_FP16
	}
	else {
		net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}

	// Import (COCO) class names
	std::ifstream names_file;
	names_file.open(names_path);

	std::string name;
	int i = 0;
	while (std::getline(names_file, name))
		if (!name.empty()) {
			if (allowed_classes_.count(i) != 0) {
				class_names_.push_back(name);
				if (verbose_)
					std::cout << "Loaded class: " << name << std::endl;
			}
			else class_names_.push_back("");
			i++;
		}

	names_file.close();

	// Initialize class colors (sample Hue on class id)
	for (int i = 0; i < class_names_.size(); i++) {
		if (allowed_classes_.count(i) != 0) {
			int hue = (i * 180 / allowed_classes_.size()) % 180;  // Set hue in range [0, 179] deterministically and evenly across classes
			cv::Scalar hsvColor(hue, 255, 255);  // Set Saturation and Value to max for visibility
			cv::Mat hsvMat(1, 1, CV_8UC3, hsvColor);  // Wrap color in a 1x1 3-Channels Matrix
			cv::Mat bgrMat;
			cv::cvtColor(hsvMat, bgrMat, cv::COLOR_HSV2BGR);  // Convert HSV to BGR
			cv::Scalar bgrScalar = (bgrMat.at<cv::Vec3b>(0, 0));
			colors_.push_back(bgrScalar);
			if (verbose_)
				std::cout << "Assigned color for class '" << class_names_[i] << "': " << bgrScalar << std::endl;
		}
		else colors_.push_back(cv::Scalar());
	}

}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image) {

	// Intermediate results to later be filtered by NMS, and stored in detections
	std::vector<cv::Rect> bboxes;
	std::vector<float> scores;
	std::vector<int> class_indices;
	std::vector<Detection> detections;

	int out_size = NUM_CLASSES + 4;  // consider the returned (cx, cy, w, h) data

	cv::Mat blob;
	// Preprocessing: normalize; resize so the smaller side is YOLO_INPUT_SIZE, keeping aspect ratio; stretch to YOLO_INPUT_SIZE x YOLO_INPUT_SIZE (YOLOv8 is robust enough to handle aspect-ratio distortion)
	cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), 0, true, true);

	// Inference
	net_.setInput(blob);
	cv::Mat inference = net_.forward(); // size (1, out_size, NUM_CANDIDATES) for yolov8n
	inference = inference.reshape(0, out_size); // reshape to (out_size, NUM_CANDIDATES) for easier indexing

	// Set scales
	float x_scale = (float)image.cols / YOLO_INPUT_SIZE;
	float y_scale = (float)image.rows / YOLO_INPUT_SIZE;

	// Iterate over all candidates
	for (int i = 0; i < NUM_CANDIDATES; i++) {

		// Find the best class score among indices 4..NUM_CLASSES-1
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

		// Skip predictions below threshold
		if (maxLabel < conf_thresh_)
			continue;

		// Convert cx,cy,w,h to cv::Rect, scale by x_scale/y_scale
		float cx = inference.at<float>(0, i);
		float cy = inference.at<float>(1, i);
		float w = inference.at<float>(2, i);
		float h = inference.at<float>(3, i);

		cx *= x_scale;
		cy *= y_scale;
		w *= x_scale;
		h *= y_scale;

		cv::Rect bbox((cx - w / 2), (cy - h / 2), w, h);

		// Save intermediate result
		bboxes.push_back(bbox);
		scores.push_back(maxLabel);
		class_indices.push_back(bestClass);
	}

	// Apply Non-Maximum Suppression to intermediate results
	std::vector<int> indices;
	cv::dnn::NMSBoxes(bboxes, scores, conf_thresh_, nms_thresh_, indices);

	// Populate detections to return
	for (int i : indices) {
		Detection d{ class_indices[i], class_names_[class_indices[i]], scores[i], bboxes[i] };
		detections.push_back(d);
		if (verbose_)
			std::cout << "Candidate " << i << ": Best class = " << class_names_[class_indices[i]] << " with confidence = " << scores[i] << std::endl;
	}

	return detections;
}

void YOLODetector::draw(cv::Mat& image, const Detection& d, const int id) const {
	// Build label, text point, color
	std::string label =
		(id == -1) ? "" : ("ID" + std::to_string(id) + " - ") +
		d.class_label + ": " +
		std::to_string(d.confidence).substr(0, 4);

	cv::Point point(d.bbox.tl().x, d.bbox.tl().y - 5);
	cv::Scalar color = colors_.at(d.class_id);

	// Draw colored rectangle with attached label
	cv::rectangle(image, d.bbox, color, 2);
	cv::putText(image, label, point, cv::FONT_HERSHEY_PLAIN, 1, color, 1);
}