#include <opencv2/opencv.hpp>
#include <iostream>
#include <unordered_set>
#include "yolo_detector.hpp"

int main() {
	//std::string im_path = "Resources/lambo.png";
	std::string im_path = "Resources/cows_human.jpg";

	std::string model_path = "Resources/yolov8n.onnx";
	std::string names_path = "Resources/COCO_classes.txt";

	float conf_thresh = 0.45f;
	float nms_thresh = 0.50f;

	// Allowed classes to detect (let's use a hash structure to have O(1) search)
	std::unordered_set<int> allowed_classes = { 0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }; // Animals only: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

	if (im_path.empty() || model_path.empty() || names_path.empty()) {
		std::cerr << "Error: Missing required paths for image, model, or class names." << std::endl;
		return -1;
	}

	cv::Mat image = cv::imread(im_path);
	if (image.empty()) {
		std::cerr << "Error: Could not read the image at " << im_path << std::endl;
		return -1;
	}

	YOLODetector detector(model_path, names_path, allowed_classes, conf_thresh, nms_thresh, true);
	std::vector<Detection> detections = detector.detect(image);
	detector.draw(image, detections);

	cv::imshow("image", image);
	cv::waitKey(0);
	return 0;
}