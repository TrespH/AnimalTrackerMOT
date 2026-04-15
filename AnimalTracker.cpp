#include <opencv2/opencv.hpp>
#include <iostream>
#include <unordered_set>
#include "yolo_detector.hpp"
#include "mot_tracker.hpp"

const std::string ACQUISITION_MODE = "video"; // image, video or camera


int main() {
	/*float data[] = {
		6, 10, 5, 2,
		3, 5, 7, 7,
		9, 1, 8, 4,
		10, 3, 12, 5,
	};
	cv::Mat test(4, 4, CV_32F, data);
	
	return 0;*/

	
	//std::string im_path = "Resources/lambo.png";
	std::string im_path = "Resources/cows_human.jpg";
	std::string video_path = "Resources/Wolf and dog.mp4";

	std::string model_path = "Resources/yolov8n.onnx";
	std::string names_path = "Resources/COCO_classes.txt";

	float conf_thresh = 0.45f;
	float nms_thresh = 0.50f;

	// Allowed classes to detect (let's use a hash structure to have O(1) search per detection)
	std::unordered_set<int> allowed_classes = { 0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }; // Animals only: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

	
	if (model_path.empty() || names_path.empty()) {
		std::cerr << "Error: Missing required paths for model, or class names." << std::endl;
		return -1;
	}

	// Detector creation
	YOLODetector detector(model_path, names_path, allowed_classes, conf_thresh, nms_thresh, false);

	if (ACQUISITION_MODE == "image") {
		// Acquire image
		if (im_path.empty()) {
			std::cerr << "Error: Missing required paths for image." << std::endl;
			return -1;
		}

		cv::Mat image = cv::imread(im_path);
		if (image.empty()) {
			std::cerr << "Error: Could not read the image at " << im_path << std::endl;
			return -1;
		}

		// Detect animals and draw onto the same image
		std::vector<Detection> detections = detector.detect(image);
		detector.draw(image, detections);

		// Show image wih bboxes, labels and likelihoods
		cv::imshow("image", image);
		cv::waitKey(0);
	}

	else if (ACQUISITION_MODE == "video" || ACQUISITION_MODE == "camera") {
		cv::VideoCapture capture;

		if (ACQUISITION_MODE == "video") {
			if (video_path.empty()) {
				std::cerr << "Error: Missing required paths for video." << std::endl;
				return -1;
			}
			capture.open(video_path);
		}
		else
			capture.open(0); // Set camera stream

		if (!capture.isOpened()) {
			std::cerr << "Error: Could not read the video at " << video_path << std::endl;
			return -1;
		}

		cv::Mat frame;

		while (capture.read(frame)) {

			// Detect animals and draw onto each frame
			std::vector<Detection> detections = detector.detect(frame);
			detector.draw(frame, detections);
			cv::imshow("frame", frame);

			// Show frame wih bboxes, labels and likelihoods
			if (cv::waitKey(30) == 27) break;  // 27: Escape key
		}

	}

	return 0;
}