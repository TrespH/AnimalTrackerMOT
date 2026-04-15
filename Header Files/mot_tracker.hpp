#include <opencv2/opencv.hpp>
#include <iostream>
#include "kalman_tracker.hpp"

class MOTTracker {
public:
	MOTTracker();

	// Updates all tracks and returns them
	std::vector<KalmanTracker> update(const std::vector<Detection>& detections);

private:
	std::vector<KalmanTracker> tracks_;

	std::vector<std::pair<int, int>> hungarian_algorithm(cv::Mat& cost_matrix);
	void augment_path(std::pair<int, int>& p_start, std::vector<std::pair<int, int>>& starred, std::vector<std::pair<int, int>>& primed, int n);
	void reset_cover_cols(std::vector<bool>& row_covered, std::vector<bool>& col_covered, std::vector<std::pair<int, int>>& primed, std::vector<std::pair<int, int>>& starred);

};