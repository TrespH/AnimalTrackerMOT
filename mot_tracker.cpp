#include <opencv2/opencv.hpp>
#include <iostream>
#include "mot_tracker.hpp"


constexpr int MAX_MISSED_FRAMES = 30;
constexpr float HA_MAX = 1.0f;


float computeIoU(const cv::Rect& a, const cv::Rect& b) {  // Will be moved to utils module
	float ab_inter = (float)((a & b).area());
	float ab_union = (float)((a | b).area());
	if (ab_union == 0) return 0;
	return ab_inter / ab_union;
}


MOTTracker::MOTTracker() {}

std::vector<KalmanTracker> MOTTracker::update(const std::vector<Detection>& detections) {

	int num_tracks = tracks_.size();
	int num_detections = detections.size();
	cv::Rect track_bbox, detection_bbox;

	// IoU cost matrix — rows = tracks, cols = detections
	cv::Mat cost_matrix(num_tracks, num_detections, CV_32F);

	// Call predict() on every existing track and build IoU matrix
	for (int i = 0; i < num_tracks; i++) {
		track_bbox = cv::Rect(tracks_[i].predict());
		for (int j = 0; j < num_detections; j++) 
			cost_matrix.at<float>(i, j) = 1 - computeIoU(track_bbox, detections[j].bbox);  // Score to Cost
	}
	
	
	// Run Hungarian algorithm on cost matrix → get matched pairs
	std::vector<std::pair<int, int>> optimal_pairs;
	optimal_pairs = hungarian_algorithm(cost_matrix);
	// For matched pairs    → call track.update(detection)
	
	// For unmatched detections → create new KalmanTracker

	// For unmatched tracks     → increment missed_frames (already done in predict)

	// Delete tracks where missed_frames > MAX_MISSED_FRAMES

	// Return remaining tracks
	return tracks_;
}


std::vector<std::pair<int, int>> MOTTracker::hungarian_algorithm(cv::Mat& cost_matrix) {
	// Goal: draw the minimum number of hor/vert lines to cover all zeros, so that these are n in number
	// This way, we can extract unique (Track, Detection) pairs 

	std::vector<bool> row_covered;  // Covered rows
	std::vector<bool> col_covered;  // Covered columns
	std::vector<std::pair<int, int>> starred;  // Optimal zeros
	std::vector<std::pair<int, int>> primed;  // Candidate zeros
	double min_val = 0.0;
	cv::Mat padding;

	// Step 0: Extension
	// Pad the smaller dimension with dummy values (1.0f)
	int rows_excess = cost_matrix.rows - cost_matrix.cols;
	if (rows_excess > 0) {
		padding = HA_MAX * cv::Mat::ones(cost_matrix.rows, rows_excess, CV_32F);
		cv::hconcat(cost_matrix, padding, cost_matrix);
	}
	else if (rows_excess < 0) {
		padding = HA_MAX * cv::Mat::ones(-rows_excess, cost_matrix.cols, CV_32F);
		cost_matrix.push_back(padding);
	}

	int n = cost_matrix.rows; // Now squared

	row_covered = std::vector<bool>(n, false);
	col_covered = std::vector<bool>(n, false);

	// Step 1: Reduction
	// Subtract the row minimum from each row, then subtract the column minimum from each column
	// Results: every row and column will have at least one zero (a "candidate assignment")
	for (int i = 0; i < n; i++) {
		cv::minMaxLoc(cost_matrix.row(i), &min_val, nullptr);
		for (int j = 0; j < n; j++)
			cost_matrix.at<float>(i, j) -= (float)min_val;
	}
	for (int j = 0; j < n; j++) {
		cv::minMaxLoc(cost_matrix.col(j), &min_val, nullptr);
		for (int i = 0; i < n; i++)
			cost_matrix.at<float>(i, j) -= (float)min_val;
	}

	std::fill(row_covered.begin(), row_covered.end(), false);
	std::fill(col_covered.begin(), col_covered.end(), false);

	// Step 2: Initial Starring
	// Find zeros and 'star' them if they aren't in a row/col that already has a star.
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (std::abs(cost_matrix.at<float>(i, j)) < 1e-5f && !row_covered[i] && !col_covered[j]) {
				starred.push_back({ i, j });
				row_covered[i] = true;
				col_covered[j] = true;
			}
		}
	}

	// A fast approach I thought about would be to iteratively uncover all lines such that their 0s do not remain uncovered
	// By the way, this would result in a greedy approach, thus possibly leading to local optimum traps in complex cases


	// Step 3: reset covers and primes, and cover all columns that have a starred zero
	reset_cover_cols(row_covered, col_covered, primed, starred);

	while (starred.size() < n) {

		// Step 4: find an uncovered zero and prime it
		std::pair<int, int> p;
		bool found_zero = false;
		for (int i = 0; i < n && !found_zero; i++) {
			if (row_covered[i]) continue;
			for (int j = 0; j < n && !found_zero; j++) {
				if (!col_covered[j] && std::abs(cost_matrix.at<float>(i, j)) < 1e-5f) {
					p = { i, j };
					primed.push_back(p);
					found_zero = true;
				}
			}
		}

		if (found_zero) {
			// If its row has no star, go to Step 5
			int star_col = -1;
			for (std::pair<int, int>& ij : starred)
				if (ij.first == p.first) {
					star_col = ij.second;
					break;
				}
			if (star_col == -1) {
				// Step 5: augmenting path
				augment_path(p, starred, primed, n);
				// Step 3
				reset_cover_cols(row_covered, col_covered, primed, starred);
				// Continue while loop
			}

			// Else cover that row, uncover the star's column, keep looking
			else {
				row_covered[p.first] = true;
				col_covered[star_col] = false;
			}
		}

		else {
			// Step 6: augment values
			// Find min uncovered value
			float min_step6 = HA_MAX;
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (!row_covered[i] && !col_covered[j]) {
						float v = cost_matrix.at<float>(i, j);
						if (v < min_step6) min_step6 = v;
					}
				}
			}
			// Subtract from uncovered, add to doubly-covered
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (!row_covered[i] && !col_covered[j])
						cost_matrix.at<float>(i, j) -= min_step6;
					else if (row_covered[i] && col_covered[j])
						cost_matrix.at<float>(i, j) += min_step6;
				}
			}
			// Continue back to Step 4 (NO re-starring)
		}

	}

	// If all N columns are covered, we're done! (interpret starred zeros as assignments)
	return starred;
}

void MOTTracker::augment_path(std::pair<int, int>& p_start, std::vector<std::pair<int, int>>& starred, std::vector<std::pair<int, int>>& primed, int n) {
	// We are here guaranteed of the absence of starred zeros on p's row, but not on its column
	std::vector<std::pair<int, int>> path;
	path.push_back(p_start);  // The start of the augmenting path

	bool done = false;
	while (!done) {
		// Find a star in the same column as the last primed zero
		int star_row = -1;
		for (std::pair<int, int>& s : starred) {
			if (s.second == path.back().second) {
				star_row = s.first;
				break;
			}
		}

		// If found, add it to path
		if (star_row != -1) {
			path.push_back({ star_row, path.back().second }); // Add starred zero to path

			// Now find the primed zero in the same row as that starred zero (must exist)
			int prime_col = -1;
			for (std::pair<int, int>& p : primed) {
				if (p.first == star_row) {
					prime_col = p.second;
					break;
				}
			}
			if (prime_col == -1)
				std::cerr << "Found no primed zero in starred row\n";

			path.push_back({ star_row, prime_col }); // Add primed zero to path
		}
		else done = true;
	}

	// Flip the stars: 
	// For every pair in path: if it was starred, unstar it. If it was primed, star it.
	for (std::pair<int, int>& p : path) {
		auto it = std::find(starred.begin(), starred.end(), p);
		if (it != starred.end()) starred.erase(it);
		else starred.push_back(p);
	}
}

void reset_cover_cols(std::vector<bool>& row_covered, std::vector<bool>& col_covered, std::vector<std::pair<int, int>>& primed, std::vector<std::pair<int, int>>& starred) {
	// Reset everything for the next iteration
	primed.clear();
	std::fill(row_covered.begin(), row_covered.end(), false);
	std::fill(col_covered.begin(), col_covered.end(), false);

	// Cover all columns that have a starred zero
	for (std::pair<int, int>& ij : starred)
		col_covered[ij.second] = true;
}
