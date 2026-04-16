# AnimalTrackerMOT

A deliberate project on *Multi-Object Tracking (MOT)* applied to animal targets detected using YOLO (v8n), written in C++ using OpenCV.

The MOT is based on the *Simple Online and Realtime Tracking (SORT)* algorithm, which combines a Kalman filter for state estimation and the Hungarian algorithm for data association.

Incremental implementation steps are:

1) A class for the **YOLO detector**: using a pre-trained *yolov8n* model, we draw bounding boxes on targets which class prediction belongs to pre-selected classes of animals from the COCO dataset (these are: *person (yes, us too), bird, cat, dog, horse, sheep, cow, elephant, bear, zebra and giraffe*); we specify a threshold for the minimum confidence required (set to `0.45`) and a threshold for Non-Maximum Suppression of all bounding boxes detected (set to `0.5`)
2) A class for the **Kalman Tracker**: based on the concept of the *Kalman filter*, this module will receive first detections, assign them an id and apply the *predict-then-update* strategy, formalized by:

    ```math
    \text{Predict step:} \\
    \begin{cases}
        \hat{x}_{k|k-1} = F \hat{x}_{k-1|k-1} & \text{Next state} \\
        P_{k|k-1} = F P_{k-1|k-1} F^T + Q & \text{Next covariance}
    \end{cases}
    ```

    ```math
    \text{Update step:} \\
    \begin{cases}
        z_k = H \hat{x}_{k|k-1} & \text{Measurement} \\
        K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1} & \text{Kalman gain} \\
        \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H \hat{x}_{k|k-1}) & \text{Updated state} \\
        P_{k|k} = (I - K_k H) P_{k|k-1} & \text{Updated covariance}
    \end{cases}
    ```

> where $x$ is the state vector (in our case, the bounding box coordinates and their velocities), $z$ is the measurement vector (detected bounding box coordinates), $P$ is the covariance matrix, $F$ is the state transition matrix (which assumes **constant velocity**), $Q$ is the process noise (set to a small value of $0.01 * I$), $H$ is the measurement matrix (which identically maps the state to the measurement space), $R$ is the measurement noise (set to a moderate value of $10 * I$), and $K$ is the *Kalman gain*.

1) A class for the **MOT Tracker**: this module will, given a new list of YOLO detections for the currect frame, update/add/drop trackers and return them with their assigned ids. The main steps are:

    1) **Predict**: for each existing tracker, we call the `predict()` method of the Kalman Tracker to get the predicted bounding box for the current frame.
    2) **Associate**: we compute the Intersection over Union (IoU) between each predicted bounding box and each new detection. We then use the *Hungarian algorithm* to find the optimal assignment of detections to trackers based on the IoU scores (converted to costs), with a threshold of `0.3` to filter out low-quality matches.
        > The implementation from scratch of the *Hungarian algorithm* has definitely been the most challenging part of this project, but it was a great learning experience in terms of understanding the algorithm and its applications in MOT. For an intuitive explanation of the algorithm, I found this [article](https://www.thinkautonomous.ai/blog/hungarian-algorithm/) very helpful.
    3) **Update**: for each *matched pair* of tracker and detection, we call the `update()` method of the Kalman Tracker with the new detection to refine its state.
    4) **Create**: for each *unmatched detection*, we create a new tracker and assign it a new id.
    5) **Delete**: for each *unmatched tracker*, we increment a "missed" counter. If this counter exceeds a certain threshold (set to `5`), we delete the tracker.

2) **GPU acceleration** re-compiling OpenCV with CUDA/CuDNN, in order to speed up the YOLO detection and the Kalman filter computations.

Further steps may include the implementation of a more sophisticated data association method (e.g., using appearance features or a Re-Identification model) to improve tracking performance in crowded scenes or with occlusions.
