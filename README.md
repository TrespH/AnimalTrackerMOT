# AnimalTrackerMOT

A deliberate project on Multi-Object Tracking (MOT) on animal classes, written in C++ using OpenCV.

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

    where $x$ is the state vector (in our case, the bounding box coordinates and their velocities), $z$ is the measurement vector (detected bounding box coordinates), $P$ is the covariance matrix, $F$ is the state transition matrix (which assumes **constant velocity**), $Q$ is the process noise (set to a small value of $0.01 * I$), $H$ is the measurement matrix (which identically maps the state to the measurement space), $R$ is the measurement noise (set to a moderate value of $10 * I$), and $K$ is the *Kalman gain*.

3) A class for the **MOT Tracker**: based on the **Hungarian algorithm* this module will, given a new list of YOLO detections for the currect frame, update all tracks and return them
