# Multi-Camera Bird's-Eye View Spatial Tracking Pipeline

## Overview

This repository contains a real-time, multi-camera spatial tracking system designed to track individuals across multiple camera fields of view. The system performs batched pose-estimation inference, projects localized keypoints from image coordinates to a unified 2D floor grid using inverse homography, fuses cross-camera detections, and maintains identity continuity using a custom Kalman-filter-based tracking engine.

The project is currently under active development with a focus on optimizing multi-stream synchronization and hardening identity association in high-density environments.

---

## System Architecture

[camera streams] --> [Batched YOLO-Pose Inference] --> [Inverse Homography] --> [Cross-Camera Fusion] --> [Floor Tracker] --> [Visualization]


The pipeline is structured into four main operational stages:

1. **Multi-Stream Frame Ingestion:** Lightweight background worker threads capture video frames via GStreamer pipelines to minimize CPU overhead. Frames are managed via non-blocking queues configured to drop stale data to prevent stream lag. (we're only using video files for simulation, otherwise you gonna need to capture the camera streams using gstreamer or ffmpeg)
2. **Batched Pose Inference:** Inputs are resized and batched dynamically into a single inference tensor fed to a YOLO-Pose engine executing on NVIDIA CUDA 
3. **Geospatial Projection (Inverse Homography):** 2D image coordinates representing human foot positions (calculated as the midpoint between the left and right ankle keypoints) are undistorted using camera parameters and projected onto the physical floor plane ($Z=0$) utilizing an inverse homography matrix ($H^{-1}$).
4. **Data Fusion & Spatial Tracking:** Disparate coordinate inputs mapping to the same physical target are clustered into a unified centroid. This centroid is ingested by a state-space tracking system that models human kinematics.

---

## Technical Features

### Camera Calibration
Camera intrinsic parameters ($K$), distortion coefficients ($dist$), and extrinsic rotation/translation vectors ($R$ and $t$) were computed using standard OpenCV checkerboard calibration techniques. This offline calibration pipeline ensures accurate lens undistortion and spatial transformation matrices, matching 2D pixel coordinates reliably to a standardized, real-world metric space.

### Kinematic Kalman Filtering
The `FloorTracker` implementation utilizes a 4D state vector to track both spatial position and physical velocity:

$$x = \begin{bmatrix} x & y & v_x & v_y \end{bmatrix}^T$$

A custom damping factor is applied to the velocity state transition matrix to gracefully handle sudden human deceleration and erratic movement changes.

### Two-Stage Data Association
To optimize tracking stability during target occlusions and transitions between camera views, the tracker utilizes a two-stage data association gate executed via the Hungarian (Linear Sum Assignment) algorithm:
* **Stage 1:** Associates detections strictly with highly active, currently visible tracks using a tight spatial distance threshold.
* **Stage 2 (Lost Track Recovery):** Evaluates unassigned detections against historically lost or drifted tracks using a dynamically expanding search gate based on the duration of the track's occlusion.

### Multi-Stream Batching
Instead of running sequential inference loops per camera stream, the architecture passes multi-frame arrays into the underlying model execution step. This maximizes GPU tensor core utilization and drastically reduces inference latency compared to serialized execution.

---

## Core File Structure

* `main.py`: Core execution script containing the frame-producer multi-threading implementation, inference loops, projection mechanics, and visual display pipelines.
* `FloorTracker.py`: Contains the `KinematicTrack` state management class and the `FloorTracker` two-stage Hungarian assignment logic.
* `extrinsics.pkl`: Serialized storage containing the camera calibration data generated through the checkerboard analysis.

---

## Preview

<img width="600" height="440" alt="Screen Recording 2026-06-12 155754 - Trim" src="https://github.com/user-attachments/assets/717a3848-4007-47d3-9cf2-ff2991af677b" />

## Prerequisites

### Dependencies
Ensure the following packages and libraries are installed within your environment:

* Python 3.10+
* OpenCV (compiled with GStreamer support this was a nightmare as I use windows, if cannot access it just switch to ffmpeg to receive camera streams) 
* NumPy
* SciPy
* Ultralytics YOLO
* PyTorch (with CUDA support enabled)

### Environment Configuration
To manage OpenMP library initializations across parallel execution components, establish the following system flag:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

or add this line on top of the code:

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

