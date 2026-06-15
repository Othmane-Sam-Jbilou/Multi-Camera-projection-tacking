import cv2
import numpy as np
import pickle
from FloorTracker import FloorTracker
import os
import threading
import time
import queue
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

stream_queues = {1: queue.Queue(maxsize=1), 2: queue.Queue(maxsize=1), 3: queue.Queue(maxsize=1)}

def compute_inverse_homography(extrinsics):
    R = extrinsics['R']
    tvec = extrinsics['t']
    K = extrinsics['K']

    r1 = R[:, 0:1]
    r2 = R[:, 1:2]
    H = K @ np.block([r1, r2, tvec])
    return np.linalg.inv(H)


with open('extrinsics.pkl', 'rb') as f:
    extrinsics = pickle.load(f)

camera_registry = {
    1: {"H_inv": compute_inverse_homography(extrinsics[1]), "dist": extrinsics[1]['dist'], "K": extrinsics[1]['K']},
    2: {"H_inv": compute_inverse_homography(extrinsics[2]), "dist": extrinsics[2]['dist'], "K": extrinsics[2]['K']},
    3: {"H_inv": compute_inverse_homography(extrinsics[3]), "dist": extrinsics[3]['dist'], "K": extrinsics[3]['K']}
}


def frame_producer(stream_id, video_path):
    #thread dedicated to only dumping frames into queues.
    gstreamer_pipeline = (
        f"filesrc location={video_path} ! "
        f"decodebin ! "
        f"videoconvert ! "
        f"appsink max-buffers=1 drop=true"
    )

    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"[ERROR] Stream {stream_id} failed to initialize.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if stream_queues[stream_id].full():
            try:
                stream_queues[stream_id].get_nowait()
            except queue.Empty:
                pass

        stream_queues[stream_id].put(frame)

    cap.release()

def project_bbox_to_floor(bbox, cam_id):

    cam = camera_registry[cam_id]

    u_raw = (bbox[0] + bbox[2]) / 2.0
    v_raw = bbox[3]

    # Apply undistortion
    src_pt = np.array([[[u_raw, v_raw]]], dtype=np.float32)
    undistorted_pt = cv2.undistortPoints(src_pt, cam["K"], cam["dist"], P=cam["K"])
    u_clean, v_clean = undistorted_pt[0][0]

    # Project via Inverse Homography
    pixel_vector = np.array([u_clean, v_clean, 1.0], dtype=np.float32)
    world_homogenous = cam["H_inv"] @ pixel_vector

    X_floor = world_homogenous[0] / world_homogenous[2]
    Y_floor = world_homogenous[1] / world_homogenous[2]

    return np.array([X_floor, Y_floor], dtype=np.float32)

def project_keypoints_to_floor(keypoints, cam_id, conf_threshold=0.1):

    cam = camera_registry[cam_id]

    left_foot = keypoints[15]  # [x, y, confidence]
    right_foot = keypoints[16]

    if left_foot[2] < conf_threshold or right_foot[2] < conf_threshold:
        return None, None

    u_raw = (left_foot[0] + right_foot[0]) / 2.0
    v_raw = (left_foot[1] + right_foot[1]) / 2.0

    # Apply undistortion
    src_pt = np.array([[[u_raw, v_raw]]], dtype=np.float32)
    undistorted_pt = cv2.undistortPoints(src_pt, cam["K"], cam["dist"], P=cam["K"])
    u_clean, v_clean = undistorted_pt[0][0]

    # Project via Inverse Homography
    pixel_vector = np.array([u_clean, v_clean, 1.0], dtype=np.float32)
    world_homogenous = cam["H_inv"] @ pixel_vector

    X_floor = world_homogenous[0] / world_homogenous[2]
    Y_floor = world_homogenous[1] / world_homogenous[2]

    conf = (left_foot[2] + right_foot[2]) / 2

    return np.array([X_floor, Y_floor], dtype=np.float32), conf

def fuse_cross_camera_points(dots, distance_threshold=0.55):

    #clustering to merge coordinates from different cameras seeing the same person.

    if len(dots) == 0:
        return []

    fused_points = []
    used = np.zeros(len(dots), dtype=bool)

    for i in range(len(dots)):
        if used[i]:
            continue

        cluster = [dots[i]]
        used[i] = True

        for j in range(i + 1, len(dots)):
            if not used[j]:
                # Calculate physical distance in meters
                dist = np.linalg.norm(dots[i] - dots[j])
                if dist < distance_threshold:
                    cluster.append(dots[j])
                    used[j] = True

        # Store centroid of the cluster
        fused_points.append(np.mean(cluster, axis=0))

    return fused_points


SHELF_ZONES = {
    1: np.array([[117, 1173], [505, 1100], [810, 1800], [260, 1920]], dtype=np.int32),
    2: np.array([[1055, 1070], [600, 945], [820, 545], [1075, 600]], dtype=np.int32),
    3: np.array([[1076, 543], [780, 545], [612, 100], [915, 100]], dtype=np.int32)
}


def check_shelf_interaction(keypoints, cam_id, conf_threshold=0.4):
    """
    Checks if COCO wrist keypoints (Index 9: Left, 10: Right) are inside
    the defined 2D camera shelf polygon.
    """
    if cam_id not in SHELF_ZONES:
        return False

    left_wrist = keypoints[9]  # [x, y, confidence]
    right_wrist = keypoints[10]  # [x, y, confidence]
    polygon = SHELF_ZONES[cam_id]

    for wrist in [left_wrist, right_wrist]:
        if wrist[2] >= conf_threshold:  # Filter out low-confidence/occluded hand frames
            pt = (int(wrist[0]), int(wrist[1]))
            # Returns positive value if inside, 0 if on edge, negative if outside
            if cv2.pointPolygonTest(polygon, pt, False) >= 0:
                return True

    return False

floor_bounds = [0.0, 0.0, 2.17, 4.16]
SCALE_PX_PER_METER = 250  # Resolution scaling factor
BEV_W = int((floor_bounds[2] - floor_bounds[0]) * SCALE_PX_PER_METER) + 100
BEV_H = int((floor_bounds[3] - floor_bounds[1]) * SCALE_PX_PER_METER) + 100

def generate_bev_canvas(fused_points, raw_dots, camera_indices, tracked_outputs=None):

    #Generates an OpenCV floor map visualization matrix.

    canvas = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

    # Draw Tracking Zone boundary
    x1 = int((floor_bounds[0] - floor_bounds[0]) * SCALE_PX_PER_METER) + 50
    y1 = int((floor_bounds[1] - floor_bounds[1]) * SCALE_PX_PER_METER) + 50
    x2 = int((floor_bounds[2] - floor_bounds[0]) * SCALE_PX_PER_METER) + 50
    y2 = int((floor_bounds[3] - floor_bounds[1]) * SCALE_PX_PER_METER) + 50
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(canvas, "Tracking Zone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 1. Draw raw camera specific dots (small markers)
    # cam_colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}  # BGR mapping
    # for pt, cam_idx in zip(raw_dots, camera_indices):
    #     cx = int((pt[0] - floor_bounds[3]) * SCALE_PX_PER_METER) + 50
    #     cy = int((pt[1] - floor_bounds[4]) * SCALE_PX_PER_METER) + 50
    #     if 0 <= cx < BEV_W and 0 <= cy < BEV_H:
    #         cv2.circle(canvas, (cx, cy), 4, cam_colors[cam_idx], -1)

    # 2. Draw Fused Global Coordinates (Large clean target dots)
    # for pt in fused_points:
    #     cx = int((pt[0]) * SCALE_PX_PER_METER) + 50
    #     cy = int((floor_bounds[3] - pt[1]) * SCALE_PX_PER_METER) + 50
    #     if 0 <= cx < BEV_W and 0 <= cy < BEV_H:
    #         cv2.circle(canvas, (cx, cy), 8, (0, 255, 255), -1)  # Yellow Fused Target
    #         cv2.circle(canvas, (cx, cy), 9, (255, 255, 255), 1)


    if tracked_outputs is not None:
        for track in tracked_outputs:

            # If the track lost its physical detection
            # this frame, skip rendering it entirely. The background system keeps tracking

            if track.lost_count > 0:
                continue

            pt = track.pos
            cx = int((pt[0]) * SCALE_PX_PER_METER) + 50
            cy = int((floor_bounds[3] - pt[1]) * SCALE_PX_PER_METER) + 50

            if 0 <= cx < BEV_W and 0 <= cy < BEV_H:
                cv2.circle(canvas, (cx, cy), 8, (0, 255, 255), -1)
                cv2.putText(canvas, f"{track.id}", (cx + 5, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 2)

    # canvas = cv2.flip(canvas, 0)
    return canvas



def main():
    video_source_1 = "C:/Users/ijbil/OneDrive/Documents/surveillance_system/model/recorded\\ videos/test/shopping/cam1.mp4"
    video_source_2 = "C:/Users/ijbil/OneDrive/Documents/surveillance_system/model/recorded\\ videos/test/shopping/cam2.mp4"
    video_source_3 = "C:/Users/ijbil/OneDrive/Documents/surveillance_system/model/recorded\\ videos/test/shopping/cam3.mp4"

    t1 = threading.Thread(target=frame_producer, args=(1, video_source_1), daemon=True)
    t1.start()
    t2 = threading.Thread(target=frame_producer, args=(2, video_source_2), daemon=True)
    t2.start()
    t3 = threading.Thread(target=frame_producer, args=(3, video_source_3), daemon=True)
    t3.start()

    print("[INFO] Initializing Single YOLO26 Engine for Batch Inference...")
    model = YOLO("yolo26s-pose.onnx")

    INFERENCE_SCALE = 1/3


    # model already compiled in onnx format at half percision
    # model.half()
    # model.to("cuda")

    #init the FloorTracker class
    tracker = FloorTracker(max_dist_meters=1, max_lost_frames=20)

    cv2.namedWindow('Camera 1 (Batched)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera 1 (Batched)', 400, 600)
    cv2.namedWindow('Camera 2 (Batched)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera 2 (Batched)', 400, 600)
    cv2.namedWindow('Camera 3 (Batched)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera 3 (Batched)', 400, 600)
    cv2.namedWindow('Birdseye Floor View', cv2.WINDOW_NORMAL)

    print("Batch pipeline running. Press 'q' to exit safely.")

    while True:
        # Synchronize check: Ensure both cameras have a frame ready for this batch step
        if stream_queues[1].empty() or stream_queues[2].empty() or stream_queues[3].empty():
            time.sleep(0.001)  # 1ms micro-sleep to prevent CPU burning
            continue

        # Pull frames to form our explicit tensor batch
        frame1 = stream_queues[1].get()
        frame2 = stream_queues[2].get()
        frame3 = stream_queues[3].get()

        frame1_low = cv2.resize(frame1, (0, 0), fx=INFERENCE_SCALE, fy=INFERENCE_SCALE,
                                interpolation=cv2.INTER_LINEAR)
        frame2_low = cv2.resize(frame2, (0, 0), fx=INFERENCE_SCALE, fy=INFERENCE_SCALE,
                                interpolation=cv2.INTER_LINEAR)
        frame3_low = cv2.resize(frame3, (0, 0), fx=INFERENCE_SCALE, fy=INFERENCE_SCALE,
                                interpolation=cv2.INTER_LINEAR)

        # Grouping inputs into a list triggers underlying tensor batching, using batch = 2 to batch every 2 frames

        results = model.predict(
            source=[frame1_low, frame2_low, frame3_low], device="cuda", verbose=False, half=True, batch=2, classes=[0]
        )

        dots = []
        confs = []
        camera_indices = []

        for idx , result in enumerate(results):
            cam_id = idx + 1
            keypoints = result.keypoints.data

            if keypoints.shape[0] > 0:

                for keypoint in keypoints.cpu().numpy():

                    # u_raw = (bbox[0] + bbox[2]) / 2.0
                    # v_raw = bbox[3]
                    # if idx == 0:
                    #     cv2.circle(frame1, (keypoint[15], keypoint[16]), 5, (0, 0 , 255), 7)
                    # elif idx == 1:
                    #     cv2.circle(frame2, (keypoint[15], keypoint[16]), 5, (0, 0 , 255), 7)
                    # elif idx == 2:
                    #     cv2.circle(frame3, (keypoint[15], keypoint[16]), 5, (0, 0 , 255), 7)

                    keypoint_scaled_back = keypoint.copy()
                    keypoint_scaled_back[:, 0] = keypoint[:, 0] / INFERENCE_SCALE
                    keypoint_scaled_back[:, 1] = keypoint[:, 1] / INFERENCE_SCALE

                    print(f'keypoint: {keypoint}')

                    floor , conf = project_keypoints_to_floor(keypoint_scaled_back, cam_id)
                    if floor is not None:
                        dots.append(floor)
                        confs.append(conf)
                        camera_indices.append(cam_id)



        fused_targets = fuse_cross_camera_points(dots)

        active_tracks = tracker.update(fused_targets)

        bev_view = generate_bev_canvas(fused_targets, dots, camera_indices, active_tracks)


        # showcasing the yolo body keypoints + bbox
        annotated_frame1 = results[0].plot()

        # display the streams + coordinats bird eye view
        cv2.imshow("Camera 1 (Batched)", annotated_frame1)
        cv2.imshow("Camera 2 (Batched)", frame2)
        cv2.imshow("Camera 3 (Batched)", frame3)
        cv2.imshow("Birdseye Floor View", bev_view)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
