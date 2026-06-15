import numpy as np
from scipy.optimize import linear_sum_assignment


class KinematicTrack:
    def __init__(self, track_id, coords):
        self.id = track_id

        # State Vector: [x, y, vx, vy]
        self.state = np.array([coords[0], coords[1], 0.0, 0.0], dtype=np.float32)

        # State Covariance Matrix (Uncertainty of position and velocity)
        self.P = np.eye(4, dtype=np.float32) * 1.0

        # Process Noise Matrix (How much noise we assume human movement introduces per frame)
        self.Q = np.eye(4, dtype=np.float32) * 0.05

        # Measurement Noise Matrix (Camera/Homography jitter variance in meters)
        self.R = np.eye(2, dtype=np.float32) * 0.1

        self.lost_count = 0

    @property
    def pos(self):
        return self.state[:2]

    def predict(self):
        # State transition matrix (x = x + vx, y = y + vy)
        # We introduce a 0.95 damping factor to velocities to handle sudden human deceleration
        F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.95, 0.0],
            [0.0, 0.0, 0.0, 0.95]
        ], dtype=np.float32)

        self.state = F @ self.state

        self.P = F @ self.P @ F.T + self.Q

    def update(self, new_coords):
        # Measurement matrix (we only observe x and y directly)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        z = np.array(new_coords, dtype=np.float32)

        # Innovation (Measurement residual)
        y = z - (H @ self.state)

        # Innovation Covariance
        S = H @ self.P @ H.T + self.R

        # Optimal Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update State & Covariance
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        self.lost_count = 0


class FloorTracker:
    def __init__(self, max_dist_meters=0.8, max_lost_frames=30):
        self.max_dist = max_dist_meters
        self.max_lost = max_lost_frames
        self.next_id = 1
        self.tracks = []

    def update(self, current_points):
        for track in self.tracks:
            track.predict()

        if len(current_points) == 0:
            for track in self.tracks:
                track.lost_count += 1
            self.tracks = [t for t in self.tracks if t.lost_count <= self.max_lost]
            return self.tracks

        if len(self.tracks) == 0:
            for pt in current_points:
                self.tracks.append(KinematicTrack(self.next_id, pt))
                self.next_id += 1
            return self.tracks

        det_positions = np.array(current_points)
        assigned_tracks = set()
        assigned_dets = set()

        # STAGE 1: MATCH ACTIVE TRACKS ONLY (lost_count == 0)

        active_track_indices = [i for i, t in enumerate(self.tracks) if t.lost_count == 0]

        if len(active_track_indices) > 0:
            active_positions = np.array([self.tracks[i].pos for i in active_track_indices])
            cost_matrix_s1 = np.linalg.norm(active_positions[:, np.newaxis] - det_positions, axis=2)

            row_ind_s1, col_ind_s1 = linear_sum_assignment(cost_matrix_s1)
            for r, c in zip(row_ind_s1, col_ind_s1):
                if cost_matrix_s1[r, c] < self.max_dist:
                    track_idx = active_track_indices[r]
                    self.tracks[track_idx].update(det_positions[c])
                    assigned_tracks.add(track_idx)
                    assigned_dets.add(c)

        # STAGE 2: MATCH LOST/DRIFTED TRACKS ONLY

        remaining_track_indices = [i for i in range(len(self.tracks)) if i not in assigned_tracks]
        remaining_det_indices = [c for c in range(len(det_positions)) if c not in assigned_dets]

        if len(remaining_track_indices) > 0 and len(remaining_det_indices) > 0:
            lost_positions = np.array([self.tracks[i].pos for i in remaining_track_indices])
            sub_dets = det_positions[remaining_det_indices]
            cost_matrix_s2 = np.linalg.norm(lost_positions[:, np.newaxis] - sub_dets, axis=2)

            row_ind_s2, col_ind_s2 = linear_sum_assignment(cost_matrix_s2)
            for r, c in zip(row_ind_s2, col_ind_s2):
                track_idx = remaining_track_indices[r]
                det_idx = remaining_det_indices[c]

                # Dynamic gate allows lost tracks a wider search radius
                dynamic_gate = self.max_dist * (1.0 + 0.3 * self.tracks[track_idx].lost_count)
                if cost_matrix_s2[r, c] < dynamic_gate:
                    self.tracks[track_idx].update(det_positions[det_idx])
                    assigned_tracks.add(track_idx)
                    assigned_dets.add(det_idx)


        # STAGE 3: AGE OUT AND SPAWN NEW TRACKS

        for r in range(len(self.tracks)):
            if r not in assigned_tracks:
                self.tracks[r].lost_count += 1

        for c in range(len(det_positions)):
            if c not in assigned_dets:
                self.tracks.append(KinematicTrack(self.next_id, det_positions[c]))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.lost_count <= self.max_lost]
        return self.tracks