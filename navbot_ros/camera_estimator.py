import cv2
import cv2.aruco as aruco
import numpy as np
from . import parameters

estimate_pose_single_markers = getattr(aruco, "estimatePoseSingleMarkers")

class CameraEstimator:
    def __init__(self, camera_id, marker_length, camera_matrix, dist_coeffs):
        self.marker_length = marker_length
        self.camera_matrix = np.asarray(camera_matrix)
        self.dist_coeffs = np.asarray(dist_coeffs)

        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        detector_params = aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.detector = aruco.ArucoDetector(dictionary, detector_params)

    def read_frame(self):
        return self.cap.read()
    
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        vis = frame.copy()
        markers = []

        if ids is not None:
            aruco.drawDetectedMarkers(vis, corners, ids)
            rvecs, tvecs, _ = estimate_pose_single_markers(
                corners,
                self.marker_length,
                self.camera_matrix,
                self.dist_coeffs
            )

            for i in range(len(ids)):
                marker_id = int(ids[i])
                tvec = tvecs[i].ravel()
                rvec = rvecs[i].ravel()
                markers.append((marker_id, tvec, rvec))
                cv2.drawFrameAxes(
                    vis, 
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvecs[i],
                    tvecs[i],
                    0.05
                )
        return vis, markers
            
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        while True:
            ret, frame = self.read_frame()
            if not ret or frame is None:
                print("No frame")
                break

            vis, markers = self.process_frame(frame)
            print(f"Detected {len(markers)} markers")

            for marker_id, tvec, rvec in markers:
                xi = np.concatenate([tvec, rvec])
                print("id", marker_id, "camera_frame_xi(6):", xi)

            cv2.imshow("Aruco Marker Detection", vis)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.close()


if __name__ == "__main__":
    estimator = CameraEstimator(parameters.camera_id, parameters.marker_length, parameters.camera_matrix, parameters.dist_coeffs)
    estimator.run()