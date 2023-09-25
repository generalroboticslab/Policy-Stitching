import pyrealsense2 as rs
import numpy as np
import cv2.aruco
import argparse
import math
from scipy.spatial.transform import Rotation as R

class rs_camera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        if device_product_line == 'L500':
            print("is L500")
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            print("is not L500")
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
        arucoParams = cv2.aruco.DetectorParameters()

        fx, cx, fy, cy = 604.3001708984375, 316.2286071777344, 603.91796875, 238.58575439453125
        self.distortion_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0], np.float64)
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)

        # Start streaming
        self.pipeline.start(config)

    def euler_from_quaternion(self, quaternion):
        rotation = R.from_quat(quaternion)
        euler_angles = rotation.as_euler('XYZ', degrees=True)
        return euler_angles

    def l2_norm(self, point):
        return np.linalg.norm(point)

    def detect_pos(self):
        obj_w_pos = None
        goal_w_pos = None

        # print("----------------------")
        for attempts in range(10):  # try at most 10 times to get the object position
            try:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                # detect ArUco markers
                markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(color_image, self.arucoDict)

                # Estimate the camera pose relative to the marker
                marker_size = 0.04

                if len(markerCorners) > 0:
                    ids = markerIds.flatten()
                    sort = np.argsort(ids)
                    # assert ids[sort[0]] == 0 and ids[sort[1]] == 1 and ids[sort[2]] == 2 and ids[sort[3]] == 5 and \
                    #        ids[sort[4]] == 6 and ids[sort[-1]] == 38
                    assert ids[sort[0]] == 0 and ids[sort[1]] == 1 and ids[sort[2]] == 2 and ids[sort[3]] == 5 and \
                           ids[sort[4]] == 6
                    # get the corners of each anchor points
                    corner0 = markerCorners[sort[0]]
                    corner1 = markerCorners[sort[1]]
                    corner2 = markerCorners[sort[2]]
                    corner5 = markerCorners[sort[3]]
                    corner6 = markerCorners[sort[4]]

                    # get vectors
                    rvecs0, tvecs0, obj_points0 = cv2.aruco.estimatePoseSingleMarkers(corner0, marker_size,
                                                                                      self.camera_matrix,
                                                                                      self.distortion_coeff)
                    rvecs1, tvecs1, obj_points1 = cv2.aruco.estimatePoseSingleMarkers(corner1, marker_size,
                                                                                      self.camera_matrix,
                                                                                      self.distortion_coeff)
                    rvecs2, tvecs2, obj_points2 = cv2.aruco.estimatePoseSingleMarkers(corner2, marker_size,
                                                                                      self.camera_matrix,
                                                                                      self.distortion_coeff)
                    rvecs5, tvecs5, obj_points5 = cv2.aruco.estimatePoseSingleMarkers(corner5, marker_size,
                                                                                      self.camera_matrix,
                                                                                      self.distortion_coeff)
                    rvecs6, tvecs6, obj_points6 = cv2.aruco.estimatePoseSingleMarkers(corner6, marker_size,
                                                                                      self.camera_matrix,
                                                                                      self.distortion_coeff)
                    pos0_x = tvecs0[0][0][0]
                    pos0_y = tvecs0[0][0][1]
                    pos0_z = tvecs0[0][0][2]

                    pos1_x = tvecs1[0][0][0]
                    pos1_y = tvecs1[0][0][1]
                    pos1_z = tvecs1[0][0][2]

                    pos2_x = tvecs2[0][0][0]
                    pos2_y = tvecs2[0][0][1]
                    pos2_z = tvecs2[0][0][2]

                    pos5_x = tvecs5[0][0][0]
                    pos5_y = tvecs5[0][0][1]
                    pos5_z = tvecs5[0][0][2]

                    pos6_x = tvecs6[0][0][0]
                    pos6_y = tvecs6[0][0][1]
                    pos6_z = tvecs6[0][0][2]

                    vec25 = np.array([pos5_x - pos2_x, pos5_y - pos2_y, pos5_z - pos2_z])
                    vec56 = np.array([pos6_x - pos5_x, pos6_y - pos5_y, pos6_z - pos5_z])
                    vec21 = np.array([pos1_x - pos2_x, pos1_y - pos2_y, pos1_z - pos2_z])
                    vec10 = np.array([pos0_x - pos1_x, pos0_y - pos1_y, pos0_z - pos1_z])

                    if ids[sort[5]] == 18 or ids[sort[5]] == 28:
                        # print("detect object")
                        # number 18 is for the cube, 28 for cylinder
                        corner_obj = markerCorners[sort[5]]
                        rvecs_obj, tvecs_obj, obj_points_obj = cv2.aruco.estimatePoseSingleMarkers(corner_obj,
                                                                                                   marker_size,
                                                                                                   self.camera_matrix,
                                                                                                   self.distortion_coeff)

                        pos_obj_x = tvecs_obj[0][0][0]
                        pos_obj_y = tvecs_obj[0][0][1]
                        pos_obj_z = tvecs_obj[0][0][2]

                        vec_obj = np.array([pos_obj_x - pos2_x, pos_obj_y - pos2_y, pos_obj_z - pos2_z])

                        obj_x = np.dot(vec_obj,
                                       (vec25 / self.l2_norm(vec25) + vec56 / self.l2_norm(vec56))) / 2.0 - 0.17  # 0.17
                        obj_y = np.dot(vec_obj,
                                       (vec21 / self.l2_norm(vec21) + vec10 / self.l2_norm(vec10))) / 2.0 - 0.19  # 0.17
                        obj_z = 0.03
                        obj_w_pos = np.array([obj_x, obj_y, obj_z])
                        # print(obj_w_pos)

                        # number 38 is for the goal
                        if ids[sort[-1]] == 38:
                            corner_goal = markerCorners[sort[-1]]
                            rvecs_goal, tvecs_goal, obj_points_goal = cv2.aruco.estimatePoseSingleMarkers(corner_goal,
                                                                                                          marker_size,
                                                                                                          self.camera_matrix,
                                                                                                          self.distortion_coeff)

                            pos_goal_x = tvecs_goal[0][0][0]
                            pos_goal_y = tvecs_goal[0][0][1]
                            pos_goal_z = tvecs_goal[0][0][2]

                            vec_goal = np.array([pos_goal_x - pos2_x, pos_goal_y - pos2_y, pos_goal_z - pos2_z])

                            goal_x = np.dot(vec_goal,
                                            (vec25 / self.l2_norm(vec25) + vec56 / self.l2_norm(vec56))) / 2.0 - 0.17
                            goal_y = np.dot(vec_goal,
                                            (vec21 / self.l2_norm(vec21) + vec10 / self.l2_norm(vec10))) / 2.0 - 0.17
                            goal_z = 0.03
                            goal_w_pos = np.array([goal_x, goal_y, goal_z])
                            # print(goal_w_pos)
                        break
            except:
                continue

        # for step in range(10):  # each game has 50 steps
        #     print("----------------------")
        #     for attempts in range(10):  # try at most 10 times to get the object position
        #         try:
        #             # Wait for a coherent pair of frames: depth and color
        #             frames = self.pipeline.wait_for_frames()
        #             color_frame = frames.get_color_frame()
        #             if not color_frame:
        #                 continue
        #             # Convert images to numpy arrays
        #             color_image = np.asanyarray(color_frame.get_data())
        #             # detect ArUco markers
        #             markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(color_image, self.arucoDict)
        #
        #             # Estimate the camera pose relative to the marker
        #             marker_size = 0.04
        #
        #             if len(markerCorners) > 0:
        #                 ids = markerIds.flatten()
        #                 sort = np.argsort(ids)
        #                 assert ids[sort[0]] == 0 and ids[sort[1]] == 1 and ids[sort[2]] == 2 and ids[sort[3]] == 5 and \
        #                        ids[sort[4]] == 6 and ids[sort[-1]] == 38
        #                 # get the corners of each anchor points
        #                 corner0 = markerCorners[sort[0]]
        #                 corner1 = markerCorners[sort[1]]
        #                 corner2 = markerCorners[sort[2]]
        #                 corner5 = markerCorners[sort[3]]
        #                 corner6 = markerCorners[sort[4]]
        #
        #                 # get vectors
        #                 rvecs0, tvecs0, obj_points0 = cv2.aruco.estimatePoseSingleMarkers(corner0, marker_size,
        #                                                                                   self.camera_matrix,
        #                                                                                   self.distortion_coeff)
        #                 rvecs1, tvecs1, obj_points1 = cv2.aruco.estimatePoseSingleMarkers(corner1, marker_size,
        #                                                                                   self.camera_matrix,
        #                                                                                   self.distortion_coeff)
        #                 rvecs2, tvecs2, obj_points2 = cv2.aruco.estimatePoseSingleMarkers(corner2, marker_size,
        #                                                                                   self.camera_matrix,
        #                                                                                   self.distortion_coeff)
        #                 rvecs5, tvecs5, obj_points5 = cv2.aruco.estimatePoseSingleMarkers(corner5, marker_size,
        #                                                                                   self.camera_matrix,
        #                                                                                   self.distortion_coeff)
        #                 rvecs6, tvecs6, obj_points6 = cv2.aruco.estimatePoseSingleMarkers(corner6, marker_size,
        #                                                                                   self.camera_matrix,
        #                                                                                   self.distortion_coeff)
        #                 pos0_x = tvecs0[0][0][0]
        #                 pos0_y = tvecs0[0][0][1]
        #                 pos0_z = tvecs0[0][0][2]
        #
        #                 pos1_x = tvecs1[0][0][0]
        #                 pos1_y = tvecs1[0][0][1]
        #                 pos1_z = tvecs1[0][0][2]
        #
        #                 pos2_x = tvecs2[0][0][0]
        #                 pos2_y = tvecs2[0][0][1]
        #                 pos2_z = tvecs2[0][0][2]
        #
        #                 pos5_x = tvecs5[0][0][0]
        #                 pos5_y = tvecs5[0][0][1]
        #                 pos5_z = tvecs5[0][0][2]
        #
        #                 pos6_x = tvecs6[0][0][0]
        #                 pos6_y = tvecs6[0][0][1]
        #                 pos6_z = tvecs6[0][0][2]
        #
        #                 vec25 = np.array([pos5_x - pos2_x, pos5_y - pos2_y, pos5_z - pos2_z])
        #                 vec56 = np.array([pos6_x - pos5_x, pos6_y - pos5_y, pos6_z - pos5_z])
        #                 vec21 = np.array([pos1_x - pos2_x, pos1_y - pos2_y, pos1_z - pos2_z])
        #                 vec10 = np.array([pos0_x - pos1_x, pos0_y - pos1_y, pos0_z - pos1_z])
        #
        #                 if ids[sort[-2]] == 18 or ids[sort[-2]] == 28:
        #                     print("detect object")
        #                     # number 18 is for the cube, 28 for cylinder
        #                     corner_obj = markerCorners[sort[-2]]
        #                     rvecs_obj, tvecs_obj, obj_points_obj = cv2.aruco.estimatePoseSingleMarkers(corner_obj,
        #                                                                                                marker_size,
        #                                                                                                self.camera_matrix,
        #                                                                                                self.distortion_coeff)
        #
        #                     pos_obj_x = tvecs_obj[0][0][0]
        #                     pos_obj_y = tvecs_obj[0][0][1]
        #                     pos_obj_z = tvecs_obj[0][0][2]
        #
        #                     vec_obj = np.array([pos_obj_x - pos2_x, pos_obj_y - pos2_y, pos_obj_z - pos2_z])
        #
        #                     obj_x = np.dot(vec_obj, (vec25 / self.l2_norm(vec25) + vec56 / self.l2_norm(vec56))) / 2.0 - 0.17
        #                     obj_y = np.dot(vec_obj, (vec21 / self.l2_norm(vec21) + vec10 / self.l2_norm(vec10))) / 2.0 - 0.18
        #                     obj_z = 0.03
        #                     obj_w_pos = np.array([obj_x, obj_y, obj_z])
        #                     print(obj_w_pos)
        #
        #                     # number 38 is for the goal
        #                     corner_goal = markerCorners[sort[-1]]
        #                     rvecs_goal, tvecs_goal, obj_points_goal = cv2.aruco.estimatePoseSingleMarkers(corner_goal,
        #                                                                                                   marker_size,
        #                                                                                                   self.camera_matrix,
        #                                                                                                   self.distortion_coeff)
        #
        #                     pos_goal_x = tvecs_goal[0][0][0]
        #                     pos_goal_y = tvecs_goal[0][0][1]
        #                     pos_goal_z = tvecs_goal[0][0][2]
        #
        #                     vec_goal = np.array([pos_goal_x - pos2_x, pos_goal_y - pos2_y, pos_goal_z - pos2_z])
        #
        #                     goal_x = np.dot(vec_goal, (vec25 / self.l2_norm(vec25) + vec56 / self.l2_norm(vec56))) / 2.0 - 0.17
        #                     goal_y = np.dot(vec_goal, (vec21 / self.l2_norm(vec21) + vec10 / self.l2_norm(vec10))) / 2.0 - 0.17
        #                     goal_z = 0.03
        #                     goal_w_pos = np.array([goal_x, goal_y, goal_z])
        #                     print(goal_w_pos)
        #                     break
        #         except:
        #             continue

        return obj_w_pos, goal_w_pos

