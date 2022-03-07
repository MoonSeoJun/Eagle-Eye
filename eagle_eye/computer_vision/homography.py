import numpy as np
import cv2 as cv


class Homography:
    def __init__(self, video_point, pitch_point):
        self.src_list = video_point
        self.dst_list = pitch_point

        __src_pts = np.array(self.src_list).reshape(-1,1,2)
        __dst_pts = np.array(self.dst_list).reshape(-1,1,2)

        self.H, _ = cv.findHomography(__src_pts, __dst_pts, cv.RANSAC, 5.0)

    def get_bird_view_position(self, positions):
        result_dict = []
        for position in positions:
            position_arr = [(h[0] * position[0]) + (h[1] * position[1]) + h[2] for h in self.H]
            result_dict.append([int(position_arr[0]/position_arr[2]), int(position_arr[1]/position_arr[2])])

        return result_dict