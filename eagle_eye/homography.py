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
            sample_arr = []
            position_arr = []
            for h in self.H:
                sample_arr.append((h[0] * position[0]) + (h[1] * position[1]) + h[2])
            position_arr = [int(sample_arr[0]/sample_arr[2]), int(sample_arr[1]/sample_arr[2])]
            result_dict.append(position_arr)

        return result_dict