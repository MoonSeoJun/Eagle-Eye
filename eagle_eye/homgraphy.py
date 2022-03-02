import numpy as np
import cv2 as cv


class Homography:
    def __init__(self, video_point, pitch_point):
        self.src_list = video_point
        self.dst_list = pitch_point

    def get_bird_view_position(self, position):
        src_pts = np.array(self.src_list).reshape(-1,1,2)
        dst_pts = np.array(self.dst_list).reshape(-1,1,2)

        H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        return self.find_confrontation_point(H, position)

    @staticmethod
    def find_confrontation_point(homo_matrix , position) -> list:
        result_dict = []
        for j in range(0, len(position)):
            sample_arr = []
            for i in range(0,3):
                sample_arr.append((homo_matrix[i][0] * position[j][0]) + (homo_matrix[i][1] * position[j][1]) + homo_matrix[i][2])
            result_dict.append(sample_arr)

        return result_dict
