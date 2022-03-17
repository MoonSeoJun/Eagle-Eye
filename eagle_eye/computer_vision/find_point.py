import cv2
import numpy as np


class PointFinder:
    def __init__(self, video_path, pitch_img):
        self.video_path = video_path
        self.pitch_img = pitch_img

        self.pts_cnt = 0
        self.pts = np.zeros((4,2), dtype=np.float32)
        self.pts_return = np.zeros((4,2), dtype=np.float32)

    def find_point(self, type):
        if type == "video":
            img = cv2.VideoCapture(self.video_path)
            (_, img) = img.read()
        elif type == "image":
            img = cv2.imread(self.pitch_img, cv2.IMREAD_ANYCOLOR)
        draw = img.copy()

        def onMouse(event, x, y, flags, param): 
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(draw, (x,y), 5, (0,255,0), -1)
                cv2.imshow("scanning", draw)

                self.pts[self.pts_cnt] = [x,y]
                self.pts_cnt+=1
                if self.pts_cnt == 4:
                    sm = self.pts.sum(axis=1)
                    diff = np.diff(self.pts, axis = 1)

                    topLeft = self.pts[np.argmin(sm)]
                    bottomRight = self.pts[np.argmax(sm)]
                    topRight = self.pts[np.argmin(diff)]
                    bottomLeft = self.pts[np.argmax(diff)]

                    self.pts = np.int32([topLeft, topRight, bottomRight , bottomLeft])

                    self.pts_return = self.pts

                    self.pts = np.zeros((4,2), dtype=np.float32)
                    self.pts_cnt = 0

                    cv2.destroyAllWindows()

        cv2.imshow("scanning", img)
        cv2.setMouseCallback("scanning", onMouse)
        cv2.waitKey(0)

        return self.pts_return