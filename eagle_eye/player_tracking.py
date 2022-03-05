from sort import Sort
from homography import Homography

import numpy as np
import argparse
import imutils
import time
import cv2
import os
import json

class VideoCreator:
    def __init__(self, input_video:str, output_video:str, homogrph:Homography) -> None:
        self.args = self.__setup_args(input_video, output_video)

        self.weightsPath = os.path.sep.join([self.args["yolo"], "yolov3.weights"])
        self.configPath = os.path.sep.join([self.args["yolo"], "yolov3.cfg"])

        self.sort_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.15)
        self.homogrph = homogrph

        self.players_data_dict = {}
        self.players_data_arr = []

    def __setup_args(self, input_video:str, output_video:str) -> dict:
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--input", help="path to input video", default="")
        ap.add_argument("-o", "--output", help="path to output video", default="")
        ap.add_argument("-y", "--yolo", help="base path to YOLO directory", default='\\Eagle-Eye\\eagle_eye\\yolo-coco')
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
            help="minimum probability to filter weak detections")
        ap.add_argument("-t", "--threshold", type=float, default=0.3,
            help="threshold when applyong non-maxima suppression")
        args = vars(ap.parse_args())

        args["input"] = input_video
        args["output"] = f"/Eagle-Eye/result/videos/{output_video}"

        return args

    def __create_dataset(self):
        self.players_data_dict[self.args["output"]] = self.players_data_arr
        json_title = self.args["output"].split('/')[-1].replace(".avi", "")
        with open(f"/Eagle-Eye/result/datas/{json_title}.json", "w") as json_file:
            json.dump(self.players_data_dict, json_file)

    def create_video(self):
        frame_count = 0

        np.random.seed(42)

        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        vs = cv2.VideoCapture(self.args["input"])
        writer = None
        (W, H) = (None, None)

        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2()\
                else cv2.CAP_PROP_FRAME_COUNT
            total = int(vs.get(prop))
            print("[INFO] {} total frames in video".format(total))
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1

        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break

            if W is None or H is None:
                (H, W) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            boxes = []
            confidences = []
            rects = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > self.args["confidence"]:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.args["confidence"], self.args["threshold"])
            img = cv2.imread("/Eagle-Eye/source/images/pitch.jpg")

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    center_x = x + int(w)
                    center_y = y + int(h)

                    rects.append([center_x, center_y, int(w) + center_x, int(h) + center_y])

            objects = self.sort_tracker.update(rects)
            object_dict = self.homogrph.get_bird_view_position(objects)

            player_datas = []

            for position, object in zip(object_dict, objects):
                object_id = str(object[-1])
                cv2.putText(img, object_id, (position[0]+10, position[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                cv2.circle(img, (position[0], position[1]), 3, (0, 255, 0), -1)
                player_datas.append({object_id : (position[0], position[1])})

            if cv2.waitKey(10) & 0xFF == 27:
                break

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(self.args["output"], fourcc, 30,
                    (img.shape[1], img.shape[0]), True)

                if total > 0:
                    elap = (end - start)
                    print("[INFO] single frame took {:.4f} seconds".format(elap))
                    print("[INFO] estimated total time to finish: {:.4f}".format(
                        elap * total))
            writer.write(img)

            self.players_data_arr.append({f"frame{frame_count}" : player_datas})
            frame_count += 1

        self.__create_dataset()
        cv2.waitKey(0)
        vs.release()