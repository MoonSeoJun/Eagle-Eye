from sort import Sort

import numpy as np
import argparse
import imutils
import time
import cv2
import os

def create_video(input_video, output_video, homo_class):
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
    args["output"] = f"/Eagle-Eye/result/{output_video}"

    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    sort_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.15)
    vs = cv2.VideoCapture(args["input"])
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

                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    rects.append([x, y, int(width) + x, int(height) + y])

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
        img = cv2.imread("/Eagle-Eye/source/images/pitch.jpg")

        player_position_list = []
        objects = sort_tracker.update(rects)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                player_position_list.append([x + int(w/2), y + int(h)])

        result_dict = homo_class.get_bird_view_position(player_position_list)

        for position, object in zip(result_dict, objects):
            text = str(object[-1])
            cv2.putText(img, text, (int(position[0]/position[2] - 10), int(position[1]/position[2] + 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            cv2.circle(img, (int(position[0]/position[2]), int(position[1]/position[2])), 3, (0, 255, 0), -1)

        if cv2.waitKey(10) & 0xFF == 27:
            break

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (img.shape[1], img.shape[0]), True)

            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))

        writer.write(img)

    cv2.waitKey(0)
    vs.release()