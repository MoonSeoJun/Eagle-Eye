from homography import Homography
from find_point import PointFinder

import cv2

def merge_views(src, dst, H):
    plan_view = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
    for i in range(0,dst.shape[0]):
        for j in range(0, dst.shape[1]):
            if(plan_view.item(i,j,0) == 0 and \
               plan_view.item(i,j,1) == 0 and \
               plan_view.item(i,j,2) == 0):
                plan_view.itemset((i,j,0),dst.item(i,j,0))
                plan_view.itemset((i,j,1),dst.item(i,j,1))
                plan_view.itemset((i,j,2),dst.item(i,j,2))
    return plan_view;

pitch = "/Eagle-Eye/source/images/pitch.jpg"

videos = [
    "/Eagle-Eye/source/videos/game_b1.avi",
    "/Eagle-Eye/source/videos/game_b2.avi",
    "/Eagle-Eye/source/videos/game_b3.avi"
]

img_videos = []

pitch_image = cv2.imread(pitch)

for video in videos:
    video_image = cv2.VideoCapture(video)
    (_, img) = video_image.read()

    dst = cv2.resize(img, (int(1920/3), int(1080/3)))

    if video == "/Eagle-Eye/source/videos/game_b2.avi":
        flip_img = cv2.flip(dst, 1)
        img_videos.append(flip_img)
    else:
        img_videos.append(dst)

hstack = cv2.hconcat(img_videos)

cv2.imwrite("./sample.jpg", hstack)

# video_image = cv2.resize(video_image, (int(video_image.shape[1]/3), int(video_image.shape[0]/3)))

point_finder = PointFinder("/Eagle-Eye/sample.jpg", pitch)
video_point = point_finder.find_point("video")
image_point = point_finder.find_point("image")

homograph = Homography(video_point, image_point)

merged_view = merge_views(hstack, pitch_image, homograph.H)

cv2.imshow("mer", merged_view)
cv2.waitKey()
cv2.destroyAllWindows()