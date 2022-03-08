import cv2

x = int(1920/3)
y = int(1080/3)

video1 = cv2.VideoCapture("/Eagle-Eye/source/videos/game_b1.avi")
video2 = cv2.VideoCapture("/Eagle-Eye/source/videos/game_b2.avi")
video3 = cv2.VideoCapture("/Eagle-Eye/source/videos/game_b3.avi")

while True:
    (grabbed1, video1_img) = video1.read()
    (grabbed2, video2_img) = video2.read()
    (grabbed3, video3_img) = video3.read()

    if not grabbed1 or not grabbed2 or not grabbed3:
        break

    video1_img_re = cv2.resize(video1_img, (x,y))
    video2_img_re = cv2.resize(video2_img, (x,y))
    video3_img_re = cv2.resize(video3_img, (x,y))

    video2_img_re = cv2.flip(video2_img_re, 1)

    total = [video1_img_re, video2_img_re, video3_img_re]

    stitch = cv2.Stitcher_create()
    status, dst = stitch.stitch(total)

    if status != cv2.Stitcher_OK:
        print("stitch fail")
        continue

    # hstack = cv2.hconcat([video1_img_re, video2_img_re, video3_img_re])

    # cv2.imshow("v1", video1_img_re)
    # cv2.imshow("v2", video2_img_re)
    # cv2.imshow("v3", video3_img_re)
    cv2.imshow("hs", dst)
    cv2.waitKey()
cv2.destroyAllWindows()