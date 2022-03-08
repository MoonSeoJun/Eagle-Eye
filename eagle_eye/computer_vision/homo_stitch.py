import numpy as np, cv2

 

# 왼쪽/오른쪽 사진 읽기

# 왼쪽 사진 = train = 매칭의 대상

imgL_video = cv2.VideoCapture("/Eagle-Eye/source/videos/game_b2.avi")
imgR_vdieo = cv2.VideoCapture("/Eagle-Eye/source/videos/game_b3.avi")

while True:
    (grabbed1, imgL) = imgL_video.read()
    imgL = cv2.flip(imgL, 1)
    imgL = cv2.resize(imgL, (int(1920/3), int(1080/3)))

    # 오른쪽 사진 = query = 매칭의 기준

    (grabbed2, imgR) = imgR_vdieo.read()
    imgR = cv2.resize(imgR, (int(1920/3), int(1080/3)))

    if not grabbed1 or not grabbed2:
        break

    hl, wl = imgL.shape[:2]     # 왼쪽 사진 높이, 넓이
    hr, wr = imgR.shape[:2]     # 오른쪽 사진 높이, 넓이

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)  # 그레이 스케일 변환
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)  # 그레이 스케일 변환

    # SIFT 특징 검출기 생성 및 특징점 검출

    descriptor = cv2.SIFT_create()  # SIFT 추출기 생성
    (kpsL, featuresL) = descriptor.detectAndCompute(imgL, None) # 키포인트, 디스크립터 
    (kpsR, featuresR) = descriptor.detectAndCompute(imgR, None) # 키포인트, 디스크립터 

    # BF 매칭기 생성 및 knn 매칭
    matcher = cv2.DescriptorMatcher_create("BruteForce")    # BF 매칭기 생성
    matches = matcher.knnMatch(featuresR, featuresL, 2)     # knn 매칭

    # 좋은 매칭점 선별
    good_matches = []

    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:   ### 75%
            good_matches.append((m[0].trainIdx, m[0].queryIdx))

    print('matches:{}/{}'.format(len(good_matches), len(matches)))

    # 좋은 매칭점이 4개 이상인 원근 변환행렬 구하기
    if len(good_matches) > 4:
        ptsL = np.float32([kpsL[i].pt for (i,_) in good_matches])   # 좋은 매칭점 좌표
        ptsR = np.float32([kpsR[i].pt for (_,i) in good_matches])   # 좋은 매칭점 좌표
        mtrx, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)
        # 원근 변환행렬로 오른쪽 사진을 원근 변환, 결과 이미지 크기는 사진 2장 크기
        panorama = cv2.warpPerspective(imgR, mtrx, (wr + wl, hr))
        # 왼쪽 사진을 원근 변환한 왼쪽 영역에 합성
        panorama[0:hl, 0:wl] = imgL
    else:   # 좋은 매칭점이 4개가 안되는 경우
        panorama = imgL

    

    # 결과 출력
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
cv2.destroyWindow()