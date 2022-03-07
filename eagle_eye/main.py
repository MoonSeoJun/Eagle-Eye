from eagle_eye.video_creator.birdview_video import BirdViewVideoCreator
from eagle_eye.computer_vision.find_point import PointFinder
from eagle_eye.computer_vision.homography import Homography


if __name__ == "__main__":
    input_video = input("Enter your video : ")
    output_video = input("Insert output video title : ") + ".avi"

    video_path = f"/Eagle-Eye/source/videos/{input_video}"
    pitch_img = f"/Eagle-Eye/source/images/pitch.jpg"

    pint_finder = PointFinder(video_path, pitch_img)

    input_point = pint_finder.find_point("video")
    pitch_point = pint_finder.find_point("image")

    homography_class = Homography(input_point, pitch_point)

    video_creator = BirdViewVideoCreator(video_path, output_video, homography_class)

    video_creator.create_video()