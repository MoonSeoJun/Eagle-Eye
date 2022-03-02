from player_tracking import create_video
from find_point import PointFinder
from homgraphy import Homography


if __name__ == "__main__":
    input_video = input("Enter your video : ")
    output_video = input("Insert output video title(.avi) : ")

    video_path = f"/Eagle-Eye/source/videos/{input_video}"
    pitch_img = "/Eagle-Eye/src/images/pitch.jpg"

    pint_finder = PointFinder(video_path, pitch_img)

    input_point = pint_finder.find_point("video")
    pitch_point = pint_finder.find_point("image")

    homo_class = Homography(input_point, pitch_point)

    create_video(input_video, output_video, homo_class)