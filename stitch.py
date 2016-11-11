from Stitcher import Stitcher
import argparse
import imutils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-f",
                    "--first",
                    required=True,
                    help="path to first image")
parser.add_argument("-s",
                    "--second",
                    required=True,
                    help="path to second image")
args = vars(parser.parse_args())

image_A = cv2.imread(args['first'])
image_B = cv2.imread(args['second'])

stitcher = Stitcher()
(result, visualization) = stitcher.stitch([image_A, image_B], show_matches=True)

cv2.imshow("Image A", image_A)
cv2.imshow("Image B", image_B)
cv2.imshow("Keypoint Matches", visualization)
cv2.imshow("Result", result)
cv2.waitKey(0)