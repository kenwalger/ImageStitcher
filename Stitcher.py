import numpy
import imutils
import cv2


class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojection_threshold=4.0,
               show_matches=False):
        (image_a, image_b) = images
        (key_points_a, features_a) = self.detect_and_describe(image_a)
        (key_points_b, features_b) = self.detect_and_describe(image_b)

        keypoint_matches = self.match_keypoints(key_points_a,
                                                key_points_b,
                                                features_a,
                                                features_b,
                                                ratio,
                                                reprojection_threshold)
        if keypoint_matches is None:
            print("No matches in images")
            return None
        (matches, homography_matrix, status) = keypoint_matches
        keypoint_image = cv2.warpPerspective(image_a,
                                             homography_matrix,
                                             (image_a.shape[1] + image_b.shape[1],
                                              image_a.shape[0])
                                             )
        keypoint_image[0:image_b.shape[0], 0:image_b.shape[1]] = image_b

        if show_matches:
            visualization = self.draw_matches(image_a,
                                              image_b,
                                              key_points_a,
                                              key_points_b,
                                              matches,
                                              status
                                              )
            return keypoint_image, visualization

        return keypoint_image

    def detect_and_describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.isv3:
            descriptor = cv2.xfeatures2d.SIFT_create()
            (key_points, features) = descriptor.detect_and_compute(image, None)
        else:
            detector = cv2.FeatureDetector_create("SIFT")
            key_points = detector.detect(gray)
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (key_points, features) = extractor.compute(gray, key_points)

        key_points = numpy.float32([key_point.pt for key_point in key_points])
        return key_points, features

    def match_keypoints(self, key_points_a, key_points_b, features_a,
                        features_b, ratio, reprojection_threshold):

        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(features_a, features_b, 2)
        matches = []

        for match in raw_matches:
            if len(match) == 2 and match[0].distance < match[1].distance * \
                    ratio:
                matches.append((match[0].trainIdx, match[0].queryIdx))

            if len(matches) > 4:
                points_A = numpy.float32([key_points_a[i] for (_,
                                                               i) in matches])
                points_b = numpy.float32([key_points_b[i] for (i,
                                                               _) in matches])
                (homography_matrix, status) = \
                    cv2.findHomography(points_A,
                                       points_b,
                                       cv2.RANSAC,
                                       reprojection_threshold
                                       )
                return matches, homography_matrix, status

            return None

    def draw_matches(self,
                     image_a,
                     image_b,
                     keypoints_a,
                     keypoints_b,
                     matches,
                     status):
        # initialize the output visualization image
        (hA, wA) = image_a.shape[:2]
        (hB, wB) = image_b.shape[:2]
        vis = numpy.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = image_a
        vis[0:hB, wA:] = image_b

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                point_a = (int(keypoints_a[queryIdx][0]),
                       int(keypoints_a[queryIdx][1]))
                point_b = (int(keypoints_b[trainIdx][0]) + wA,
                       int(keypoints_b[trainIdx][1]))
                cv2.line(vis, point_a, point_b, (0, 255, 0), 1)

        # return the visualization
        return vis
