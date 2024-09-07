import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors
from std_msgs.msg import Float32

# Constants
QOS_PROFILE_DEFAULT = 10
PI = math.pi
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.35
VECTOR_MAGNITUDE_MINIMUM = 2.5

# Fixed Trackbar Values
TRACKBAR_VALUES = {
    "Width Top": 71,
    "Height Top": 170,
    "Width Bottom": 0,
    "Height Bottom": 240
}

class EdgeVectorsPublisher(Node):
    def __init__(self):
        super().__init__("edge_vectors_publisher")
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT,
        )
        self.mycam_subscription = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self.getLaneCurve,
            QOS_PROFILE_DEFAULT,
        )
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors, "/edge_vectors", QOS_PROFILE_DEFAULT
        )
        self.publisher_thresh_image = self.create_publisher(
            CompressedImage, "/debug_images/thresh_image", QOS_PROFILE_DEFAULT
        )
        self.publisher_vector_image = self.create_publisher(
            CompressedImage, "/debug_images/vector_image", QOS_PROFILE_DEFAULT
        )
        self.turn_publisher = self.create_publisher(
            Float32, "/turn", QOS_PROFILE_DEFAULT
        )
        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0
        self.curveList = []
        self.avgVal = 10

    def publish_debug_image(self, publisher, image):
        try:
            message = CompressedImage()
            _, encoded_data = cv2.imencode(".jpg", image)
            message.format = "jpeg"
            message.data = encoded_data.tobytes()
            publisher.publish(message)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {str(e)}")

    def get_vector_angle_in_radians(self, vector):
        try:
            if (vector[0][0] - vector[1][0]) == 0:
                theta = PI / 2
            else:
                slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
                theta = math.atan(slope)
            return theta
        except Exception as e:
            self.get_logger().error(f"Failed to get vector angle: {str(e)}")
            return 0

    def compute_vectors_from_image(self, image, thresh):
        try:
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            vectors = []
            for i in range(len(contours)):
                coordinates = contours[i][:, 0, :]
                min_y_value = np.min(coordinates[:, 1])
                max_y_value = np.max(coordinates[:, 1])
                min_y_coords = np.array(coordinates[coordinates[:, 1] == min_y_value])
                max_y_coords = np.array(coordinates[coordinates[:, 1] == max_y_value])
                min_y_coord = min_y_coords[0]
                max_y_coord = max_y_coords[0]
                magnitude = np.linalg.norm(min_y_coord - max_y_coord)
                if magnitude > VECTOR_MAGNITUDE_MINIMUM:
                    rover_point = [self.image_width / 2, self.lower_image_height]
                    middle_point = (min_y_coord + max_y_coord) / 2
                    distance = np.linalg.norm(middle_point - rover_point)
                    angle = self.get_vector_angle_in_radians([min_y_coord, max_y_coord])
                    if angle > 0:
                        min_y_coord[0] = np.max(min_y_coords[:, 0])
                    else:
                        max_y_coord[0] = np.max(max_y_coords[:, 0])
                    vectors.append([list(min_y_coord), list(max_y_coord), distance])
                cv2.line(image, min_y_coord, max_y_coord, BLUE_COLOR, 2)
            return vectors, image
        except Exception as e:
            self.get_logger().error(f"Failed to compute vectors from image: {str(e)}")
            return [], image

    def process_image_for_edge_vectors(self, image):
        try:
            self.image_height, self.image_width, color_count = image.shape
            self.lower_image_height = int(
                self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE
            )
            self.upper_image_height = int(self.image_height - self.lower_image_height)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold_black = 25
            thresh = cv2.threshold(gray, threshold_black, 255, cv2.THRESH_BINARY_INV)[1]
            thresh = thresh[self.image_height - self.lower_image_height :]
            image = image[self.image_height - self.lower_image_height :]
            
            # Use fixed trackbar values
            points = np.float32([
                (TRACKBAR_VALUES["Width Top"], TRACKBAR_VALUES["Height Top"]),
                (self.image_width - TRACKBAR_VALUES["Width Top"], TRACKBAR_VALUES["Height Top"]),
                (TRACKBAR_VALUES["Width Bottom"], TRACKBAR_VALUES["Height Bottom"]),
                (self.image_width - TRACKBAR_VALUES["Width Bottom"], TRACKBAR_VALUES["Height Bottom"])
            ])
            
            vectors, image = self.compute_vectors_from_image(image, thresh)

            if not vectors:
                return []

            vectors = sorted(vectors, key=lambda x: x[2])
            half_width = self.image_width / 2
            vectors_left = [i for i in vectors if ((i[0][0] + i[1][0]) / 2) < half_width]
            vectors_right = [i for i in vectors if ((i[0][0] + i[1][0]) / 2) >= half_width]
            final_vectors = []
            for vectors_inst in [vectors_left, vectors_right]:
                if vectors_inst:
                    cv2.line(image, vectors_inst[0][0], vectors_inst[0][1], GREEN_COLOR, 2)
                    vectors_inst[0][0][1] += self.upper_image_height
                    vectors_inst[0][1][1] += self.upper_image_height
                    final_vectors.append(vectors_inst[0][:2])

            self.publish_debug_image(self.publisher_thresh_image, thresh)
            self.publish_debug_image(self.publisher_vector_image, image)

            return final_vectors
        except Exception as e:
            self.get_logger().error(f"Failed to process image for edge vectors: {str(e)}")
            return []

    def camera_image_callback(self, message):
        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            vectors = self.process_image_for_edge_vectors(image)

            vectors_message = EdgeVectors()
            vectors_message.image_height = image.shape[0]
            vectors_message.image_width = image.shape[1]
            vectors_message.vector_count = 0
            if len(vectors) > 0:
                vectors_message.vector_1[0].x = float(vectors[0][0][0])
                vectors_message.vector_1[0].y = float(vectors[0][0][1])
                vectors_message.vector_1[1].x = float(vectors[0][1][0])
                vectors_message.vector_1[1].y = float(vectors[0][1][1])
                vectors_message.vector_count += 1
            if len(vectors) > 1:
                vectors_message.vector_2[0].x = float(vectors[1][0][0])
                vectors_message.vector_2[0].y = float(vectors[1][0][1])
                vectors_message.vector_2[1].x = float(vectors[1][1][0])
                vectors_message.vector_2[1].y = float(vectors[1][1][1])
                vectors_message.vector_count += 1
            self.publisher_edge_vectors.publish(vectors_message)
            print(vectors_message)
        except Exception as e:
            self.get_logger().error(f"Failed to process camera image callback: {str(e)}")

    def publish_curve(self, curve):
        try:
            curve_message = Float32()
            curve_message.data = curve
            self.turn_publisher.publish(curve_message)
        except Exception as e:
            self.get_logger().error(f"Failed to publish curve: {str(e)}")

    def thresholding(self, img):
        try:
            imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lowerWhite = np.array([0, 0, 230])
            upperWhite = np.array([180, 25, 255])
            lowerGray = np.array([0, 0, 95])
            upperGray = np.array([180, 15, 105])
            lowerAdditional = np.array([0, 0, 110])
            upperAdditional = np.array([180, 20, 170])
            maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
            maskGray = cv2.inRange(imgHsv, lowerGray, upperGray)
            maskAdditional = cv2.inRange(imgHsv, lowerAdditional, upperAdditional)
            maskCombined = cv2.bitwise_or(maskWhite, maskGray)
            maskCombined = cv2.bitwise_or(maskCombined, maskAdditional)
            return maskCombined
        except Exception as e:
            self.get_logger().error(f"Failed to threshold image: {str(e)}")
            return np.zeros_like(img)

    def warpImg(self, img, points, w, h, inv=False):
        try:
            pts1 = np.float32(points)
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            if inv:
                matrix = cv2.getPerspectiveTransform(pts2, pts1)
            else:
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarp = cv2.warpPerspective(img, matrix, (w, h))
            return imgWarp
        except Exception as e:
            self.get_logger().error(f"Failed to warp image: {str(e)}")
            return img

    def drawPoints(self, img, points):
        try:
            for x in range(4):
                cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
            return img
        except Exception as e:
            self.get_logger().error(f"Failed to draw points: {str(e)}")
            return img

    def getHistogram(self, img, display=False, minPer=0.1, region=1):
        try:
            if region == 1:
                histValues = np.sum(img, axis=0)
            else:
                histValues = np.sum(img[img.shape[0] // region:, :], axis=0)
            maxValue = np.max(histValues)
            minValue = minPer * maxValue
            indexArray = np.where(histValues >= minValue)
            basePoint = int(np.average(indexArray))
            if display:
                imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                for x, intensity in enumerate(histValues):
                    cv2.line(imgHist, (x, img.shape[0]), (x, img.shape[0] - intensity // 255 // region), (255, 0, 255), 1)
                cv2.line(imgHist, (basePoint, img.shape[0]), (basePoint, 0), (0, 255, 255), 2)
                return basePoint, imgHist
            return basePoint
        except Exception as e:
            self.get_logger().error(f"Failed to get histogram: {str(e)}")
            return 0, img

    def getLaneCurve(self, message):
        try:
            myData = np.frombuffer(message.data, np.uint8)
            img = cv2.imdecode(myData, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (480, 240))
            imgThres = self.thresholding(img)
            hT, wT, c = img.shape
            points = np.float32([
                (TRACKBAR_VALUES["Width Top"], TRACKBAR_VALUES["Height Top"]),
                (wT - TRACKBAR_VALUES["Width Top"], TRACKBAR_VALUES["Height Top"]),
                (TRACKBAR_VALUES["Width Bottom"], TRACKBAR_VALUES["Height Bottom"]),
                (wT - TRACKBAR_VALUES["Width Bottom"], TRACKBAR_VALUES["Height Bottom"])
            ])
            imgWarp = self.warpImg(imgThres, points, wT, hT)
            imgWarpPoints = self.drawPoints(img, points)
            middlePoint, imgHist = self.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
            curveAveragePoint, imgHist = self.getHistogram(imgWarp, display=True, minPer=0.9)
            curveRaw = curveAveragePoint - middlePoint
            self.curveList.append(curveRaw)
            if len(self.curveList) > self.avgVal:
                self.curveList.pop(0)
            curve = int(sum(self.curveList) / len(self.curveList))
            curve = curve / 100
            if curve > 1:
                curve = 1
            if curve < -1:
                curve = -1

            self.publish_curve(curve)
            
            # cv2.imshow('Thres', imgThres)
            # cv2.imshow('Warp', imgWarp)
            # cv2.imshow('Warp Points', imgWarpPoints)
            # cv2.imshow('Histogram', imgHist)
            # cv2.imshow('Vid', img)
            # cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to get lane curve: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorsPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
