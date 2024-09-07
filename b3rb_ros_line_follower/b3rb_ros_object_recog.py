import rclpy
from rclpy.node import Node
from synapse_msgs.msg import TrafficStatus

import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage

QOS_PROFILE_DEFAULT = 10
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.60
stop_sign = cv2.CascadeClassifier('install/b3rb_ros_line_follower/share/b3rb_ros_line_follower/stop_sign_classifier_2.xml')


class ObjectRecognizer(Node):
    """ Initializes object recognizer node with the required publishers and subscriptions.

        Returns:
            None
    """
    def __init__(self):
        super().__init__('object_recognizer')

        try:
            # Subscription for camera images.
            self.subscription_camera = self.create_subscription(
                CompressedImage,
                '/camera/image_raw/compressed',
                self.camera_image_callback,
                QOS_PROFILE_DEFAULT)

            # Publisher for traffic status.
            self.publisher_traffic = self.create_publisher(
                TrafficStatus,
                '/traffic_status',
                QOS_PROFILE_DEFAULT)
            self.stop_signs_global = 0
            self.image_height = 0
            self.image_width = 0
            self.lower_image_height = 0
            self.upper_image_height = 0
        except Exception as e:
            self.get_logger().error(f"Initialization error: {e}")

    """ Analyzes the image received from /camera/image_raw/compressed to detect traffic signs.
        Publishes the existence of traffic signs in the image on the /traffic_status topic.

        Args:
            message: "docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CompressedImage.html"

        Returns:
            None
    """
    def camera_image_callback(self, message):
        try:
            # Convert message to an n-dimensional numpy array representation of image.
            np_arr = np.frombuffer(message.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                self.get_logger().warn("Failed to decode image")
                return

            self.image_height, self.image_width, color_count = img.shape
            self.lower_image_height = int(
                self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE
            )
            self.upper_image_height = int(self.image_height - self.lower_image_height)
            image = img[self.image_height - self.lower_image_height:]

            brightness_increase = 50
            bright_img = cv2.add(image, np.array([brightness_increase, brightness_increase, brightness_increase]))
            image_np = np.array(bright_img) 
            # Convert to BGR(openCV format) and then to gray scale
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            # Apply Gaussian filter
            gray_filtered = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detectioncv2.imscv2.ims
            stop_signs = stop_sign.detectMultiScale(gray_filtered, scaleFactor=1.05, minNeighbors=5, minSize=(5, 5))
            
            # self.get_logger().info(f"Number of stop signs detected: {len(stop_signs)}")
            self.get_logger().info(f"Number of stop signs detected: {self.stop_signs_global}")

            # Draw rectangles
            for (x, y, w, h) in stop_signs:
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # cv2.namedWindow("screenshot", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('screenshot', 640, 480)
            # cv2.imshow('screenshot', image_np)
            # cv2.imshow('image_small', image)
            cv2.waitKey(1)  # Add a small delay to process GUI events

            traffic_status_message = TrafficStatus()

            if len(stop_signs) == 1:
                traffic_status_message.stop_sign = True
                self.stop_signs_global = len(stop_signs)

            self.publisher_traffic.publish(traffic_status_message)
        except cv2.error as e:
            self.get_logger().error(f"OpenCV error: {e}")
        except Exception as e:
            self.get_logger().error(f"Exception occurred: {e}")

def main(args=None):
    rclpy.init(args=args)

    object_recognizer = ObjectRecognizer()

    try:
        rclpy.spin(object_recognizer)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        object_recognizer.get_logger().error(f"Main function error: {e}")
    finally:
        # Cleanup
        object_recognizer.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed

if __name__ == '__main__':
    main()
