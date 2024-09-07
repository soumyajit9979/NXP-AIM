# Copyright 2024 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy

import math

import time

from synapse_msgs.msg import EdgeVectors
from synapse_msgs.msg import TrafficStatus
from sensor_msgs.msg import LaserScan
from datetime import datetime
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

QOS_PROFILE_DEFAULT = 10

PI = math.pi

LEFT_TURN = +1.0
RIGHT_TURN = -1.0

TURN_MIN = 0.0
TURN_MAX = 1.0
SPEED_MIN = 0.0
SPEED_MAX = 1.0
SPEED_25_PERCENT = SPEED_MAX / 4
SPEED_50_PERCENT = SPEED_25_PERCENT * 2
SPEED_75_PERCENT = SPEED_25_PERCENT * 3
SPEED_125_PERCENT = 1.25
THRESHOLD_OBSTACLE_VERTICAL = 1.0
THRESHOLD_OBSTACLE_HORIZONTAL = 0.25


class LineFollower(Node):
	""" Initializes line follower node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('line_follower')

		try:
			# Subscription for edge vectors.
			self.subscription_vectors = self.create_subscription(
				EdgeVectors,
				'/edge_vectors',
				self.edge_vectors_callback,
				QOS_PROFILE_DEFAULT)

			self.subscription_vectors = self.create_subscription(
				Float32,
				'/turn',
				self.get_turn_callback,
				QOS_PROFILE_DEFAULT)

			# Publisher for joy (for moving the rover in manual mode).
			self.publisher_joy = self.create_publisher(
				Joy,
				'/cerebri/in/joy',
				QOS_PROFILE_DEFAULT)

			# Subscription for traffic status.
			self.subscription_traffic = self.create_subscription(
				TrafficStatus,
				'/traffic_status',
				self.traffic_status_callback,
				QOS_PROFILE_DEFAULT)

			# Subscription for LIDAR data.
			self.subscription_lidar = self.create_subscription(
				LaserScan,
				'/scan',
				self.lidar_callback,
				QOS_PROFILE_DEFAULT)

			self.traffic_status = TrafficStatus()

			self.obstacle_detected = False
			self.obstacle_detected_right = False
			self.obstacle_detected_left = False
			self.left_obs_val = 0.0
			self.right_obs_val = 0.0

			self.ramp_detected = False
			self.ramp_detected_time = None
			self.integral = 0
			self.previous_error = 0
			self.start_time = datetime.now().timestamp()
			self.restart_time = True

			self.state = 0
			self.state1_current_time = 0	
			self.test_turn = 0.0
			self.val = 0.0
			self.range_inf_detected = False
		except Exception as e:
			self.get_logger().error(f"Initialization error: {e}")


	""" Operates the rover in manual mode by publishing on /cerebri/in/joy.

		Args:
			speed: the speed of the car in float. Range = [-1.0, +1.0];
				Direction: forward for positive, reverse for negative.
			turn: steer value of the car in float. Range = [-1.0, +1.0];
				Direction: left turn for positive, right turn for negative.

		Returns:
			None
	"""
	def rover_move_manual_mode(self, speed, turn):
		try:
			msg = Joy()

			msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]

			msg.axes = [0.0, speed, 0.0, turn]

			self.publisher_joy.publish(msg)
		except Exception as e:
			self.get_logger().error(f"Manual mode movement error: {e}")

	""" Analyzes edge vectors received from /edge_vectors to achieve line follower application.
		It checks for existence of ramps & obstacles on the track through instance members.
			These instance members are updated by the lidar_callback using LIDAR data.
		The speed and turn are calculated to move the rover using rover_move_manual_mode.

		Args:
			message: "~/cognipilot/cranium/src/synapse_msgs/msg/EdgeVectors.msg"

		Returns:
			None
	"""
	def get_turn_callback(self, message):
		try:
			turn = message
			self.test_turn = -1 * round(float(turn.data), 2)
			# self.get_logger().info(f"Turn: {self.test_turn}")

			return turn
		except Exception as e:
			self.get_logger().error(f"Turn callback error: {e}")


	def edge_vectors_callback(self, message):
		if self.traffic_status.stop_sign:
			self.traffic_status.stop_sign = True
			speed = 0.00
			turn = 0.00
			sys.exit()
		try:
			# Default values for speed and turn
			speed = SPEED_75_PERCENT
			turn = TURN_MIN

			vectors = message
			half_width = vectors.image_width / 2

			if self.ramp_detected:
				current_time = time.time()
				self.get_logger().info("Ramp started")
				if current_time - self.ramp_detected_time >= 7:
					self.ramp_detected = False
					self.ramp_detected_time = None
					self.get_logger().info("7 seconds done")

				if vectors.vector_count == 0:  # No vectors detected
					speed = SPEED_25_PERCENT
				elif vectors.vector_count == 1:  # Curve detected
					deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
					speed = SPEED_25_PERCENT
					if deviation < 0:
						turn = -0.75
					elif deviation > 0:
						turn = 0.75

				elif vectors.vector_count == 2:  # Straight line detected
					middle_x_left = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2
					middle_x_right = (vectors.vector_2[0].x + vectors.vector_2[1].x) / 2
					middle_x = (middle_x_left + middle_x_right) / 2
					deviation = half_width - middle_x
					# Speed control
					speed = SPEED_25_PERCENT
					# Adjust turn based on deviation from the center
					if abs(deviation) < half_width / 10:
						turn = TURN_MIN
					else:
						turn = deviation / half_width
					turn = max(-1, min(1, turn))  # Ensure turn value is between -1 and 1
				if self.obstacle_detected_right :
					turn = abs(turn + (1/(self.right_obs_val*10 + 0.01)))
					self.obstacle_detected_right = False
					print("correction_right_turn", turn)
				if self.obstacle_detected_left :
					turn = -abs(turn - (1/(self.left_obs_val*10 + 0.01)))
					self.obstacle_detected_left = False
					print("correction_left_turn", turn)

				speed = SPEED_50_PERCENT  # You may adjust the speed for ramps if needed
			else:
				# If not on a ramp, calculate turn based on youtube camera algo
				turn = self.test_turn
				speed = SPEED_MAX 

				if abs(turn) > 0.2:
					speed = SPEED_75_PERCENT

				if self.traffic_status.stop_sign:
					speed = SPEED_MIN
					self.get_logger().info("Stop sign detected")
				if self.obstacle_detected_left and self.left_obs_val < self.right_obs_val:
					turn = -abs(turn - 0.3)
					# turn = -abs(turn - (1/(self.left_obs_val*10 + 0.01)))
					# turn = -1.0
					speed = SPEED_50_PERCENT
					print("correction_left_turn", turn)
					self.obstacle_detected_left = False
				if self.obstacle_detected_right and self.right_obs_val < self.left_obs_val:
					turn =  abs(turn +0.3)
					# turn = abs(turn + (1/(self.right_obs_val*10 + 0.01)))
					# turn = 1.0
					print("correction_right_turn", turn)
					speed = SPEED_50_PERCENT
					self.obstacle_detected_right = False
				

			# Publish the movement command
			# self.get_logger().info(f"Speed: {speed}, Turn: {turn}")
			# print(turn)
			self.rover_move_manual_mode(speed, turn)
		except Exception as e:
			self.get_logger().error(f"Edge vectors callback error: {e}")


	def traffic_status_callback(self, message):
		try:
			self.traffic_status = message
		except Exception as e:
			self.get_logger().error(f"Traffic status callback error: {e}")


	def lidar_callback(self, message):
		try:
			shield_vertical = 4
			shield_horizontal = 1
			theta = math.atan(shield_vertical / shield_horizontal)

			length = float(len(message.ranges))
			ranges = message.ranges[int(length / 4): int(3 * length / 4)]

			length = float(len(ranges))
			front_ranges = ranges[int(length * theta / PI): int(length * (PI - theta) / PI)]
			side_ranges_right = ranges[0: int(length * theta / PI)]
			side_ranges_left = ranges[int(length * (PI - theta) / PI):]

			angle = theta - PI / 2
			ran = message.ranges[90:270]

			self.val = message.ranges[180]
			left_range = message.ranges[181:360]
			right_range = message.ranges[0:179]

			for i in range(65,103):
				if ran[i] < 1.8:
					self.yes_flag = True
				else:
					self.yes_flag = False
					break

			if message.ranges[180] == float('inf'):
				self.range_inf_detected = True
			elif message.ranges[180] != float('inf') and not self.ramp_detected:
				self.range_inf_detected = False

			if (message.ranges[180]) < 2.0 and (message.ranges[180]) > 0.81 and self.yes_flag:
				self.ramp_detected = True
				self.ramp_detected_time = time.time()
				return

			for i in range(len(front_ranges)):
				if front_ranges[i] < THRESHOLD_OBSTACLE_VERTICAL:
					self.obstacle_detected = True
					# print("Front")
					return
				angle += message.angle_increment


			# side_ranges_left.reverse()
			left_range.reverse()
			# for side_ranges in [side_ranges_left, side_ranges_right]:
			# 	angle = 0.0
			# 	for i in range(len(side_ranges)):
			# 		if side_ranges[i] < THRESHOLD_OBSTACLE_HORIZONTAL:
			# 			self.obstacle_detected = True
			# 			print("side")
			# 			return
			# 		angle += message.angle_increment
			for i in range(len(left_range)):
				if left_range[i] <= 0.35 and left_range[i] >=0: 
					self.obstacle_detected_left = True
					self.left_obs_val = abs(min(left_range))
					print("left side")
					print("distance_left", self.left_obs_val, i)
					print("distance_right", self.right_obs_val, i)
					break
				
				
			for i in range(len(right_range)):
				if right_range[i] <= 0.35 and right_range[i] >=0:
					self.obstacle_detected_right = True
					self.right_obs_val = abs(min(right_range))
					print("right side")
					print("distance_left", self.left_obs_val, i)
					print("distance_right", self.right_obs_val, i)
					break
				


			# for side_ranges in [left_range, right_range]:
			# 	angle = 0.0
			# 	for i in range(len(side_ranges)):
			# 		if side_ranges[i] < THRESHOLD_OBSTACLE_HORIZONTAL :
			# 			self.obstacle_detected = True
			# 			print("left side")
			# 			return
			# 		if side_ranges[i] < THRESHOLD_OBSTACLE_HORIZONTAL and side_ranges == 1:
			# 			self.obstacle_detected = True
			# 			print("right side")
			# 			return
			# 		angle += message.angle_increment
			# self.obstacle_detected_left = False
			# self.obstacle_detected_right = False

			self.obstacle_detected = False
			self.yes_flag = False
		except Exception as e:
			self.get_logger().error(f"LIDAR callback error: {e}")


def main(args=None):
	try:
		rclpy.init(args=args)

		line_follower = LineFollower()

		rclpy.spin(line_follower)

		line_follower.destroy_node()
		rclpy.shutdown()
	except Exception as e:
		print(f"Main function error: {e}")


if __name__ == '__main__':
	main()
