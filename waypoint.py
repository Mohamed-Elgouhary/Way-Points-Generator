#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import atexit
from os.path import expanduser
from time import gmtime, strftime
from numpy import linalg as LA
from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import pandas as pd
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Constants
home = expanduser('~')
filename = strftime(home + '/mo_ws/src/wp-%Y-%m-%d-%H-%M', gmtime()) + '.csv'
file = open(filename, 'w')
file.write('# x_m, y_m, w_tr_right_m, w_tr_left_m\n')

class WaypointsLogger(Node):
    def __init__(self):
        super().__init__('waypoints_logger')

        # Declare parameters
        self.declare_parameter('is_real', True)
        self.declare_parameter('min_spacing', 0.01)
        self.is_real = self.get_parameter('is_real').value
        self.min_spacing = self.get_parameter('min_spacing').value

        # Topics
        odom_topic = '/pf/pose/odom' if self.is_real else '/ego_racecar/odom'
        self.subscription_odom = self.create_subscription(
            Odometry, odom_topic, self.process_odometry, 10)
        self.subscription_scan = self.create_subscription(
            LaserScan, '/scan', self.process_scan, 10)

        # Internal state
        self.latest_scan = None
        self.latest_odometry = None
        self.previous_point = None
        self.waypoints = []  # Each item: (x, y, left_width, right_width)

    def process_scan(self, scan_data):
        """Estimate track widths from LiDAR data."""
        ranges = np.array(scan_data.ranges)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))

        valid_indices = ~np.isinf(ranges) & ~np.isnan(ranges)
        ranges = ranges[valid_indices]
        angles = angles[valid_indices]

        if len(ranges) == 0:
            self.get_logger().warn('No valid ranges in scan data.')
            return

        left_indices = angles > 0
        right_indices = angles <= 0

        # Use 20th percentile instead of min for stability
        self.left_width = np.percentile(ranges[left_indices], 20) if np.any(left_indices) else float('inf')
        self.right_width = np.percentile(ranges[right_indices], 20) if np.any(right_indices) else float('inf')

        self.latest_scan = True
        self.save_waypoint()

    def process_odometry(self, odometry_data):
        self.latest_odometry = odometry_data
        self.save_waypoint()

    def save_waypoint(self):
        """Save a waypoint if both LiDAR and odometry data are available."""
        if self.latest_scan and self.latest_odometry:
            data = self.latest_odometry
            x = data.pose.pose.position.x
            y = data.pose.pose.position.y
            quaternion = [
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w
            ]
            yaw = euler_from_quaternion(quaternion)[2]

            if self.previous_point is None or LA.norm([x - self.previous_point[0], y - self.previous_point[1]]) >= self.min_spacing:
                self.get_logger().info(f'Saving waypoint: x={x:.2f}, y={y:.2f}, left_width={self.left_width:.2f}, right_width={self.right_width:.2f}')
                self.waypoints.append((x, y, self.left_width, self.right_width))
                self.previous_point = (x, y)

            self.latest_scan = None
            self.latest_odometry = None

    def filter_outliers(self, points, threshold=2.0):
        """Filter out waypoints that are far from the moving average."""
        if len(points) < 3:
            return points
        filtered = [points[0]]
        for i in range(1, len(points)):
            dist = LA.norm(np.array(points[i]) - np.array(filtered[-1]))
            if dist < threshold:
                filtered.append(points[i])
        return np.array(filtered)

    def save_and_interpolate(self):
        """Filter, interpolate, save to CSV, and plot the trajectory."""
        if len(self.waypoints) > 1:
            wp = np.array(self.waypoints)
            x, y = wp[:, 0], wp[:, 1]
            left_w, right_w = wp[:, 2], wp[:, 3]

            # Filter outliers
            filtered_points = self.filter_outliers(wp[:, :2], threshold=2.0)
            x_filt, y_filt = filtered_points[:, 0], filtered_points[:, 1]

            # Spline interpolation
            tck, u = splprep([x_filt, y_filt], s=0.2)
            unew = np.linspace(0, 1, 5000)
            x_new, y_new = splev(unew, tck)

            # Use average width (as we can’t interpolate widths easily)
            avg_left = np.mean(left_w)
            avg_right = np.mean(right_w)

            for i in range(len(x_new)):
                file.write(f'{x_new[i]}, {y_new[i]}, {avg_right}, {avg_left}\n')

            self.get_logger().info(f'Waypoints filtered and saved with {len(x_new)} points.')

            # Save and show the trajectory plot
            fig_path = filename.replace('.csv', '.png')
            plt.figure()
            plt.plot(x, y, 'o', label='Original Waypoints')
            plt.plot(x_filt, y_filt, 'x', label='Filtered Waypoints')
            plt.plot(x_new, y_new, '-', label='Interpolated Path')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('Waypoint Path')
            plt.legend()
            plt.grid()
            plt.savefig(fig_path)
            plt.show()

            if len(x_new) < 10:
                self.get_logger().warn('Generated trajectory has fewer points than the required horizon (10).')

def shutdown():
    file.close()
    print('Goodbye. File saved.')

def main(args=None):
    atexit.register(shutdown)
    print('Starting waypoint logger...')
    rclpy.init(args=args)
    node = WaypointsLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_and_interpolate()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
