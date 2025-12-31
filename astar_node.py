import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from math import atan2, sqrt, sin, pi
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import heapq
import numpy as np
import cv2

class NodeAStar:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
        

    def __eq__(self, other): 
        return self.position == other.position

    def __lt__(self, other): 
        return self.f < other.f


class IntegratedNavigation(Node):
    def __init__(self):
        super().__init__('integrated_navigation')
        self.lookahead_dist = 0.5
        self.linear_vel = 0.2
        self.stop_tolerance = 0.15
        self.safety_margin = 4

        self.map_data = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.map_width = 0
        self.map_height = 0

        self.current_pose = None
        self.current_yaw = 0.0
        self.global_path = []
        self.path_index = 0
        self.cam_count = 0
        qos_cam = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST,depth=1)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/planned_path', 10)
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.sub_pose = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.sub_cam = self.create_subscription(Image, '/image_raw', self.camera_callback, qos_cam)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.bridge = CvBridge()
        self.green_detected = False
        self.stopped = False

        self.get_logger().info("Let's Run!")

    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.map_data = np.array(msg.data).reshape((self.map_height, self.map_width))

    def pose_callback(self, msg):
        self.current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        q = msg.pose.pose.orientation
        self.current_yaw = atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))

    def goal_callback(self, msg):
        if self.map_data is None or self.current_pose is None:
            return
        goal_pose = [msg.pose.position.x, msg.pose.position.y]
        start_grid = self.world_to_grid(self.current_pose)
        goal_grid = self.world_to_grid(goal_pose)

        self.get_logger().info("Calculating Path...")
        path_grid = self.run_astar(start_grid, goal_grid)

        if path_grid:
            self.global_path = [self.grid_to_world(p) for p in path_grid]
            self.path_index = 0
            self.publish_path_viz()
            self.get_logger().info("Path Found! Go!")
        else:
            self.get_logger().warn("No Path Found.")

    def check_safety(self, y, x):
        margin = self.safety_margin
        for r in range(y - margin, y + margin + 1):
            for c in range(x - margin, x + margin + 1):
                if 0 <= r < self.map_height and 0 <= c < self.map_width:
                    if self.map_data[r][c] != 0:
                        return False
        return True

    def run_astar(self, start, end):
        if not (0 <= start[0] < self.map_height and 0 <= start[1] < self.map_width):
            return None
        if not (0 <= end[0] < self.map_height and 0 <= end[1] < self.map_width):
            return None

        start_node = NodeAStar(None, start)
        end_node = NodeAStar(None, end)
        open_list = []
        heapq.heappush(open_list, start_node)
        visited = set()
        moves = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]

        while open_list:
            current_node = heapq.heappop(open_list)
            if current_node.position in visited:
                continue
            visited.add(current_node.position)

            if current_node.position == end_node.position:
                path = []
                current = current_node
                while current:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            for move in moves:
                ny, nx = current_node.position[0] + move[0], current_node.position[1] + move[1]
                if not (0 <= ny < self.map_height and 0 <= nx < self.map_width):
                    continue
                if self.map_data[ny][nx] != 0:
                    continue
                if not self.check_safety(ny, nx):
                    continue

                new_node = NodeAStar(current_node, (ny, nx))
                new_node.g = current_node.g + 1
                new_node.h = sqrt((ny - end[0])**2 + (nx - end[1])**2)
                new_node.f = new_node.g + new_node.h
                heapq.heappush(open_list, new_node)
        return None

    def camera_callback(self, msg):
        self.cam_count += 1
        if self.cam_count % 3 != 0:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.resize(frame, (320, 240))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([35, 60, 20])
        upper_bound = np.array([85, 255, 120])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        M = cv2.moments(mask)
        if M["m00"] > 50000:
            self.green_detected = True
        else:
            self.green_detected = False

    def control_loop(self):
        if self.green_detected:
            if not self.stopped:
                self.get_logger().warn("Green detected! Stopping robot.")
                self.stopped = True
            self.stop_robot()
            return
        if not self.green_detected:
            self.stopped = False

        if not self.global_path:
            return

        final_goal = self.global_path[-1]
        dist_to_final = sqrt((final_goal[0]-self.current_pose[0])**2 + (final_goal[1]-self.current_pose[1])**2)
        if dist_to_final < self.stop_tolerance:
            self.global_path = []
            self.stop_robot()
            return

        target_x, target_y = final_goal
        for i in range(self.path_index, len(self.global_path)):
            px, py = self.global_path[i]
            dist = sqrt((px - self.current_pose[0])**2 + (py - self.current_pose[1])**2)
            if dist >= self.lookahead_dist:
                target_x, target_y = px, py
                self.path_index = i
                break

        dx = target_x - self.current_pose[0]
        dy = target_y - self.current_pose[1]
        alpha = atan2(dy, dx) - self.current_yaw

        if alpha > pi: alpha -= 2*pi
        elif alpha < -pi: alpha += 2*pi

        angular_velocity = self.linear_vel * (2.0 * sin(alpha)) / self.lookahead_dist
        cmd = Twist()
        cmd.linear.x = self.linear_vel
        cmd.angular.z = max(min(angular_velocity, 1.0), -1.0)

        self.pub_cmd.publish(cmd)

    def world_to_grid(self, world):
        return (int((world[1]-self.map_origin[1])/self.map_resolution), 
                int((world[0]-self.map_origin[0])/self.map_resolution))

    def grid_to_world(self, grid):
        return [(grid[1]*self.map_resolution)+self.map_origin[0], 
                (grid[0]*self.map_resolution)+self.map_origin[1]]

    def publish_path_viz(self):
        msg = Path()
        msg.header.frame_id = 'map'
        for p in self.global_path:
            ps = PoseStamped()
            ps.pose.position.x, ps.pose.position.y = p[0], p[1]
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def stop_robot(self):
        self.pub_cmd.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()