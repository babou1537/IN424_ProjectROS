__author__ = "Johvany Gustave, Jonatan Alvarez"
__copyright__ = "Copyright 2025, IN424, IPSA 2025"
__credits__ = ["Johvany Gustave", "Jonatan Alvarez"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

import rclpy
from rclpy.node import Node
import heapq
import math
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion
import numpy as np

# Constants
UNEXPLORED_SPACE_VALUE = -1
FREE_SPACE_VALUE = 0
OBSTACLE_VALUE = 100
PATH_VALUE = 50
OTHER_AGENT_VALUE = 25
EXPLORED_SPACE_VALUE = 1
TEMPORARY_GOAL_VALUE = 75

class Agent(Node):
    def __init__(self):
        Node.__init__(self, "Agent")
        
        self.load_params()

        # Initialize attributes
        self.agents_pose = [None]*self.nb_agents
        self.x = self.y = self.yaw = None
        self.current_target = None
        self.avoiding_obstacle = False
        self.avoided_obstacles = set()

        # Publishers and Subscribers
        self.map_agent_pub = self.create_publisher(OccupancyGrid, f"/{self.ns}/map", 1)
        self.init_map()

        odom_methods_cb = [self.odom1_cb, self.odom2_cb, self.odom3_cb]
        for i in range(1, self.nb_agents + 1):  
            self.create_subscription(Odometry, f"/bot_{i}/odom", odom_methods_cb[i-1], 1)
        
        if self.nb_agents != 1:
            self.create_subscription(OccupancyGrid, "/merged_map", self.merged_map_cb, 1)
        
        self.create_subscription(LaserScan, f"{self.ns}/laser/scan", self.lidar_cb, qos_profile=qos_profile_sensor_data)
        self.cmd_vel_pub = self.create_publisher(Twist, f"{self.ns}/cmd_vel", 1)

        # Timers
        self.create_timer(0.2, self.map_update)
        self.create_timer(0.5, self.strategy)
        self.create_timer(0.5, self.publish_maps)

    def load_params(self):
        self.declare_parameters(
            namespace="",
            parameters=[
                ("ns", rclpy.Parameter.Type.STRING),
                ("robot_size", rclpy.Parameter.Type.DOUBLE),
                ("env_size", rclpy.Parameter.Type.INTEGER_ARRAY),
                ("nb_agents", rclpy.Parameter.Type.INTEGER),
            ]
        )
        self.ns = self.get_parameter("ns").value
        self.robot_size = self.get_parameter("robot_size").value
        self.env_size = self.get_parameter("env_size").value
        self.nb_agents = self.get_parameter("nb_agents").value

    def init_map(self):
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_msg.info.resolution = self.robot_size
        self.map_msg.info.height = int(self.env_size[0]/self.map_msg.info.resolution)
        self.map_msg.info.width = int(self.env_size[1]/self.map_msg.info.resolution)
        self.map_msg.info.origin.position.x = -self.env_size[1]/2
        self.map_msg.info.origin.position.y = -self.env_size[0]/2
        self.map_msg.info.origin.orientation.w = 1.0
        self.map = np.ones(shape=(self.map_msg.info.height, self.map_msg.info.width), dtype=np.int8)*UNEXPLORED_SPACE_VALUE
        self.w, self.h = self.map_msg.info.width, self.map_msg.info.height

    def merged_map_cb(self, msg):
        received_map = np.flipud(np.array(msg.data).reshape(self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                if (received_map[i, j] != UNEXPLORED_SPACE_VALUE) and ((self.map[i, j] == UNEXPLORED_SPACE_VALUE) or (self.map[i, j] == FREE_SPACE_VALUE)):
                    self.map[i, j] = received_map[i, j]

    def odom1_cb(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 1:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                                            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[0] = (x, y)

    def odom2_cb(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 2:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                                            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[1] = (x, y)

    def odom3_cb(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 3:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                                            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[2] = (x, y)

    def lidar_cb(self, msg):
        self.lidar_data = msg

    def map_update(self):
        if self.x is None or self.y is None or not hasattr(self, 'lidar_data'):
            return

        agent_x = int((self.x - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
        agent_y = self.map_msg.info.height - int((self.y - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution) -1

        for i, distance in enumerate(self.lidar_data.ranges):
            angle = self.lidar_data.angle_min + i * self.lidar_data.angle_increment

            if self.lidar_data.range_min < distance < self.lidar_data.range_max:
                x_offset = distance * np.cos(angle + self.yaw)
                y_offset = distance * np.sin(angle + self.yaw)
            else:
                distance = self.lidar_data.range_max
                x_offset = distance * np.cos(angle + self.yaw)
                y_offset = distance * np.sin(angle + self.yaw)

            map_x = int((self.x + x_offset - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
            map_y = self.map_msg.info.height - int((self.y + y_offset - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution) -1

            for agent_pos in self.agents_pose:
                agent_x_pos, agent_y_pos = agent_pos
                if agent_x_pos is not None and agent_y_pos is not None:
                    agent_map_x = int((agent_x_pos - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
                    agent_map_y = self.map_msg.info.height - int((agent_y_pos - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution) -1

                    if abs(map_x - agent_map_x) <= 1 and abs(map_y - agent_map_y) <= 1:
                        self.map[map_y, map_x] = FREE_SPACE_VALUE
                        break
            else:
                if self.lidar_data.range_min < distance < self.lidar_data.range_max:
                    if 0 <= map_x < self.map_msg.info.width and 0 <= map_y < self.map_msg.info.height:
                        self.map[map_y, map_x] = OBSTACLE_VALUE

            num_steps = int(distance / self.map_msg.info.resolution)
            for step in range(num_steps):
                interp_x = int(agent_x + (map_x - agent_x) * step / num_steps)
                interp_y = int(agent_y + (map_y - agent_y) * step / num_steps)

                if 0 <= interp_x < self.map_msg.info.width and 0 <= interp_y < self.map_msg.info.height:
                    self.map[interp_y, interp_x] = FREE_SPACE_VALUE

    def publish_maps(self):
        self.map_msg.data = np.flipud(self.map).flatten().tolist()
        self.map_agent_pub.publish(self.map_msg)

    def strategy(self):
        if self.x is None or self.y is None or not hasattr(self, 'map'):
            return
            
        if self.avoiding_obstacle:
            if not self.is_near_obstacle():
                self.avoiding_obstacle = False
                return
            else:
                self.avoid_obstacle()
                return
        
        frontier = self.find_nearest_frontier()
        if frontier is None:
            self.get_logger().info("Toute la carte a été explorée !")
            self.stop_robot()
            return 
        
        self.current_target = frontier
        start = self.world_to_map(self.x, self.y)
        path = self.a_star_search(start, frontier)

        if path and len(path) > 1:
            next_cell = path[1]
            next_pos = self.map_to_world(next_cell[0], next_cell[1])
            
            distance_to_target = math.sqrt((self.x - next_pos[0])**2 + (self.y - next_pos[1])**2)
            if distance_to_target < self.robot_size * 2:
                self.mark_area_as_explored(frontier)
                return
                
            if self.is_near_obstacle():
                self.avoiding_obstacle = True
                self.avoid_obstacle()
                return
                
            self.move_to_position(next_pos)
        else:
            if not self.avoid_obstacle():
                self.mark_area_as_explored(frontier)

    def find_nearest_frontier(self):
        agent_x, agent_y = self.world_to_map(self.x, self.y)
        closest = None
        min_dist = float('inf')
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        
        for y in range(self.h):
            for x in range(self.w):
                if self.map[y, x] == FREE_SPACE_VALUE:
                    is_frontier = False
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.w and 0 <= ny < self.h:
                            if self.map[ny, nx] == UNEXPLORED_SPACE_VALUE:
                                is_frontier = True
                                break
                    
                    if is_frontier:
                        dist = (x - agent_x)**2 + (y - agent_y)**2
                        if dist < min_dist:
                            min_dist = dist
                            closest = (x, y)
        
        return closest

    def a_star_search(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
    
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
    
        open_set_hash = {start}
    
        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)
        
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < self.w and 0 <= neighbor[1] < self.h):
                    continue
                
                if self.map[neighbor[1], neighbor[0]] == OBSTACLE_VALUE:
                    continue
                
                move_cost = 1 if dx == 0 or dy == 0 else math.sqrt(2)
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
    
        return None

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def world_to_map(self, world_x, world_y):
        map_x = int((world_x - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
        map_y = self.map_msg.info.height - int((world_y - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution) - 1
        return (map_x, map_y)

    def map_to_world(self, map_x, map_y):
        world_x = self.map_msg.info.origin.position.x + (map_x + 0.5) * self.map_msg.info.resolution
        world_y = self.map_msg.info.origin.position.y + ((self.map_msg.info.height - map_y - 0.5) * self.map_msg.info.resolution)
        return (world_x, world_y)

    def move_to_position(self, target_pos):
        cmd = Twist()
        
        dx = target_pos[0] - self.x
        dy = target_pos[1] - self.y
        
        distance = math.sqrt(dx**2 + dy**2)
        target_yaw = math.atan2(dy, dx)
        
        angle_diff = self.normalize_angle(target_yaw - self.yaw)
        
        angle_threshold = 0.1
        distance_threshold = self.robot_size * 0.5
        
        if distance > distance_threshold:
            if abs(angle_diff) > angle_threshold:
                cmd.angular.z = 0.3 * angle_diff
            else:
                cmd.linear.x = 0.2 * min(distance, 1.0)
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def is_near_obstacle(self):
        robot_x, robot_y = self.world_to_map(self.x, self.y)
        detection_radius = 2
        
        for dy in range(-detection_radius, detection_radius + 1):
            for dx in range(-detection_radius, detection_radius + 1):
                nx, ny = robot_x + dx, robot_y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    if self.map[ny, nx] == OBSTACLE_VALUE:
                        return True
        return False

    def avoid_obstacle(self):
        robot_x, robot_y = self.world_to_map(self.x, self.y)
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        
        scored_directions = []
        for dx, dy in directions:
            nx, ny = robot_x + dx, robot_y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                if self.map[ny, nx] == FREE_SPACE_VALUE:
                    unexplored_score = self.count_unexplored_around(nx, ny)
                    scored_directions.append((unexplored_score, (dx, dy)))
        
        if not scored_directions:
            return False
        
        scored_directions.sort(reverse=True)
        best_dir = scored_directions[0][1]
        
        target_x = robot_x + best_dir[0] * 3
        target_y = robot_y + best_dir[1] * 3
        target_pos = self.map_to_world(target_x, target_y)
        self.move_to_position(target_pos)
        return True

    def count_unexplored_around(self, x, y, radius=3):
        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    if self.map[ny, nx] == UNEXPLORED_SPACE_VALUE:
                        count += 1
        return count

    def mark_area_as_explored(self, position):
        x, y = position
        radius = 3
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    if self.map[ny, nx] == FREE_SPACE_VALUE:
                        self.map[ny, nx] = EXPLORED_SPACE_VALUE

def main():
    rclpy.init()
    node = Agent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

