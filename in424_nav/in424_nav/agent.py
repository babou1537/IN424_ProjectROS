__author__ = "Johvany Gustave, Jonatan Alvarez"
__copyright__ = "Copyright 2025, IN424, IPSA 2025"
__credits__ = ["Johvany Gustave", "Jonatan Alvarez"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"


import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, String

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion

import numpy as np
from collections import deque # Gestion of list
import json, time

from .my_common import *    #common variables are stored here

import heapq
from collections import defaultdict


class Agent(Node):
    """
    This class is used to define the behavior of ONE agent
    """
    def __init__(self):
        Node.__init__(self, "Agent")
        
        self.load_params()

        # Agent state tracking
        self.agents_pose = [None]*self.nb_agents    # Array of tuples storing all agents' positions [(x1,y1), (x2,y2)...]
        self.x = self.y = self.yaw = None           # Current agent's pose (x,y,yaw)
        self.last_positions = deque(maxlen=5)       # Recent position history for progress tracking

        # Navigation parameters
        self.nav_params = {
        'target_refresh_rate': 1.0,  # Target update frequency (seconds)
        'linear_speed': 2,           # Maximum linear velocity (m/s)
        'angular_speed': 2,          # Maximum angular velocity (rad/s)
        'arrival_threshold': 0.3,    # Distance threshold for target reached (meters)
        'progress_threshold': 0.5    # Minimum progress distance (meters)
        }

        # Frontier selection weights (MODIFIED for Testing)
        self.frontier_weights = {
            'distance': 0.3,       # Distance weight factor
            'size': 0.8,           # Frontier size weight factor
            'accessibility': 0.3,  # Accessibility weight factor
            'max_distance': 20,    # Normalized maximum distance (in grid cells)
            'depth_penalty': 0.2   # Depth penalty coefficient
        }

        
        # LIDAR configuration
        self.lidar_params = {
            'min_distance': 1.1,  # Minimum obstacle distance (meters)
            'fov_degrees': 120,   # Field of view (±degrees)
        }

        # Mapping system
        self.map_agent_pub = self.create_publisher(OccupancyGrid, f"/{self.ns}/map", 1)
        self.init_map()

        # Pose subscribers for all agents
        odom_methods_cb = [self.odom1_cb, self.odom2_cb, self.odom3_cb]
        for i in range(1, self.nb_agents + 1):  
            self.create_subscription(Odometry, f"/bot_{i}/odom", odom_methods_cb[i-1], 1)
        
        # Multi-agent coordination
        if self.nb_agents != 1:
            self.create_subscription(OccupancyGrid, "/merged_map", self.merged_map_cb, 1)
        
        # Sensor subscriptions
        self.create_subscription(LaserScan, f"{self.ns}/laser/scan", self.lidar_cb, qos_profile=qos_profile_sensor_data)
        self.cmd_vel_pub = self.create_publisher(Twist, f"{self.ns}/cmd_vel", 1)

        # Frontier exploration system
        self.assigned_frontier = None    # Currently assigned exploration target
        self.shared_frontiers = {}       # Dictionary of frontiers being explored { (x,y): robot_id }
        self.target_lock_duration = 5.0  # Frontier reservation duration (seconds)
        

        # Frontier communication system
        self.frontier_pub = self.create_publisher(Int32MultiArray, '/shared_frontiers', 10)
        self.id_pub = self.create_publisher(String, '/frontier_owners', 10)
        self.create_subscription(Int32MultiArray, '/shared_frontiers', self.frontiers_cb, 10)
        self.create_subscription(String, '/frontier_owners', self.owners_cb, 10)

        # Timer-based system updates
        self.frontier_timer = self.create_timer(self.nav_params['target_refresh_rate'], self.update_frontiers)
        self.create_timer(0.3, self.map_update)         # Map update (3.33Hz)
        self.create_timer(1.0, self.update_frontiers)   # Frontier update (1Hz)
        self.create_timer(0.1, self.navigation_loop)    # Navigation control (10Hz)
        self.create_timer(0.5, self.publish_maps)       # Map publishing (2Hz)
    

    def load_params(self):
        """ Load parameters from launch file """
        self.declare_parameters(    #A node has to declare ROS parameters before getting their values from launch files
            namespace="",
            parameters=[
                ("ns", rclpy.Parameter.Type.STRING),    #robot's namespace: either 1, 2 or 3
                ("robot_size", rclpy.Parameter.Type.DOUBLE),    #robot's diameter in meter
                ("env_size", rclpy.Parameter.Type.INTEGER_ARRAY),   #environment dimensions (width height)
                ("nb_agents", rclpy.Parameter.Type.INTEGER),    #total number of agents (this agent included) to map the environment
            ]
        )

        #Get launch file parameters related to this node
        self.ns = self.get_parameter("ns").value
        self.robot_size = self.get_parameter("robot_size").value
        self.env_size = self.get_parameter("env_size").value
        self.nb_agents = self.get_parameter("nb_agents").value
    

    def init_map(self):
        """ Initialize the map to share with others if it is bot_1 """
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"    #set in which reference frame the map will be expressed (DO NOT TOUCH)
        self.map_msg.header.stamp = self.get_clock().now().to_msg() #get the current ROS time to send the msg
        self.map_msg.info.resolution = self.robot_size  #Map cell size corresponds to robot size
        self.map_msg.info.height = int(self.env_size[0]/self.map_msg.info.resolution)   #nb of rows
        self.map_msg.info.width = int(self.env_size[1]/self.map_msg.info.resolution)    #nb of columns
        self.map_msg.info.origin.position.x = -self.env_size[1]/2   #x and y coordinates of the origin in map reference frame
        self.map_msg.info.origin.position.y = -self.env_size[0]/2
        self.map_msg.info.origin.orientation.w = 1.0    #to have a consistent orientation in quaternion: x=0, y=0, z=0, w=1 for no rotation
        self.map = np.ones(shape=(self.map_msg.info.height, self.map_msg.info.width), dtype=np.int8)*UNEXPLORED_SPACE_VALUE #all the cells are unexplored initially
        self.w, self.h = self.map_msg.info.width, self.map_msg.info.height  
    

    def merged_map_cb(self, msg):
        """ 
            Get the current common map and update ours accordingly.
            This method is automatically called whenever a new message is published on the topic /merged_map.
            'msg' is a nav_msgs/msg/OccupancyGrid message.
        """
        received_map = np.flipud(np.array(msg.data).reshape(self.h, self.w))    #convert the received list into a 2D array and reverse rows
        for i in range(self.h):
            for j in range(self.w):
                # if (self.map[i, j] == UNEXPLORED_SPACE_VALUE) and (received_map[i, j] != UNEXPLORED_SPACE_VALUE):
                if (received_map[i, j] != UNEXPLORED_SPACE_VALUE) and ((self.map[i, j] == UNEXPLORED_SPACE_VALUE) or (self.map[i, j] == FREE_SPACE_VALUE) or (self.map[i,j] == FRONTIER_VALUE)):
                    self.map[i, j] = received_map[i, j]


    def odom1_cb(self, msg):
        """ 
            @brief Get agent 1 position.
            This method is automatically called whenever a new message is published on topic /bot_1/odom.
            
            @param msg This is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 1:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[0] = (x, y)
        # self.get_logger().info(f"Agent 1: ({x:.2f}, {y:.2f})")
    

    def odom2_cb(self, msg):
        """ 
            @brief Get agent 2 position.
            This method is automatically called whenever a new message is published on topic /bot_2/odom.
            
            @param msg This is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 2:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[1] = (x, y)
        # self.get_logger().info(f"Agent 2: ({x:.2f}, {y:.2f})")


    def odom3_cb(self, msg):
        """ 
            @brief Get agent 3 position.
            This method is automatically called whenever a new message is published on topic /bot_3/odom.
            
            @param msg This is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 3:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[2] = (x, y)
        # self.get_logger().info(f"Agent 3: ({x:.2f}, {y:.2f})")


    def lidar_cb(self, msg):
        """ 
            @brief Get messages from LIDAR topic.
            This method is automatically called whenever a new message is published on topic /bot_x/laser/scan, where 'x' is either 1, 2 or 3.
            
            @param msg This is a sensor_msgs/msg/LaserScan message.
        """
        self.lidar_data = msg


    def publish_maps(self):
        """ 
            Publish updated map to topic /bot_x/map, where x is either 1, 2 or 3.
            This method is called periodically (1Hz) by a ROS2 timer, as defined in the constructor of the class.
        """
        self.map_msg.data = np.flipud(self.map).flatten().tolist()  #transform the 2D array into a list to publish it
        self.map_agent_pub.publish(self.map_msg)                    #publish map to other agents

    ## ====================== ======= ========================= ##
    ## ====================== MAPPING ========================= ##
    ## ====================== ======= ========================= ##
    def map_update(self):
        """
        Updates the agent's occupancy grid map using LiDAR scan data.
        
        Performs raycasting to:
        - Mark detected obstacles
        - Clear free spaces along LiDAR rays
        - Handle special cases for other agents' positions
        - Maintain an updated occupancy grid
        """
        if self.x is None or self.y is None or not hasattr(self, 'lidar_data'):
            return  # Wait for valid agent position and LiDAR data

        # Convert agent's position to map grid coordinates
        agent_x = int((self.x - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
        agent_y = self.map_msg.info.height - int((self.y - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution) -1

        # Process each LiDAR measurement
        for i, distance in enumerate(self.lidar_data.ranges):
            angle = self.lidar_data.angle_min + i * self.lidar_data.angle_increment

            if self.lidar_data.range_min < distance < self.lidar_data.range_max:
                # Valid obstacle detection
                x_offset = distance * np.cos(angle + self.yaw)
                y_offset = distance * np.sin(angle + self.yaw)
            else:
                # No obstacle detected - use max range
                distance = self.lidar_data.range_max
                x_offset = distance * np.cos(angle + self.yaw)
                y_offset = distance * np.sin(angle + self.yaw)

            # Calculate detected point in map coordinates
            map_x = int((self.x + x_offset - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
            map_y = self.map_msg.info.height - int((self.y + y_offset - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution) -1

            # Check if detection matches other agents' positions
            for agent_pos in self.agents_pose:
                agent_x_pos, agent_y_pos = agent_pos
                if agent_x_pos is not None and agent_y_pos is not None:
                    agent_map_x = int((agent_x_pos - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
                    agent_map_y = self.map_msg.info.height - int((agent_y_pos - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution) -1

                    # If detection is near another agent, mark as free space
                    if abs(map_x - agent_map_x) <= 1 and abs(map_y - agent_map_y) <= 1:
                        self.map[map_y, map_x] = FREE_SPACE_VALUE
                        break

            else:  # If not another agent, mark as obstacle if valid detection
                if self.lidar_data.range_min < distance < self.lidar_data.range_max:
                    if 0 <= map_x < self.map_msg.info.width and 0 <= map_y < self.map_msg.info.height:
                        self.map[map_y, map_x] = OBSTACLE_VALUE

            # Raycasting: Mark free space along the LiDAR beam
            num_steps = int(distance / self.map_msg.info.resolution)
            for step in range(num_steps):
                interp_x = int(agent_x + (map_x - agent_x) * step / num_steps)
                interp_y = int(agent_y + (map_y - agent_y) * step / num_steps)

                if 0 <= interp_x < self.map_msg.info.width and 0 <= interp_y < self.map_msg.info.height:
                    self.map[interp_y, interp_x] = FREE_SPACE_VALUE
            
        # Publish the updated map
        self.publish_maps()


    def world_to_map(self, world_x, world_y):
        """
        Converts world coordinates (meters) to map grid indices.
        
        Args:
            world_x (float): X coordinate in world frame (meters)
            world_y (float): Y coordinate in world frame (meters)
            
        Returns:
            tuple: (map_x, map_y) grid indices in the occupancy grid
        """
        map_x = int((world_x - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
        map_y = self.map_msg.info.height - int((world_y - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution)
        return map_x, map_y


    ## ====================== ======= ========================= ##
    ## ===================== FRONTIERE ======================== ##
    ## ====================== ======= ========================= ##
    def update_frontiers(self):
        """
        Optimized frontier exploration update with efficient target management.
        
        Performs:
        - Rate-limited frontier updates based on navigation parameters
        - Frontier detection and evaluation
        - Dynamic target reallocation when:
          * No current target assigned
          * Current target reached
          * Insufficient progress toward target
        """
        # Only process at configured refresh rate
        now = self.get_clock().now()
        if hasattr(self, 'last_frontier_update'):
            elapsed = (now - self.last_frontier_update).nanoseconds / 1e9
            if elapsed < self.nav_params['target_refresh_rate']:
                return
        
        self.last_frontier_update = now
        
        # Detect new frontiers in current map
        frontiers = self.detect_frontiers()
        if not frontiers:
            return
        
        # Reallocate target only when necessary
        if (self.assigned_frontier is None 
            or self.is_target_reached()
            or not self.is_making_progress()):
            
            self.allocate_frontiers(frontiers)


    def evaluate_frontier(self, frontier):
        """
        Enhanced frontier scoring system that evaluates exploration targets based on:
        1. Distance from agent
        2. Size of unexplored area
        3. Accessibility/obstacle density
        4. Depth penalty for occluded areas
        
        Args:
            frontier (tuple): (i,j) map coordinates of frontier candidate
            
        Returns:
            float: Comprehensive score combining all factors with normalization
        """
        i, j = frontier
        
        # 1. Distance component
        agent_x, agent_y = self.world_to_map(self.x, self.y)
        distance = np.sqrt((i - agent_x)**2 + (j - agent_y)**2)
        norm_distance = min(distance / self.frontier_weights['max_distance'], 1.0)  # Normalized [0-1] with 20 cell cap
        
        # 2. Frontier size component
        frontier_size = 0
        unexplored_directions = []
        
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:  # Check 4-connected neighbors
            ni, nj = i+di, j+dj
            if 0 <= ni < self.h and 0 <= nj < self.w:
                if self.map[ni,nj] == UNEXPLORED_SPACE_VALUE:
                    frontier_size += 1
                    unexplored_directions.append((di,dj))
        
        # Bonus for large contiguous unexplored areas
        contiguous_bonus = 1.0 + 0.2 * len(unexplored_directions)
        norm_size = min(frontier_size / 4, 1.0) * contiguous_bonus  # Normalized [0-1.2]

        # 3. Accessibility component
        obstacle_count = sum(
            1 for di, dj in [(-1,-1),(1,1),(-1,1),(1,-1)]  # Check diagonal neighbors
            if 0 <= i+di < self.h and 0 <= j+dj < self.w
            and self.map[i+di,j+dj] == OBSTACLE_VALUE
        )
        
        # Exponential penalty for enclosed areas
        accessibility = max(0.1, 1.0 - 0.3**obstacle_count)  # Range [0.1-1.0]

        # 4. Depth penalty factor (dynamic)
        depth_factor = 1.0
        for di, dj in unexplored_directions:
            for step in range(1, 5):  # Look 4 cells ahead
                ni, nj = i + di*step, j + dj*step
                if 0 <= ni < self.h and 0 <= nj < self.w:
                    if self.map[ni,nj] == OBSTACLE_VALUE:
                        depth_factor *= self.frontier_weights['depth_penalty']
                        break

        # Additional obstacle density check in 5x5 area
        obstacle_count = sum(
            1 for di in range(-2, 3) for dj in range(-2, 3)
            if 0 <= i+di < self.h and 0 <= j+dj < self.w
            and self.map[i+di, j+dj] == OBSTACLE_VALUE
            )
        
        # Final weighted score calculation
        score = (
            norm_distance * self.frontier_weights['distance'] +
            norm_size * self.frontier_weights['size'] + 
            accessibility * self.frontier_weights['accessibility']
        ) * depth_factor

        # Apply obstacle density penalty
        score *= max(0.1, 1.0 - obstacle_count / 10)

        return score


    def detect_frontiers(self):
        """
        Detects frontier cells between explored and unexplored areas.
        
        A frontier cell is defined as:
        - Free space (not obstacle)
        - Has at least one unexplored neighbor (8-direction check)
        
        Returns:
            list: Coordinates (i,j) of all detected frontier cells
        """
        frontiers = []
        for i in range(1, self.h-1):  # Skip map borders
            for j in range(1, self.w-1):
                # Must be free space AND have unexplored neighbors
                if self.map[i,j] == FREE_SPACE_VALUE:
                    # Check 8-connected neighborhood for unexplored areas
                    has_unknown = any(
                        self.map[i+di,j+dj] == UNEXPLORED_SPACE_VALUE
                        for di,dj in [(-1,-1), (-1,0), (-1,1), (0,-1), 
                                    (0,1), (1,-1), (1,0), (1,1)]
                        if 0 <= i+di < self.h and 0 <= j+dj < self.w
                    )
                    if has_unknown:
                        frontiers.append((i,j))
        return frontiers


    def publish_frontiers(self, frontiers):
        """
        Publishes frontier information to other agents.
        
        Sends two messages:
        1. Frontier coordinates as flattened list of integers
        2. Frontier ownership assignments as JSON string
        """
        # Publish frontier positions (flattened coordinates)
        pos_msg = Int32MultiArray()
        pos_msg.data = [coord for frontier in frontiers for coord in frontier]
        self.frontier_pub.publish(pos_msg)
        
        # Publish ownership assignments (JSON format)
        assign_msg = String()
        assign_msg.data = json.dumps({self.ns: frontiers})
        self.id_pub.publish(assign_msg)


    def frontiers_cb(self, msg):
        """
        Callback for receiving frontier positions from other agents.
        
        Args:
            msg (Int32MultiArray): Contains alternating x,y coordinates of frontiers
        """
        frontiers = [(msg.data[i], msg.data[i+1]) for i in range(0, len(msg.data), 2)]
        self.known_frontiers = frontiers  # Update global frontier list


    def owners_cb(self, msg):
        """
        Callback for receiving frontier ownership assignments.
        
        Args:
            msg (String): JSON string mapping robot IDs to their assigned frontiers
            
        Handles JSON decode errors with warning log messages.
        """
        try:
            assignments = json.loads(msg.data)
            for robot_id, frontiers in assignments.items():
                for frontier in frontiers:
                    self.shared_frontiers[tuple(frontier)] = robot_id
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"JSON decode error: {str(e)}")


    def allocate_frontiers(self, frontiers):
        """
        Selects and assigns the optimal frontier for exploration.
        
        Performs:
        1. Filters out conflicting/inaccessible frontiers
        2. Scores remaining frontiers using evaluate_frontier()
        3. Applies proximity penalty factor
        4. Selects highest scoring frontier
        5. Triggers immediate navigation
        
        Args:
            frontiers (list): List of candidate frontier coordinates
            
        Returns:
            tuple: Coordinates of selected frontier or None if none available
        """
        if not frontiers:
            return None

        # Filter out frontiers with conflicts (already assigned)
        available = [
            (self.evaluate_frontier(f), f) 
            for f in frontiers
            if not self.check_frontier_conflict(f)
        ]

        if not available:
            self.get_logger().warn("No available frontiers (all taken or conflicts)")
            return None

        # Score all available frontiers with proximity penalty
        best_score, best_frontier = max(available, key=lambda x: (
            x[0] * self.proximity_penalty(x[1])  # Apply distance penalty factor
        ))

        # Sort by score and take best (redundant with max() but kept for debugging)
        available.sort(reverse=True, key=lambda x: x[0])
        best_score, best_frontier = available[0]
        
        # Convert grid coordinates to world coordinates
        world_x = best_frontier[0] * self.map_msg.info.resolution + self.map_msg.info.origin.position.x
        world_y = (self.map_msg.info.height - best_frontier[1]) * self.map_msg.info.resolution + self.map_msg.info.origin.position.y
        
        self.assigned_frontier = (world_x, world_y)  # Store in world coordinates
        self.publish_frontiers([best_frontier])  # Broadcast assignment
        
        # Trigger immediate navigation update
        self.navigation_loop()
        
        self.get_logger().info(
            f"New target: ({world_x:.2f}, {world_y:.2f}) | Score: {best_score:.2f}",
            throttle_duration_sec=1.0
        )
        return best_frontier
    
    
    def check_frontier_conflict(self, frontier):
        """
        Checks if a frontier is already claimed by another agent.
        
        Args:
            frontier (tuple): (i,j) coordinates of frontier to check
            
        Returns:
            bool: True if frontier is claimed by another agent, False otherwise
        """
        return (tuple(frontier) in self.shared_frontiers and 
                self.shared_frontiers[tuple(frontier)] != self.ns)


    def proximity_penalty(self, frontier):
        """
        Applies distance-based penalty to frontiers near other agents' targets.
        
        Args:
            frontier (tuple): (i,j) coordinates of frontier to evaluate
            
        Returns:
            float: Penalty factor (0.5 if too close to others, 1.0 otherwise)
        """
        min_distance = 10  # In grid cells
        
        for assigned_pos, robot_id in self.shared_frontiers.items():
            if robot_id == self.ns:  # Skip our own claims
                continue
                
            # Calculate distance to other agents' frontiers
            dist = np.sqrt((frontier[0]-assigned_pos[0])**2 + (frontier[1]-assigned_pos[1])**2)
            if dist < min_distance:
                return 0.5  # Apply 50% score penalty if too close
                
        return 1.0  # No penalty


    ## ====================== ======= ========================= ##
    ## ===================== NAVIGATION ======================= ##
    ## ====================== ======= ========================= ##
    def navigation_loop(self):
        """
        Main navigation loop with safety checks.
        
        Handles:
        - Emergency obstacle avoidance
        - Path collision detection
        - Normal navigation to frontier
        """
        # Safety checks take priority
        if self.is_wall_ahead():
            self.emergency_avoidance()
            return
        
        if self.check_path_collision():
            self.replan_path()
            return
        
        # Skip if no target assigned
        if not self.assigned_frontier:
            return

        # Normal navigation execution
        if not hasattr(self, 'current_path') or not self.current_path:
            self.plan_path_to_frontier()
        self.follow_path()


    def plan_path_to_frontier(self):
        """
        A* path planning with 2-cell safety buffer around obstacles.
        
        Features:
        - 8-direction movement (including diagonals)
        - Obstacle safety margin
        - Other robot avoidance
        - Coordinate validation
        """
        if not self.assigned_frontier or None in (self.x, self.y, self.yaw):
            return

        # Convert positions to map coordinates
        start_x, start_y = self.world_to_map(self.x, self.y)
        target_x, target_y = self.world_to_map(*self.assigned_frontier)

        # Validate positions
        if not (0 <= start_x < self.w and 0 <= start_y < self.h) or \
        not (0 <= target_x < self.w and 0 <= target_y < self.h) or \
        self.is_near_obstacle(target_x, target_y, radius=2):  # Target proximity check
            self.get_logger().warn("Target position too close to obstacle")
            self.current_path = []
            return

        # Modified A* implementation
        open_set = []
        heapq.heappush(open_set, (0, (start_x, start_y)))
        
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[(start_x, start_y)] = 0
        
        f_score = defaultdict(lambda: float('inf'))
        f_score[(start_x, start_y)] = self.heuristic(start_x, start_y, target_x, target_y)
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == (target_x, target_y):
                self.current_path = self.reconstruct_path(came_from, current)
                return
                
            # Check all 8 directions
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Boundary check
                if not (0 <= neighbor[0] < self.w and 0 <= neighbor[1] < self.h):
                    continue
                    
                # Safety checks
                if (self.is_near_obstacle(*neighbor, radius=2) or 
                    self.is_near_other_robot(*neighbor, safety_radius=3)):  # Robot avoidance
                    continue
                    
                # Movement cost (diagonal = 1.4, straight = 1.0)
                move_cost = 1.4 if dx !=0 and dy !=0 else 1.0
                tentative_g = g_score[current] + move_cost
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(*neighbor, target_x, target_y)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        self.get_logger().warn("No safe path found (safety margin enforced)")
        self.current_path = []


    def heuristic(self, x1, y1, x2, y2):
        """Standard Euclidean distance heuristic for A* pathfinding"""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


    def reconstruct_path(self, came_from, current):
        """
        Reconstructs path from A* results and converts to world coordinates.
        
        Args:
            came_from (dict): Path history from A*
            current (tuple): Target position (x,y)
            
        Returns:
            list: World coordinate path [(x1,y1), (x2,y2), ...]
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        
        # Convert path to world coordinates
        world_path = []
        for x, y in path:
            world_x = x * self.map_msg.info.resolution + self.map_msg.info.origin.position.x
            world_y = (self.map_msg.info.height - y) * self.map_msg.info.resolution + self.map_msg.info.origin.position.y
            world_path.append((world_x, world_y))
        
        return world_path


    def emergency_avoidance(self):
        """
        Emergency obstacle avoidance maneuver with 3-phase recovery:
        1. Immediate backward motion (1 second)
        2. Controlled rotation until path is clear (5s timeout)
        3. Careful forward movement (2-3 grid cells)
        
        Resets navigation target after completion.
        """
        # Phase 1: Immediate backward motion (1 second)
        cmd = Twist()
        cmd.linear.x = -0.3  # Moderate reverse speed
        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

        # Phase 2: Rotation until path is clear (5s timeout)
        cmd.linear.x = 0.0
        cmd.angular.z = 0.8  # Moderate rotation speed
        start_time = time.time()
        while (time.time() - start_time < 5.0):  # 5 second timeout
            if not self.is_wall_ahead():
                break
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

        # Phase 3: Careful forward movement (2-3 grid cells)
        cmd.angular.z = 0.0
        cmd.linear.x = 0.2  # Reduced speed for precision
        
        # Convert grid cells to meters
        target_distance = 2.5 * self.map_msg.info.resolution  
        start_x, start_y = self.x, self.y
        distance_traveled = 0.0
        
        while distance_traveled < target_distance:
            current_x, current_y = self.x, self.y  # Updated via odometry
            distance_traveled = np.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

        # Final stop and target reset
        self.cmd_vel_pub.publish(Twist())
        self.assigned_frontier = self.get_safe_target()


    def get_safe_target(self):
        """
        Generates a safe navigation target in opposite direction with safety margin.
        
        Returns:
            tuple: (x,y) world coordinates within map bounds
        """
        # Calculate direction 180° ± 30° from current heading
        angle = self.yaw + np.pi + np.random.uniform(-0.5, 0.5)  
        dist = 8 * self.map_msg.info.resolution  # 8 grid cells
        
        # Calculate target position
        x = self.x + dist * np.cos(angle)
        y = self.y + dist * np.sin(angle)
        
        # Constrain to map boundaries with 0.5m margin
        return (
            max(self.map_msg.info.origin.position.x + 0.5, 
                min(x, self.map_msg.info.origin.position.x + self.w * self.map_msg.info.resolution - 0.5)),
            max(self.map_msg.info.origin.position.y + 0.5,
                min(y, self.map_msg.info.origin.position.y + self.h * self.map_msg.info.resolution - 0.5))
        )


    def is_wall_ahead(self):
        """
        Reliable wall detection within ±angle° FOV (angle° total) using LIDAR data.
        
        Returns:
            bool: True if obstacle detected within safety distance, False otherwise
        """
        if not hasattr(self, 'lidar_data'):
            return False

        # 1. Extract LIDAR parameters
        num_readings = len(self.lidar_data.ranges)
        angle_increment = self.lidar_data.angle_increment
        
        # 2. Calculate indices for ±angle° FOV
        center_index = num_readings // 2
        degrees_per_index = np.degrees(angle_increment)
        indices_30deg = int((self.lidar_params['fov_degrees']/2) / degrees_per_index)
        
        # Ensure indices stay within valid range
        start_idx = max(0, center_index - indices_30deg)
        end_idx = min(num_readings, center_index + indices_30deg)

        # 3. Analyze measurements in angle° FOV
        danger_zones = []
        for i in range(start_idx, end_idx):
            distance = self.lidar_data.ranges[i]
            
            # Valid distance check:
            # - Finite value (not NaN/inf)
            # - Within danger threshold
            if (np.isfinite(distance) and 
                0 < distance < self.lidar_params['min_distance']):
                danger_zones.append(distance)
        
        # 4. Trigger condition:
        # - Minimum 3 valid detections
        # - Average below safety threshold with margin
        return (len(danger_zones) >= 3 and
                np.mean(danger_zones) < self.lidar_params['min_distance'] - 0.2)


    def check_path_collision(self):
        """
        Checks for obstacles in the immediate path (next 3-4 steps).
        
        Returns:
            bool: True if collision detected, False if path is clear
        """
        if not hasattr(self, 'current_path') or len(self.current_path) < 2:
            return False
            
        # Check first 4 waypoints in current path
        for wx, wy in self.current_path[:4]:
            x, y = self.world_to_map(wx, wy)
            # Verify waypoint is within map bounds and not blocked
            if not (0 <= x < self.w and 0 <= y < self.h):
                return True
            if self.map[y, x] == OBSTACLE_VALUE:
                return True
        return False

    
    def get_opposite_target(self):
        """
        Generates a retreat target in the opposite direction from current heading.
        
        Returns:
            tuple: (x,y) world coordinates of safe retreat position
        """
        # Convert current position to map coordinates
        x, y = self.world_to_map(self.x, self.y)
        
        # Calculate opposite direction vector
        angle = self.yaw + np.pi  # 180° from current heading
        dist = 10 * self.map_msg.info.resolution  # 10 grid cells ahead
        
        # Calculate target in map coordinates
        target_x = x + dist * np.cos(angle)
        target_y = y + dist * np.sin(angle)
        
        # Convert back to world coordinates
        world_x = target_x * self.map_msg.info.resolution + self.map_msg.info.origin.position.x
        world_y = (self.h - target_y) * self.map_msg.info.resolution + self.map_msg.info.origin.position.y
        
        return (world_x, world_y)
    

    def follow_path(self):
        """
        Executes path following with obstacle avoidance and adaptive speed control.
        
        Features:
        - Lookahead point tracking
        - Adaptive speed based on target proximity
        - Smooth angular control
        - Integrated safety checks
        """
        # Safety checks take priority
        if self.is_wall_ahead():
            self.emergency_avoidance()
            return
        
        if self.check_path_collision():
            self.replan_path()
            return

        # Skip if no valid path
        if not hasattr(self, 'current_path') or not self.current_path:
            return   
         
        # Find lookahead point on path
        lookahead_dist = 0.5  # meters
        target_point = None
        
        # Search for first point beyond lookahead distance
        for i, (wx, wy) in enumerate(self.current_path):
            dist = np.sqrt((wx - self.x)**2 + (wy - self.y)**2)
            if dist >= lookahead_dist:
                target_point = (wx, wy)
                # Trim passed segments from path
                self.current_path = self.current_path[i:]
                break
        
        # Default to final point if none found
        if not target_point:
            target_point = self.current_path[-1]
            self.current_path = []
        
        # Calculate heading to target
        dx = target_point[0] - self.x
        dy = target_point[1] - self.y
        target_angle = np.arctan2(dy, dx)
        angle_diff = (target_angle - self.yaw + np.pi) % (2 * np.pi) - np.pi  # Normalized to [-π, π]
        
        cmd_vel = Twist()
        
        # Adaptive speed control
        dist_to_target = np.sqrt((self.assigned_frontier[0] - self.x)**2 + 
                                (self.assigned_frontier[1] - self.y)**2)
        
        # Slow down when approaching final target
        if dist_to_target < 1.0:
            cmd_vel.linear.x = 0.1  # Slow speed
        else:
            cmd_vel.linear.x = 0.2  # Normal speed
            
        # Angular control with different gain regimes
        if abs(angle_diff) > 0.2:  # Large angle correction (~11°)
            cmd_vel.angular.z = 0.5 * np.clip(angle_diff, -1, 1)  # Higher gain
        else:
            cmd_vel.angular.z = 0.3 * angle_diff  # Fine adjustment
        
        self.cmd_vel_pub.publish(cmd_vel)


    def replan_path(self):
        """Replanning with penalty on the danger zone"""
        self.plan_path_to_frontier()
        self.get_logger().warn("Replanification du chemin détourné !")


    ## ===================== DETECTION ======================= ##
    def is_near_obstacle(self, x, y, radius=2):
        """Checks if a cell is too close to an obstacle"""
        for di in range(-radius, radius+1):
            for dj in range(-radius, radius+1):
                nx, ny = x + di, y + dj
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    if self.map[ny, nx] == OBSTACLE_VALUE:
                        return True
        return False
    

    def is_near_other_robot(self, x, y, safety_radius=5):
        """
        Check if the position (x,y) is too close to another robot.
        safety_radius: number of boxes to avoid
        """
        if not hasattr(self, 'agents_pose') or self.agents_pose is None:
            return False
        
        for i, (other_x, other_y) in enumerate(self.agents_pose):
            # Skip soi-même et les robots sans position
            if i == int(self.ns[-1]) - 1 or other_x is None or other_y is None:
                continue
            
            # Conversion en coordonnées carte si nécessaire
            other_mx, other_my = self.world_to_map(other_x, other_y)
            
            # Distance en cases
            distance = max(abs(x - other_mx), abs(y - other_my))
            
            if distance < safety_radius:
                return True
                
        return False


    def is_target_reached(self):
        """Checks if the current target is reached"""
        if not self.assigned_frontier:
            return False
            
        current_x, current_y = self.world_to_map(self.x, self.y)
        target_x, target_y = self.assigned_frontier
        
        return (abs(current_x - target_x) < 2 and 
                abs(current_y - target_y) < 2)


    def is_making_progress(self):
        """Checks progress towards the current target"""
        if not hasattr(self, 'last_positions'):
            self.last_positions = deque(maxlen=5)
        
        current_pos = self.world_to_map(self.x, self.y)
        self.last_positions.append(current_pos)
        
        if len(self.last_positions) < 5:
            return True
        
        old_x, old_y = self.last_positions[0]
        current_x, current_y = current_pos
        distance = np.sqrt((current_x-old_x)**2 + (current_y-old_y)**2)
        return distance > 2
    



def main():
    rclpy.init()
    node = Agent()

    start_time = time.time()
    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        # Calculs lorsqu'on arrête avec Ctrl+C
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calcul du pourcentage exploré
        total_cells = node.map.shape[0] * node.map.shape[1]
        explored_cells = np.sum(node.map != UNEXPLORED_SPACE_VALUE)
        explored_percent = (explored_cells / total_cells) * 100
        
        # Affichage des résultats
        node.get_logger().info("\n" + "="*50)
        node.get_logger().info(f"Temps d'exécution: {execution_time:.2f} secondes")
        node.get_logger().info(f"Surface explorée: {explored_percent:.2f}%")
        node.get_logger().info("Détail:")
        node.get_logger().info(f"- Cellules explorées: {explored_cells}/{total_cells}")
        node.get_logger().info(f"- Résolution carte: {node.map_msg.info.resolution:.3f} m/cellule")
        node.get_logger().info("="*50 + "\n")
        
    finally:
        node.destroy_node()
        rclpy.shutdown()
