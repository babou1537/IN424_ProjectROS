__author__ = "Johvany Gustave, Jonatan Alvarez"
__copyright__ = "Copyright 2025, IN424, IPSA 2025"
__credits__ = ["Johvany Gustave", "Jonatan Alvarez"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"


import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion

import numpy as np

from .my_common import *    #common variables are stored here


class Agent(Node):
    """
    This class is used to define the behavior of ONE agent
    """
    def __init__(self):
        Node.__init__(self, "Agent")
        
        self.load_params()

        #initialize attributes
        self.agents_pose = [None]*self.nb_agents    #[(x_1, y_1), (x_2, y_2), (x_3, y_3)] if there are 3 agents
        self.x = self.y = self.yaw = None   #the pose of this specific agent running the node

        self.map_agent_pub = self.create_publisher(OccupancyGrid, f"/{self.ns}/map", 1) #publisher for agent's own map
        self.init_map()

        #Subscribe to agents' pose topic
        odom_methods_cb = [self.odom1_cb, self.odom2_cb, self.odom3_cb]
        for i in range(1, self.nb_agents + 1):  
            self.create_subscription(Odometry, f"/bot_{i}/odom", odom_methods_cb[i-1], 1)
        
        if self.nb_agents != 1: #if other agents are involved subscribe to the merged map topic
            self.create_subscription(OccupancyGrid, "/merged_map", self.merged_map_cb, 1)
        
        self.create_subscription(LaserScan, f"{self.ns}/laser/scan", self.lidar_cb, qos_profile=qos_profile_sensor_data) #subscribe to the agent's own LIDAR topic
        self.cmd_vel_pub = self.create_publisher(Twist, f"{self.ns}/cmd_vel", 1)    #publisher to send velocity commands to the robot

        #Create timers to autonomously call the following methods periodically
        self.create_timer(0.2, self.map_update) #0.2s of period <=> 5 Hz
        self.create_timer(0.5, self.strategy)      #0.5s of period <=> 2 Hz
        self.create_timer(0.5, self.publish_maps) #1Hz
    

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
                if (received_map[i, j] != UNEXPLORED_SPACE_VALUE) and ((self.map[i, j] == UNEXPLORED_SPACE_VALUE) or (self.map[i, j] == FREE_SPACE_VALUE)):
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

    
    # """VERSION 1"""
    # def map_update(self):
    #     """ Met à jour la carte de l'agent avec les données du LiDAR """
    #     if self.x is None or self.y is None or not hasattr(self, 'lidar_data'):
    #         return  # Attendre que l'agent ait une position définie et que les données LiDAR soient disponibles

    #     # Récupérer la position de l'agent en indices de carte
    #     agent_x = int((self.x - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
    #     agent_y = self.map_msg.info.height - int((self.y - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution)

    #     # Vérifier si la position de l'agent est dans les limites
    #     if 0 <= agent_x < self.map_msg.info.width and 0 <= agent_y < self.map_msg.info.height:
    #         self.map[agent_y, agent_x] = FREE_SPACE_VALUE  # Marquer la position de l'agent en blanc

    #     # Parcourir les données LiDAR
    #     """VERSION 1"""
    #     for i, distance in enumerate(self.lidar_data.ranges):
    #         if self.lidar_data.range_min < distance < self.lidar_data.range_max:  
    #             angle = self.lidar_data.angle_min + i * self.lidar_data.angle_increment

    #             # Calculer la position en absolu
    #             x_offset = distance * np.cos(angle + self.yaw)
    #             y_offset = distance * np.sin(angle + self.yaw)

    #             # Convertir en indices de carte
    #             map_x = int((self.x + x_offset - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
    #             map_y = self.map_msg.info.height - int((self.y + y_offset - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution)

    #             # Vérifier si c'est un autre agent au lieu d'un obstacle
    #             for agent_x, agent_y in self.agents_pose:
    #                 if agent_x is not None and agent_y is not None:
    #                     agent_map_x = int((agent_x - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
    #                     agent_map_y = self.map_msg.info.height - int((agent_y - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution)

    #                     if abs(map_x - agent_map_x) <= 1 and abs(map_y - agent_map_y) <= 1:
    #                         self.map[map_y, map_x] = OTHER_AGENT_VALUE  # Marquer en tant que robot
    #                         break  # On sort de la boucle, pas besoin de chercher plus loin

    #             else:  # Si ce n'est pas un autre agent, alors c'est un obstacle
    #                 if 0 <= map_x < self.map_msg.info.width and 0 <= map_y < self.map_msg.info.height:
    #                     self.map[map_y, map_x] = OBSTACLE_VALUE  # Marquer en noir

    #     # Publier la carte mise à jour
    #     self.publish_maps()

    """VERSION 2"""
    def map_update(self):
        """ Met à jour la carte de l'agent avec les données du LiDAR """
        if self.x is None or self.y is None or not hasattr(self, 'lidar_data'):
            return  # Attendre que l'agent ait une position définie et que les données LiDAR soient disponibles

        # Récupérer la position de l'agent en indices de carte
        agent_x = int((self.x - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
        agent_y = self.map_msg.info.height - int((self.y - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution) -1

        # Vérifier si la position de l'agent est dans les limites
        if 0 <= agent_x < self.map_msg.info.width and 0 <= agent_y < self.map_msg.info.height:
            self.map[agent_y, agent_x] = PATH_VALUE  # Marquer la position du robot

        # Parcourir les données LiDAR
        for i, distance in enumerate(self.lidar_data.ranges):
            angle = self.lidar_data.angle_min + i * self.lidar_data.angle_increment

            if self.lidar_data.range_min < distance < self.lidar_data.range_max:
                # Obstacle détecté
                x_offset = distance * np.cos(angle + self.yaw)
                y_offset = distance * np.sin(angle + self.yaw)
            else:
                # Pas d'obstacle => utiliser le rayon max du LIDAR
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

                    # Vérifier si la détection LiDAR correspond à la position d'un autre agent
                    if abs(map_x - agent_map_x) <= 1 and abs(map_y - agent_map_y) <= 1:
                        # self.map[map_y, map_x] = OTHER_AGENT_VALUE  # Autre couleur
                        self.map[map_y, map_x] = FREE_SPACE_VALUE
                        break

            else:  # Si ce n'est pas un autre agent, alors c'est un obstacle
                if self.lidar_data.range_min < distance < self.lidar_data.range_max:
                    if 0 <= map_x < self.map_msg.info.width and 0 <= map_y < self.map_msg.info.height:
                        self.map[map_y, map_x] = OBSTACLE_VALUE  # Marquer en tant qu'obstacle

            # Remplir l'espace entre le robot et la détection avec FREE_SPACE_VALUE
            num_steps = int(distance / self.map_msg.info.resolution)
            for step in range(num_steps):
                interp_x = int(agent_x + (map_x - agent_x) * step / num_steps)
                interp_y = int(agent_y + (map_y - agent_y) * step / num_steps)

                if 0 <= interp_x < self.map_msg.info.width and 0 <= interp_y < self.map_msg.info.height:
                    self.map[interp_y, interp_x] = FREE_SPACE_VALUE
            

        # Publier la carte mise à jour
        self.publish_maps()

    

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
        self.map_agent_pub.publish(self.map_msg)    #publish map to other agents


    def strategy(self):
        """Final wall-proof strategy"""
        if self.x is None or self.y is None or self.yaw is None:
            return

        # 1. Immediate collision prevention (highest priority)
        emergency_status = self.check_wall_collision_risk()
        if emergency_status != "safe":
            self.execute_wall_escape(emergency_status)
            return

        # 2. Get current position in map coordinates
        agent_x, agent_y = self.get_map_coordinates()
        if agent_x is None or agent_y is None:
            return

        # 3. Check if we're in a potential stuck situation
        if self.detect_potential_stuck():
            self.perform_preventative_maneuver()
            return

        # 4. Normal exploration behavior
        frontiers = self.detect_frontiers()
        
        if not frontiers:
            self.adaptive_rotation()
            return

        # Fixed method name to match what's actually defined
        target_frontier = self.select_safest_frontier(frontiers, agent_x, agent_y)
        
        if target_frontier is None:
            self.adaptive_rotation()
            return

        self.intelligent_navigation(target_frontier, agent_x, agent_y)

    def select_safest_frontier(self, frontiers, agent_x, agent_y):
        """Choose frontier with safest path (renamed from select_optimal_frontier)"""
        if not frontiers:
            return None
            
        best_frontier = None
        best_score = -float('inf')
        
        for fx, fy in frontiers:
            # Calculate path safety
            safety = self.calculate_path_safety(agent_x, agent_y, fx, fy)
            
            # Distance factor (prefer closer frontiers)
            distance = np.sqrt((fx - agent_x)**2 + (fy - agent_y)**2)
            dist_score = 1.0 / (1.0 + distance/10.0)
            
            # Combine scores
            combined_score = 0.7*safety + 0.3*dist_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_frontier = (fx, fy)
                
        return best_frontier
    
    def calculate_path_safety(self, start_x, start_y, end_x, end_y):
        """Calculate safety score for a potential path (0-1)"""
        safety = 1.0
        steps = max(abs(end_x - start_x), abs(end_y - start_y))
        
        for i in range(1, steps + 1):
            x = int(start_x + (end_x - start_x) * i / steps)
            y = int(start_y + (end_y - start_y) * i / steps)
            
            # Check if position is within map bounds
            if not (0 <= x < self.w and 0 <= y < self.h):
                return 0.0
            
            # Check for obstacles in 3x3 area around path point
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h:
                        if self.map[ny, nx] == OBSTACLE_VALUE:
                            safety *= 0.8  # Reduce safety for nearby obstacles
        
        return safety

    def check_wall_collision_risk(self):
        """Precise wall collision risk assessment"""
        if not hasattr(self, 'lidar_data'):
            return "safe"
            
        # Analyze different sectors
        sectors = {
            'front': self.lidar_data.ranges[len(self.lidar_data.ranges)//3:2*len(self.lidar_data.ranges)//3],
            'left': self.lidar_data.ranges[:len(self.lidar_data.ranges)//3],
            'right': self.lidar_data.ranges[2*len(self.lidar_data.ranges)//3:]
        }
        
        # Immediate collision danger
        if min(self.lidar_data.ranges) < self.robot_size * 0.7:
            return "emergency"
            
        # Wall hugging detection
        front_avg = np.mean(sectors['front'])
        side_avg = (np.mean(sectors['left']) + np.mean(sectors['right'])) / 2
        
        if front_avg < self.robot_size * 1.2 and side_avg < self.robot_size * 1.0:
            return "corner"
        elif front_avg < self.robot_size * 1.5:
            return "front_wall"
        elif side_avg < self.robot_size * 0.8:
            return "side_wall"
            
        return "safe"

    def execute_wall_escape(self, situation):
        """Specialized wall escape maneuvers"""
        cmd_vel = Twist()
        
        if situation == "emergency":
            # Immediate danger - quick reverse and turn
            cmd_vel.linear.x = -0.15
            cmd_vel.angular.z = 0.4 if np.random.random() > 0.5 else -0.4
        elif situation == "corner":
            # In a corner - strong reverse and turn
            cmd_vel.linear.x = -0.2
            cmd_vel.angular.z = 0.5 if np.mean(self.lidar_data.ranges[:len(self.lidar_data.ranges)//2]) > \
                                   np.mean(self.lidar_data.ranges[len(self.lidar_data.ranges)//2:]) else -0.5
        elif situation == "front_wall":
            # Facing wall - turn in place
            cmd_vel.angular.z = 0.4 if np.random.random() > 0.5 else -0.4
        elif situation == "side_wall":
            # Hugging wall - move away diagonally
            cmd_vel.linear.x = 0.1
            cmd_vel.angular.z = 0.3 if np.mean(self.lidar_data.ranges[:len(self.lidar_data.ranges)//2]) < \
                                   np.mean(self.lidar_data.ranges[len(self.lidar_data.ranges)//2:]) else -0.3
        
        self.cmd_vel_pub.publish(cmd_vel)
        time.sleep(0.5)  # Execute maneuver for fixed time

    def detect_potential_stuck(self):
        """Early detection of potential stuck situations"""
        if not hasattr(self, 'lidar_data'):
            return False
            
        # Check if we're making progress
        if hasattr(self, 'last_positions'):
            if len(self.last_positions) >= 5:
                total_movement = sum(
                    np.sqrt((self.last_positions[i][0]-self.last_positions[i-1][0])**2 +
                           (self.last_positions[i][1]-self.last_positions[i-1][1])**2)
                    for i in range(1, 5)
                )
                if total_movement < self.robot_size * 0.5:
                    return True
        
        # Check if surrounded by obstacles
        if (np.mean(self.lidar_data.ranges) < self.robot_size * 1.5 and
            min(self.lidar_data.ranges) < self.robot_size * 0.8):
            return True
            
        return False

    def perform_preventative_maneuver(self):
        """Prevent getting stuck before it happens"""
        cmd_vel = Twist()
        
        # Find the most open direction
        sector_size = len(self.lidar_data.ranges) // 4
        sector_sums = [
            sum(self.lidar_data.ranges[i*sector_size:(i+1)*sector_size])
            for i in range(4)
        ]
        best_sector = np.argmax(sector_sums)
        
        # Turn toward most open direction
        turn_angle = (best_sector - 1.5) * (np.pi/2)  # Convert sector to angle
        cmd_vel.angular.z = 0.3 * np.sign(turn_angle)
        cmd_vel.linear.x = 0.05  # Small forward motion
        
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Update position history
        if hasattr(self, 'last_positions'):
            self.last_positions.pop(0)
        else:
            self.last_positions = []
        self.last_positions.append((self.x, self.y))

    def intelligent_navigation(self, frontier, agent_x, agent_y):
        """Navigation with continuous wall avoidance"""
        fx, fy = frontier
        target_x = fx * self.map_msg.info.resolution + self.map_msg.info.origin.position.x
        target_y = (self.map_msg.info.height - fy - 1) * self.map_msg.info.resolution + self.map_msg.info.origin.position.y
        
        dx = target_x - self.x
        dy = target_y - self.y
        target_dist = np.sqrt(dx**2 + dy**2)
        target_angle = np.arctan2(dy, dx)
        
        # Get detailed environment info
        min_dist, left_open, right_open = self.analyze_environment()
        
        cmd_vel = Twist()
        angle_diff = self.normalize_angle(target_angle - self.yaw)
        
        # Dynamic speed adjustment
        base_speed = 0.2 * min(1.0, target_dist)
        if min_dist < self.robot_size * 1.5:
            base_speed *= 0.5
        elif min_dist < self.robot_size * 2.0:
            base_speed *= 0.8
            
        # Dynamic angular adjustment
        turn_gain = 0.4
        if min_dist < self.robot_size * 1.2:
            turn_gain = 0.6  # Sharper turns near obstacles
            
        # If path is blocked but target is visible
        if min_dist < self.robot_size * 1.0 and abs(angle_diff) < np.pi/4:
            # Try to "hug" the obstacle
            hug_direction = 1 if left_open > right_open else -1
            cmd_vel.linear.x = base_speed * 0.3
            cmd_vel.angular.z = turn_gain * hug_direction
        else:
            # Normal navigation with obstacle avoidance
            avoidance = self.calculate_avoidance_vector()
            combined_x = dx/target_dist + 0.5*avoidance[0]
            combined_y = dy/target_dist + 0.5*avoidance[1]
            cmd_angle = np.arctan2(combined_y, combined_x)
            angle_diff = self.normalize_angle(cmd_angle - self.yaw)
            
            cmd_vel.linear.x = base_speed * (1 - min(0.5, abs(angle_diff)/(np.pi/2)))
            cmd_vel.angular.z = turn_gain * angle_diff
        
        self.cmd_vel_pub.publish(cmd_vel)

    def analyze_environment(self):
        """Detailed analysis of surrounding environment"""
        if not hasattr(self, 'lidar_data'):
            return float('inf'), 0, 0
            
        min_dist = min(self.lidar_data.ranges)
        
        # Calculate openness on left and right
        left_sector = self.lidar_data.ranges[:len(self.lidar_data.ranges)//2]
        right_sector = self.lidar_data.ranges[len(self.lidar_data.ranges)//2:]
        
        left_open = sum(1 for d in left_sector if d > self.robot_size * 1.5)
        right_open = sum(1 for d in right_sector if d > self.robot_size * 1.5)
        
        return min_dist, left_open, right_open

    def calculate_avoidance_vector(self):
        """Calculate obstacle avoidance direction"""
        if not hasattr(self, 'lidar_data'):
            return (0, 0)
            
        obs_x, obs_y = 0, 0
        for i, distance in enumerate(self.lidar_data.ranges):
            if distance < self.robot_size * 2.0:
                angle = self.lidar_data.angle_min + i * self.lidar_data.angle_increment + self.yaw
                weight = (2.0 - distance/self.robot_size) ** 2
                obs_x -= np.cos(angle) * weight
                obs_y -= np.sin(angle) * weight
                
        norm = np.sqrt(obs_x**2 + obs_y**2)
        if norm > 0:
            obs_x /= norm
            obs_y /= norm
            
        return (obs_x, obs_y)

    def adaptive_rotation(self):
        """Smart rotation that considers walls"""
        if not hasattr(self, 'lidar_data'):
            cmd_vel = Twist()
            cmd_vel.angular.z = 0.3
            self.cmd_vel_pub.publish(cmd_vel)
            return
            
        # Find the most open direction
        sector_size = len(self.lidar_data.ranges) // 8
        sector_avgs = [
            np.mean(self.lidar_data.ranges[i*sector_size:(i+1)*sector_size])
            for i in range(8)
        ]
        best_sector = np.argmax(sector_avgs)
        
        # Calculate target angle for this sector
        target_angle = self.lidar_data.angle_min + (best_sector + 0.5) * sector_size * self.lidar_data.angle_increment
        
        cmd_vel = Twist()
        angle_diff = self.normalize_angle(target_angle)
        
        if abs(angle_diff) > np.pi/8:
            cmd_vel.angular.z = 0.3 * np.sign(angle_diff)
        else:
            cmd_vel.linear.x = 0.1
            cmd_vel.angular.z = 0.2 * angle_diff
        
        self.cmd_vel_pub.publish(cmd_vel)


    def get_map_coordinates(self):
        """Convert current position to map coordinates"""
        if self.x is None or self.y is None:
            return None, None
        x = int((self.x - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
        y = self.map_msg.info.height - int((self.y - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution) - 1
        return x, y



    def detect_frontiers(self):
        """
        Detect frontier cells between explored free space and unexplored areas
        Returns list of frontier cells as (x, y) tuples in map coordinates
        """
        frontiers = []
        
        # Find all free space cells adjacent to unexplored areas
        for y in range(1, self.h-1):
            for x in range(1, self.w-1):
                if self.map[y, x] == FREE_SPACE_VALUE:
                    # Check neighbors for unexplored space
                    if (UNEXPLORED_SPACE_VALUE in [self.map[y-1, x], self.map[y+1, x], 
                                                  self.map[y, x-1], self.map[y, x+1]]):
                        frontiers.append((x, y))
        
        return frontiers


    def normalize_angle(self, angle):
        """ Normalize angle to [-π, π] range """
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle




def main():
    rclpy.init()

    node = Agent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
