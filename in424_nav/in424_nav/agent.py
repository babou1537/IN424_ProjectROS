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


    """VERSION 2"""
    def map_update(self):
        """ Met à jour la carte de l'agent avec les données du LiDAR """
        if self.x is None or self.y is None or not hasattr(self, 'lidar_data'):
            return  # Attendre que l'agent ait une position définie et que les données LiDAR soient disponibles

        # Récupérer la position de l'agent en indices de carte
        agent_x = int((self.x - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
        agent_y = self.map_msg.info.height - int((self.y - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution) -1

        # Vérifier si la position de l'agent est dans les limites
        # if 0 <= agent_x < self.map_msg.info.width and 0 <= agent_y < self.map_msg.info.height:
        #     self.map[agent_y, agent_x] = PATH_VALUE  # Marquer la position du robot

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
        # Test

    

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
            """Decision and action layers with A* pathfinding with obstacle avoidance"""
            if self.x is None or self.y is None or not hasattr(self, 'map'):
                return
        
        # Trouver la frontière la plus proche
            frontier = self.find_nearest_frontier()
        
            if frontier is None:
                self.get_logger().info("Toute la carte a été explorée !")
                self.stop_robot()
                return 
        
            start = self.world_to_map(self.x, self.y)
            path = self.a_star_search(start, frontier)

        # Si pas de chemin trouvé ou robot coincé près d'un mur
            if path is None or len(path) <= 1 or self.is_near_obstacle():
                # Essayer de contourner l'obstacle
                if not self.avoid_obstacle():
                    # Si le contournement échoue, choisir une nouvelle frontière
                    self.mark_area_as_explored(frontier)
                return
            
            next_cell = path[1]
            next_pos = self.map_to_world(next_cell[0], next_cell[1])
            
            # Vérifier si on est proche de la cible actuelle
            distance_to_target = math.sqrt((self.x - next_pos[0])**2 + (self.y - next_pos[1])**2)
            if distance_to_target < self.robot_size * 2:
                self.mark_area_as_explored(frontier)
                return
                
            self.move_to_position(next_pos)

        def is_near_obstacle(self):
            """Vérifie si le robot est proche d'un obstacle/mur"""
            robot_x, robot_y = self.world_to_map(self.x, self.y)
            detection_radius = 2  # en cellules
            
            for dy in range(-detection_radius, detection_radius + 1):
                for dx in range(-detection_radius, detection_radius + 1):
                    nx, ny = robot_x + dx, robot_y + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h:
                        if self.map[ny, nx] == OBSTACLE_VALUE:
                            return True
            return False

        def avoid_obstacle(self):
            """Tente de contourner un obstacle"""
            # Trouver la direction libre la plus prometteuse
            robot_x, robot_y = self.world_to_map(self.x, self.y)
            directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]  # 8 directions
            
            # Classer les directions par distance aux frontières inexplorées
            scored_directions = []
            for dx, dy in directions:
                nx, ny = robot_x + dx, robot_y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    if self.map[ny, nx] == FREE_SPACE_VALUE:
                        # Calculer un score basé sur le nombre de cellules inexplorées à proximité
                        unexplored_score = self.count_unexplored_around(nx, ny)
                        scored_directions.append((unexplored_score, (dx, dy)))
            
            if not scored_directions:
                return False  # Aucune direction libre
            
            # Choisir la direction avec le meilleur score
            scored_directions.sort(reverse=True)
            best_dir = scored_directions[0][1]
            
            # Se déplacer dans cette direction
            target_x = robot_x + best_dir[0] * 3  # Aller un peu plus loin
            target_y = robot_y + best_dir[1] * 3
            target_pos = self.map_to_world(target_x, target_y)
            self.move_to_position(target_pos)
            return True

        def count_unexplored_around(self, x, y, radius=3):
            """Compte les cellules inexplorées autour d'une position"""
            count = 0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h:
                        if self.map[ny, nx] == UNEXPLORED_SPACE_VALUE:
                            count += 1
            return count




def main():
    rclpy.init()

    node = Agent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()