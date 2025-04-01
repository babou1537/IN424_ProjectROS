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
from collections import deque #Gestion de liste
import json
import heapq

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
        self.last_positions = deque(maxlen=5)

        self.nav_params = {
        'target_refresh_rate': 1.0,  # Fréquence de rafraîchissement des cibles (en secondes)
        'linear_speed': 2,           # Vitesse linéaire max (m/s)
        'angular_speed': 2,          # Vitesse angulaire max (rad/s)
        'arrival_threshold': 0.3,    # Distance pour considérer la cible atteinte (mètres)
        'progress_threshold': 0.5    # Distance minimale de progression (mètres)
        }

        self.frontier_weights = {
            'distance': 0.2,       # Poids de la distance
            'size': 0.6,          # Poids de la taille
            'accessibility': 0.35, # Poids de l'accessibilité
            'max_distance': 20,    # Distance max normalisée (en cellules)
            'depth_penalty': 0.8   # Coefficient de pénalité en profondeur
        }

        # Cartographie
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

        # Système de frontières (version optimisée)
        self.assigned_frontier = None  # Un seul attribut pour la cible actuelle
        self.shared_frontiers = {}  # { (x,y): robot_id } - remplace reserved_frontiers et known_frontiers
        self.target_lock_duration = 5.0  # Durée de réservation
        
        # Communication des frontières
        self.frontier_pub = self.create_publisher(Int32MultiArray, '/shared_frontiers', 10)
        self.id_pub = self.create_publisher(String, '/frontier_owners', 10)
        self.create_subscription(Int32MultiArray, '/shared_frontiers', self.frontiers_cb, 10)
        self.create_subscription(String, '/frontier_owners', self.owners_cb, 10)

        #Create timers to autonomously call the following methods periodically
        self.create_timer(0.3, self.map_update) #0.2s of period <=> 5 Hz
        self.create_timer(1.0, self.update_frontiers)  # ~3Hz
        self.create_timer(0.1, self.navigation_loop)
        self.create_timer(0.5, self.strategy)        #0.5s of period <=> 2 Hz
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
        """Decision layer unifié"""
        if not hasattr(self, 'map') or self.x is None:
            return
        
        # Si déjà une target, laisser update_frontiers gérer
        if self.assigned_frontier:
            return
        
        # Comportement par défaut
        cmd_vel = Twist()
        
        # Éviter les obstacles (à améliorer)
        if hasattr(self, 'lidar_data'):
            front_obstacle = any(0 < d < 0.5 for d in self.lidar_data.ranges[:30]+self.lidar_data.ranges[-30:])
            if front_obstacle:
                cmd_vel.angular.z = 0.7
            else:
                cmd_vel.linear.x = 0.2
        
        self.cmd_vel_pub.publish(cmd_vel)


    def world_to_map(self, world_x, world_y):
        """Conversion optimisée world coordinates -> map indices"""
        map_x = int((world_x - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
        map_y = self.map_msg.info.height - int((world_y - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution)
        return map_x, map_y

    def is_leader(self):
        """Détermine si cet agent est le leader"""
        return int(self.ns[-1]) == 1 and self.nb_agents > 1


    def evaluate_frontier(self, frontier):
        """
        Version améliorée du scoring qui:
        1. Favorise les frontières lointaines
        2. Pèse mieux la taille des zones inconnues
        3. Pénalise fortement les zones enclavées
        """
        i, j = frontier
        
        # 1. Distance (40% du score)
        agent_x, agent_y = self.world_to_map(self.x, self.y)
        distance = np.sqrt((i - agent_x)**2 + (j - agent_y)**2)
        norm_distance = min(distance / self.frontier_weights['max_distance'], 1.0)  # Normalisé [0-1] avec cap à 20 cellules
        
        # 2. Taille de la frontière (35% du score)
        frontier_size = 0
        unexplored_directions = []
        
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < self.h and 0 <= nj < self.w:
                if self.map[ni,nj] == UNEXPLORED_SPACE_VALUE:
                    frontier_size += 1
                    unexplored_directions.append((di,dj))
        
        # Bonus pour les grandes zones contiguës
        contiguous_bonus = 1.0 + 0.2 * len(unexplored_directions)
        norm_size = min(frontier_size / 4, 1.0) * contiguous_bonus  # Normalisé [0-1.2]

        # 3. Accessibilité (25% du score)
        obstacle_count = sum(
            1 for di, dj in [(-1,-1),(1,1),(-1,1),(1,-1)]
            if 0 <= i+di < self.h and 0 <= j+dj < self.w
            and self.map[i+di,j+dj] == OBSTACLE_VALUE
        )
        
        # Pénalité exponentielle pour les zones enclavées
        accessibility = max(0.1, 1.0 - 0.3**obstacle_count)  # Entre 0.1 et 1.0

        # 4. Facteur de profondeur (nouveau)
        depth_factor = 1.0
        for di, dj in unexplored_directions:
            for step in range(1, 5):  # Regarde 4 cases plus loin
                ni, nj = i + di*step, j + dj*step
                if 0 <= ni < self.h and 0 <= nj < self.w:
                    if self.map[ni,nj] == OBSTACLE_VALUE:
                        depth_factor *= self.frontier_weights['depth_penalty']
                        break

        # Score final pondéré
        score = (
            norm_distance * self.frontier_weights['distance'] +
            norm_size * self.frontier_weights['size'] + 
            accessibility * self.frontier_weights['accessibility']
        ) * depth_factor

        return score


    def detect_frontiers(self):
        """Détecte uniquement les cellules frontières externes (bord entre connu/inconnu)"""
        frontiers = []
        for i in range(1, self.h-1):
            for j in range(1, self.w-1):
                # Doit être espace libre ET avoir du vide adjacent
                if self.map[i,j] == FREE_SPACE_VALUE:
                    # Vérifie les 8 directions pour du vide
                    has_unknown = any(
                        self.map[i+di,j+dj] == UNEXPLORED_SPACE_VALUE
                        for di,dj in [(-1,-1), (-1,0), (-1,1), (0,-1), 
                                    (0,1), (1,-1), (1,0), (1,1)]
                        if 0 <= i+di < self.h and 0 <= j+dj < self.w
                    )
                    if has_unknown:
                        frontiers.append((i,j))
                        self.map[i, j] = FRONTIER_VALUE
        return frontiers


    def publish_frontiers(self, frontiers):
        """Publie les frontières et leur assignation"""
        # Message des positions
        pos_msg = Int32MultiArray()
        pos_msg.data = [coord for frontier in frontiers for coord in frontier]
        self.frontier_pub.publish(pos_msg)
        
        # Message des propriétaires (JSON)
        assign_msg = String()
        assign_msg.data = json.dumps({self.ns: frontiers})
        self.id_pub.publish(assign_msg)

    def frontiers_cb(self, msg):
        """Reçoit les positions des frontières"""
        frontiers = [(msg.data[i], msg.data[i+1]) for i in range(0, len(msg.data), 2)]
        self.known_frontiers = frontiers  # Mise à jour de la liste globale

    def owners_cb(self, msg):
        """Reçoit les assignations des frontières"""
        try:
            assignments = json.loads(msg.data)
            for robot_id, frontiers in assignments.items():
                for frontier in frontiers:
                    self.shared_frontiers[tuple(frontier)] = robot_id
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Erreur décodage JSON: {str(e)}")


    def allocate_frontiers(self, frontiers):
        """Version corrigée avec déclenchement de la navigation"""
        if not frontiers:
            return None

        available = [
            (self.evaluate_frontier(f), f) 
            for f in frontiers
            if not self.check_frontier_conflict(f)
        ]

        if not available:
            return None

        available.sort(reverse=True, key=lambda x: x[0])
        best_score, best_frontier = available[0]
        
        # Conversion en coordonnées monde
        world_x = best_frontier[0] * self.map_msg.info.resolution + self.map_msg.info.origin.position.x
        world_y = (self.map_msg.info.height - best_frontier[1]) * self.map_msg.info.resolution + self.map_msg.info.origin.position.y
        
        self.assigned_frontier = (world_x, world_y)  # Stocke en coordonnées monde
        self.publish_frontiers([best_frontier])
        
        # Déclenche immédiatement la navigation
        self.navigate_to_frontier()
        
        self.get_logger().info(
            f"Nouvelle cible: ({world_x:.2f}, {world_y:.2f}) | Score: {best_score:.2f}",
            throttle_duration_sec=1.0
        )
        return best_frontier
    
    def check_frontier_conflict(self, frontier):
        """Vérifie si une frontière est déjà prise"""
        return (tuple(frontier) in self.shared_frontiers and 
                self.shared_frontiers[tuple(frontier)] != self.ns)


    import heapq
import numpy as np
from collections import deque
from geometry_msgs.msg import Twist

class Agent:
    def a_star(self, start, goal):
        """ Implémente l'algorithme A* pour trouver le meilleur chemin """
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        open_list = []  # Min-heap de nœuds à explorer
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]  # Retourne le chemin trouvé
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Mouvement 4 directions
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < self.map_msg.info.width and
                    0 <= neighbor[1] < self.map_msg.info.height and
                    self.map[neighbor[1], neighbor[0]] != OBSTACLE_VALUE):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
        
        return []  # Retourne un chemin vide si aucun chemin n'est trouvé

    def navigate_to_frontier(self):
        """Utilisation de A* pour la navigation"""
        if not self.assigned_frontier or None in (self.x, self.y, self.yaw):
            return
        
        start = self.world_to_map(self.x, self.y)
        goal = self.assigned_frontier
        
        path = self.a_star(start, goal)
        if not path:
            self.get_logger().warn("Aucun chemin trouvé vers la frontière assignée !")
            return
        
        next_x, next_y = self.map_to_world(*path[0])
        dx = next_x - self.x
        dy = next_y - self.y
        distance = np.hypot(dx, dy)
        
        if distance < 1:
            self.get_logger().info("Cible atteinte!")
            self.assigned_frontier = None
            return
        
        target_yaw = np.arctan2(dy, dx)
        yaw_error = (target_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi

        cmd_vel = Twist()
        
        if abs(yaw_error) > 0.2:
            cmd_vel.angular.z = 0.5 * np.clip(yaw_error, -1, 1)
        else:
            cmd_vel.linear.x = 0.2 * min(1.0, distance)
        
        self.cmd_vel_pub.publish(cmd_vel)

    def update_frontiers(self):
        """Version optimisée avec gestion fluide des targets"""
        # Recalcule seulement si le timer est écoulé
        now = self.get_clock().now()
        if hasattr(self, 'last_frontier_update'):
            elapsed = (now - self.last_frontier_update).nanoseconds / 1e9
            if elapsed < self.nav_params['target_refresh_rate']:
                return
        
        self.last_frontier_update = now
        
        frontiers = self.detect_frontiers()
        if not frontiers:
            return
        
        # Change de target seulement si nécessaire
        if (self.assigned_frontier is None 
            or self.is_target_reached()
            or not self.is_making_progress()):
            
            self.allocate_frontiers(frontiers)

# **** NAVIGATION A* ****

class Agent:
    def a_star(self, start, goal):
        """ Implémente l'algorithme A* pour trouver le meilleur chemin """
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        open_list = []  # Min-heap de nœuds à explorer
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]  # Retourne le chemin trouvé
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Mouvement 4 directions
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < self.map_msg.info.width and
                    0 <= neighbor[1] < self.map_msg.info.height and
                    self.map[neighbor[1], neighbor[0]] != OBSTACLE_VALUE):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
        
        return []  # Retourne un chemin vide si aucun chemin n'est trouvé

    def navigate_to_frontier(self):
        """Utilisation de A* pour la navigation"""
        if not self.assigned_frontier or None in (self.x, self.y, self.yaw):
            return
        
        start = self.world_to_map(self.x, self.y)
        goal = self.assigned_frontier
        
        path = self.a_star(start, goal)
        if not path:
            self.get_logger().warn("Aucun chemin trouvé vers la frontière assignée !")
            return
        
        next_x, next_y = self.map_to_world(*path[0])
        dx = next_x - self.x
        dy = next_y - self.y
        distance = np.hypot(dx, dy)
        
        if distance < 1:
            self.get_logger().info("Cible atteinte!")
            self.assigned_frontier = None
            return
        
        target_yaw = np.arctan2(dy, dx)
        yaw_error = (target_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi

        cmd_vel = Twist()
        
        if abs(yaw_error) > 0.2:
            cmd_vel.angular.z = 0.5 * np.clip(yaw_error, -1, 1)
        else:
            cmd_vel.linear.x = 0.2 * min(1.0, distance)
        
        self.cmd_vel_pub.publish(cmd_vel)

def main():
    rclpy.init()

    node = Agent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()