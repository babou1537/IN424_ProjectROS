__author__ = "Johvany Gustave, Jonatan Alvarez"
__copyright__ = "Copyright 2025, IN424, IPSA 2025"
__credits__ = ["Johvany Gustave", "Jonatan Alvarez"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

import numpy as np
from .my_common import *    #common variables are stored here



class MapManager(Node):
    """ This class is used to merge maps from all the agents and display it on RVIZ and for the agents """
    def __init__(self):
        Node.__init__(self, "map_manager")

        self.load_params()
        self.init_map()

        self.map_agents_pub = self.create_publisher(OccupancyGrid, "/merged_map", 1)
        self.map_rviz_pub = self.create_publisher(OccupancyGrid, "/map", 1)

        for i in range(1, self.nb_agents+1):   #subscribe to agents' map topic
            self.create_subscription(OccupancyGrid, f"/bot_{i}/map", self.agent_map_cb, 1)
        
        self.create_timer(1, self.publish_maps)
    

    def load_params(self):
        """ Load parameters from launch file """
        #Get parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("nb_agents", rclpy.Parameter.Type.INTEGER),
                ("robot_size", rclpy.Parameter.Type.DOUBLE),
                ("env_size", rclpy.Parameter.Type.INTEGER_ARRAY)
            ]
        )
        self.nb_agents = self.get_parameter("nb_agents").value
        self.robot_size = self.get_parameter("robot_size").value
        self.env_size = self.get_parameter("env_size").value
    

    def init_map(self):
        """ Initialize maps to publish """
        self.map_agents_msg = OccupancyGrid()
        self.map_agents_msg.header.frame_id = "map" #set in which reference frame the map will be expressed (DO NOT TOUCH)
        self.map_agents_msg.header.stamp = self.get_clock().now().to_msg()  #get the current ROS time to send the msg
        self.map_agents_msg.info.resolution = self.robot_size   #Map cell size corresponds to robot size
        self.map_agents_msg.info.height = int(self.env_size[0]/self.map_agents_msg.info.resolution) #nb of rows
        self.map_agents_msg.info.width = int(self.env_size[1]/self.map_agents_msg.info.resolution)  #nb of columns
        self.map_agents_msg.info.origin.position.x = -self.env_size[1]/2    #x and y coordinates of the origin in map reference frame
        self.map_agents_msg.info.origin.position.y = -self.env_size[0]/2
        self.map_agents_msg.info.origin.orientation.w = 1.0 #to have a consistent orientation in quaternion: x=0, y=0, z=0, w=1 for no rotation
        self.merged_map = np.ones(shape=(self.map_agents_msg.info.height, self.map_agents_msg.info.width), dtype=np.int8)*UNEXPLORED_SPACE_VALUE    #all the cells are unexplored initially
        self.w, self.h = self.map_agents_msg.info.width, self.map_agents_msg.info.height
        #Same for RVIZ map
        self.map_rviz_msg = OccupancyGrid()
        self.map_rviz_msg.header = self.map_agents_msg.header
        self.map_rviz_msg.info = self.map_agents_msg.info

    
    def agent_map_cb(self, msg):
        """ 
            @brief Get new maps from agent and merge them.
            This method is automatically called whenever a new message is published on one of the following topics:
                /bot_1/map
                /bot_2/map
                /bot_3/map
            @param msg This is a nav_msgs/msg/OccupancyGrid message
        """
        received_map = np.flipud(np.array(msg.data).reshape(self.h, self.w))    #convert the received list into a 2D array and reverse rows
        for i in range(self.h):
            for j in range(self.w):
                if (received_map[i, j] != UNEXPLORED_SPACE_VALUE) and ((self.merged_map[i, j] == UNEXPLORED_SPACE_VALUE) or (self.merged_map[i, j] == FREE_SPACE_VALUE)):
                    self.merged_map[i, j] = received_map[i, j]

    def evaluate_frontier(self, frontier):
        """
        Évalue une frontière en fonction de plusieurs critères :
        - Taille de la frontière
        - Gain d'information
        - Accessibilité
        """
        i, j = frontier

        # Calcul de la taille de la frontière
        size = self.calculate_frontier_size(i, j)

        # Gain d'information : plus une frontière est proche d'une zone non explorée, plus elle est intéressante
        info_gain = self.calculate_information_gain(i, j)

        # Accessibilité : plus une frontière est proche d'un agent ou moins entourée d'obstacles, plus elle est accessible
        accessibility = self.calculate_accessibility(i, j)

        # Pondération des critères (ajustez les poids en fonction des priorités)
        score = (size * 0.3) + (info_gain * 0.4) + (accessibility * 0.3)

        return score

    def calculate_frontier_size(self, i, j):
        """
        Calcule la taille de la frontière en comptant le nombre de cellules adjacentes marquées comme FREE_SPACE.
        """
        size = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Haut, Bas, Gauche, Droite
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.map.shape[0] and 0 <= nj < self.map.shape[1]:
                if self.map[ni, nj] == FREE_SPACE_VALUE:
                    size += 1
        return size

    def calculate_information_gain(self, i, j):
        """
        Calcule le gain d'information en fonction de la proximité d'une zone UNEXPLORED_SPACE.
        Plus la zone adjacente est grande, plus le gain d'information est élevé.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        info_gain = 0
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.map.shape[0] and 0 <= nj < self.map.shape[1]:
                if self.map[ni, nj] == UNEXPLORED_SPACE_VALUE:
                    info_gain += 1
        return info_gain

    def calculate_accessibility(self, i, j):

        if not hasattr(self, 'agents_pose') or not self.agents_pose:
            return 0  # Si on n'a pas d'info sur les agents, accessibilité nulle

        min_distance = float('inf')

        for agent_x, agent_y in self.agents_pose:
                if agent_x is not None and agent_y is not None:
                    distance = np.sqrt((i - agent_x)**2 + (j - agent_y)**2)
                    min_distance = min(min_distance, distance)

            # Normalisation de la distance (plus proche = plus accessible)
        max_dist = max(self.map.shape)  # Distance max possible
        accessibility = 1 - (min_distance / max_dist)

            # Vérification des obstacles autour
        obstacle_penalty = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.map.shape[0] and 0 <= nj < self.map.shape[1]:
                    if self.map[ni, nj] == OBSTACLE_VALUE:
                        obstacle_penalty += 0.2  # Chaque obstacle réduit l'accessibilité

        accessibility = max(0, accessibility - obstacle_penalty)

        return accessibility

    def allocate_frontiers(self):
        """
        Alloue les frontières aux agents en fonction des scores calculés par evaluate_frontier.
        """
        frontiers_with_scores = []
        # Exemple de boucle pour détecter les frontières
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                """
                if self.map[i, j] == FRONTIER_VALUE:
                """
                if self.map[i, j] == FREE_SPACE_VALUE and any(
                0 <= i+di < self.map.shape[0] and 0 <= j+dj < self.map.shape[1] and 
                self.map[i+di, j+dj] == UNEXPLORED_SPACE_VALUE
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]):

                    score = self.evaluate_frontier((i, j))
                    frontiers_with_scores.append((score, (i, j)))

        # Trier les frontières par score décroissant
        frontiers_with_scores.sort(reverse=True, key=lambda x: x[0])

        # Attribuer les meilleures frontières aux agents
        for idx, (score, frontier) in enumerate(frontiers_with_scores[:self.nb_agents]):
            agent_idx = idx % self.nb_agents  # Assigner les frontières de manière circulaire aux agents
            self.assign_frontier_to_agent(agent_idx, frontier)

    def assign_frontier_to_agent(self, agent_idx, frontier):
        """
        Assigne une frontière à un agent donné. 
        Ajoutez ici la logique de déplacement de l'agent vers la frontière.
        """
        agent_pos = self.agents_pose[agent_idx]
        if agent_pos is not None:
            agent_x, agent_y = agent_pos
            # Vous pouvez ici implémenter la logique pour envoyer l'agent à la frontière
            self.get_logger().info(f"Agent {agent_idx + 1} assigned to frontier at ({frontier[0]}, {frontier[1]})")    

    def publish_maps(self):
        """ Publish maps on corresponding topics """
        self.map_rviz = self.merged_map.copy()
        #TODO: add frontiers on rviz map

        self.map_agents_msg.data = np.flipud(self.merged_map).flatten().tolist()    #transform the 2D array into a list to publish it
        self.map_rviz_msg.data = np.flipud(self.map_rviz).flatten().tolist()    #transform the 2D array into a list to publish it

        self.map_agents_pub.publish(self.map_agents_msg)    #publish the merged map to other agents on topic /merged_map
        self.map_rviz_pub.publish(self.map_rviz_msg)    #publish the merged map to RVIZ2 on topic /map



def main():
    rclpy.init()

    node = MapManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()