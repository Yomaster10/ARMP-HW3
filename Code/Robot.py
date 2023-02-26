import itertools
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size
from shapely.geometry import Point, LineString

class Robot(object):
    
    def __init__(self):

        # define robot properties
        self.links = np.array([80.0,70.0,40.0,40.0])
        self.dim = len(self.links)

        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi/3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        '''
        Compute the Euclidean distance between two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        '''
        # TODO: Task 2.2 - DONE
        diff_vec = np.subtract(next_config, prev_config)
        return np.linalg.norm(diff_vec,2)

    def compute_forward_kinematics(self, given_config):
        '''
        Compute the 2D position (x,y) of each one of the links (including end-effector) and return.
        @param given_config Given configuration.
        '''
        # TODO: Task 2.2 - DONE
        link_base = [0,0]
        pos_vec = []
        prev_angle = 0
        for l in range(len(given_config)):
            new_angle = self.compute_link_angle(prev_angle, given_config[l])
            x = self.links[l]*np.cos(new_angle) + link_base[0]
            y = self.links[l]*np.sin(new_angle) + link_base[1]
            link_base = [x,y]
            pos_vec.append(link_base)
            prev_angle = new_angle
        return np.array(pos_vec)

    def compute_ee_angle(self, given_config):
        '''
        Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
        @param given_config Given configuration.
        '''
        ee_angle = given_config[0]
        for i in range(1,len(given_config)):
            ee_angle = self.compute_link_angle(ee_angle, given_config[i])

        return ee_angle

    def compute_link_angle(self, link_angle, given_angle):
        '''
        Compute the 1D orientation of a link given the previous link and the current joint angle.
        @param link_angle previous link angle.
        @param given_angle Given joint angle.
        '''
        if link_angle + given_angle > np.pi:
            return link_angle + given_angle - 2*np.pi
        elif link_angle + given_angle < -np.pi:
            return link_angle + given_angle + 2*np.pi
        else:
            return link_angle + given_angle
        
    def validate_robot(self, robot_positions):
        '''
        Verify that the given set of links positions does not contain self collisions.
        @param robot_positions Given links positions.
        '''
        # TODO: Task 2.2 - DONE
        arm = LineString(robot_positions)
        return arm.is_simple