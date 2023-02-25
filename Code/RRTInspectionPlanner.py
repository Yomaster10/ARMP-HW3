import numpy as np
from RRTTree import RRTTree
import time

class RRTInspectionPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob, coverage):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env, task="ip")

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage

        ## custom addition
        # set step size for extensions
        self.step_size = planning_env.step_size

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan = []

        # TODO: Task 2.4

        # your stopping condition should look like this: 
        # while self.tree.max_coverage < self.coverage:

        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time()-start_time))

        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 2.4 - DONE
        
        return self.tree.get_vertex_for_config(plan[-1]).cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # TODO: Task 2.4 - DONE

        goal = False
        goal_config = self.planning_env.goal
        if np.allclose(rand_config, goal_config):
            goal = True

        vec = np.subtract(rand_config, near_config) 
        vec_mag = np.linalg.norm(vec,2)
        unit_vec = vec / vec_mag

        new_vec = self.step_size * unit_vec
        new_config = near_config + new_vec

        # check if this intersects the goal or not
        goal_added = False
        if goal and vec_mag < self.step_size:
            new_config = goal_config
            goal_added = True

        return new_config, goal_added