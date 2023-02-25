import numpy as np
from RRTTree import RRTTree
import time

class RRTMotionPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

        ## custom addition
        # set step size for extensions
        #if planning_env.ylimit[1] < 100:
        #    self.step_size = 0.2
        #else:
        self.step_size = 10

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan = []

        # TODO: Task 2.3

        env = self.planning_env
        self.tree.add_vertex(env.start)
        
        goal_added = False; num_iter = 0
        while not goal_added:
            num_iter += 1
            goal = False

            # Here we add goal biasing
            p = np.random.uniform()
            if p < self.goal_prob:
                config = env.goal
                goal = True
            else:
                config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
            
            # Is the sample in the free space?
            if not env.config_validity_checker(config):
                continue
            
            nearest_vert = self.tree.get_nearest_config(config)
            #print(nearest_vert)
            nearest_vert_idx = nearest_vert[0]

            # Partial extensions, if enabled
            if self.ext_mode == 'E2':
                config, goal_added = self.extend(nearest_vert[1], config) # config = x_new
                if not env.config_validity_checker(config):
                    continue
            
            # Does the edge between the sample and its nearest tree node collide with any obstacles?
            if env.edge_validity_checker(config, nearest_vert[1]):
                config_idx = self.tree.add_vertex(config, nearest_vert)
                cost = env.robot.compute_distance(config, nearest_vert[1])
                self.tree.add_edge(nearest_vert_idx, config_idx, cost)
                if goal and self.ext_mode == 'E1':
                    goal_added = True
            else:
                goal_added = False

        plan.append(config)
        child_idx = config_idx
        parent_config = nearest_vert[1]
        while self.tree.edges[child_idx]:
            plan.append(parent_config)
            child_idx = self.tree.get_idx_for_config(parent_config)
            # new parent
            parent_idx = self.tree.edges[child_idx] 
            parent_config = self.tree.vertices[parent_idx].config
        plan.append(parent_config)
        plan = plan[::-1]

        print(f"Total number of iterations needed to reach goal: {num_iter}")
    
        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time()-start_time))

        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 2.3 - Check if it works
        
        return self.tree.get_vertex_for_config(plan[-1]).cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # TODO: Task 2.3
        goal = False
        goal_config = self.planning_env.goal

        #if (rand_config[0]==goal_config[0] and rand_config[1]==goal_config[1]):
        if np.allclose(rand_config, goal_config):
            goal = True

        #vec = [rand_config[i]-near_config[i] for i in range(2)]
        vec = np.subtract(rand_config, near_config) 

        #vec_mag = np.sqrt(sum(j**2 for j in vec))
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
    