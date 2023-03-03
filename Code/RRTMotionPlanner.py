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
        self.step_size = planning_env.step_size

    def plan(self, stats_mode=False):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan = []

        # TODO: Task 2.3 - DONE

        # Initialize the tree
        env = self.planning_env
        self.tree.add_vertex(env.start)
        
        goal_added = False; num_iter = 0
        while not goal_added:
            num_iter += 1
            goal = False

            # Sampling step
            p = np.random.uniform() # goal biasing
            if p < self.goal_prob:
                config = env.goal
                goal = True
            else:
                config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
            
            # Verify that the sample is in free space
            if not env.config_validity_checker(config):
                continue
            
            # Get nearest vertex to the sample
            nearest_vert = self.tree.get_nearest_config(config)
            nearest_vert_idx = nearest_vert[0]

            # Partial extensions, if enabled
            if self.ext_mode == 'E2':
                config, goal_added = self.extend(nearest_vert[1], config) # config = x_new
                if not env.config_validity_checker(config):
                    continue
            
            # Check obstacle-collision for potential edge
            if env.edge_validity_checker(config, nearest_vert[1]):
                config_idx = self.tree.add_vertex(config, nearest_vert)
                cost = env.robot.compute_distance(config, nearest_vert[1])
                self.tree.add_edge(nearest_vert_idx, config_idx, cost)
                if goal and self.ext_mode == 'E1':
                    goal_added = True
            else:
                goal_added = False

        # Record the plan
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

        total_cost = self.compute_cost(plan)
        duration = time.time()-start_time

        if stats_mode:
            return np.array(plan), [total_cost,num_iter,duration]

        # Print total number of iterations
        print(f"Total number of iterations needed to reach goal: {num_iter}")
    
        # Print total path cost and time
        print('Total cost of path: {:.2f}'.format(total_cost))
        print('Total time: {:.2f}'.format(duration))
            
        return np.array(plan), None

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 2.3 - DONE
        
        return self.tree.get_vertex_for_config(plan[-1]).cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # TODO: Task 2.3 - DONE

        goal = False
        goal_config = self.planning_env.goal
        if np.allclose(rand_config, goal_config):
            goal = True

        vec = np.subtract(rand_config, near_config) 
        vec_mag = np.linalg.norm(vec,2)
        unit_vec = vec / vec_mag

        # Projection of the step size in the direction of the neighbor
        new_vec = self.step_size * unit_vec
        new_config = near_config + new_vec # New sample point

        # Check if this overshoots the goal, if yes return the goal
        goal_added = False
        if goal and vec_mag < self.step_size:
            new_config = goal_config
            goal_added = True # Tells the motion planner to terminate

        return new_config, goal_added