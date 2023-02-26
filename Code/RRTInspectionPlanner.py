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

        ## custom additions
        self.step_size = planning_env.step_size # for extensions
        self.decay = 10**-5 # for bias parameter decay
        self.adaptive = True # if True, then bias parameter decay is enabled

        self.all_insp_pts = self.planning_env.inspection_points
        self.points_inspected = {} # keeps track of all points seen so far
        self.best_idx = None
        #self.best_vert = None
        self.best_pts = None
        self.best_config = None
        #self.best_loc = None

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan = []

        # TODO: Task 2.4

        orig_goal_prob = self.goal_prob
        env = self.planning_env
        start_pts = list(env.get_inspected_points(env.start))
        self.tree.add_vertex(env.start, inspected_points=start_pts)
        
        method1 = True; method2 = False #; method3 = False
        num_iter = 0; reset_counter = 0
        while self.tree.max_coverage < self.coverage:
            rewire_successful = False
            num_iter += 1
            #if num_iter % 10000 == 0:
            #    print(self.points_inspected)
            #    print(num_iter, time.time()-start_time)
            #    env.visualize_map(config=best_config, show_map=True, tree_edges=self.get_edges_as_locs(), 
            #                      best_configs=self.get_configs_along_best_path(best_config), best_inspected_pts=best_pts)
 
            current_coverage = self.tree.max_coverage
            #if current_coverage > 1:
            #    if method1:
            #        print("Method Change")
            #    method3 = True
            #    method1 = False

            self.update_best()
            
            # Instead of biasing towards a goal vertex, we will sample configurations in an intelligent way (exploitation)
            p = np.random.uniform()
            if p < self.goal_prob: # Exploitation
                # In this biasing method, we sample configs until one is found which sees a point not yet seen by the best path
                if method1:
                    new_cov = -1
                    while new_cov < self.tree.max_coverage: # bias towards configs with better coverage opportunities
                        config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
                        pts = env.get_inspected_points(config)
                        new_pts = env.compute_union_of_points(pts, self.best_pts)
                        new_cov = env.compute_coverage(inspected_points=new_pts)

                # In this biasing method, we will create a potential new config by randomly perturbing
                # one of the joint values in the best config
                elif method2:
                    joint_to_perturb = np.random.randint(4) # randomly pick one joint to perturb
                
                    curr_val = self.best_config[joint_to_perturb] # current value of that joint
                    perturbation = np.random.uniform(low=-np.pi, high=np.pi) # randomly initialize a perturbation

                    config = self.best_config # initialize the new config to be the best config, then perturb it
                    if curr_val + perturbation < np.pi:
                        config[joint_to_perturb] += perturbation
                    elif curr_val - perturbation > -np.pi:
                        config[joint_to_perturb] -= perturbation
                    else:
                        config[joint_to_perturb] = np.random.uniform(low=-np.pi, high=np.pi)

                # If we go 1,000 iterations without improving the max. coverage, then we should attempt to rewire the tree
                if reset_counter > 1000:
                    reset_counter = 0
                    rewire_successful = self.rewire() # True if the rewire succeeded
                    #if rewire_successful:
                    #    config = pot_config

            else: # Exploration
                config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
                if self.adaptive:
                    self.goal_prob = min(orig_goal_prob + 1 - np.exp(-self.decay*num_iter), 1 - orig_goal_prob) # add a decay that will decrease exploration over time
            
            if not rewire_successful:
                # Is the sample in the free space?
                if not env.config_validity_checker(config):
                    continue
                
                nearest_vert = self.tree.get_nearest_config(config)
                nearest_vert_idx = nearest_vert[0]

                # Partial extensions, if enabled
                if self.ext_mode == 'E2':
                    config = self.extend(nearest_vert[1], config) # config = x_new
                    if not env.config_validity_checker(config):
                        continue
                
                # Does the edge between the sample and its nearest tree node collide with any obstacles?
                if env.edge_validity_checker(config, nearest_vert[1]):
                    insp_pts_prev = env.get_inspected_points(nearest_vert[1])
                    insp_pts_new = env.get_inspected_points(config)

                    insp_pts_newer = env.compute_union_of_points(insp_pts_prev, insp_pts_new) # take the union of the inspected points of the parent and child vertex
                    config_idx = self.tree.add_vertex(config, inspected_points=insp_pts_newer)

                    # when a new vertex is added to the tree, check if its inspected points are already recorded in the dictionary
                    pts_to_check = [key for key in self.points_inspected]
                    union = env.compute_union_of_points(pts_to_check, insp_pts_new)
                    if len(union) > len(self.points_inspected):
                        for p in union:
                            if tuple(p) not in pts_to_check:
                                self.points_inspected[tuple(p)] = [config]
                            # Even if we've already seen this missing point before, maybe we should still add new configs to the list
                            #else:
                            #    self.points_inspected[tuple(p)].append(config)
                    
                    cost = env.robot.compute_distance(config, nearest_vert[1])
                    self.tree.add_edge(nearest_vert_idx, config_idx, cost)
                    
            self.update_best()
            if current_coverage < self.tree.max_coverage:
                reset_counter = 0
                print(f"New Coverage: {self.tree.max_coverage:.03}, Iteration: {num_iter}, Bias: {self.goal_prob:.03}")
                env.visualize_map(config=self.best_config, show_map=True, tree_edges=self.get_edges_as_locs(), 
                                  best_configs=self.get_configs_along_best_path(self.best_config), best_inspected_pts=self.best_pts)
            else:
                reset_counter += 1

        plan.append(self.best_config)
        child_idx = self.best_idx
        parent_idx = self.tree.edges[child_idx] 
        parent_config = self.tree.vertices[parent_idx].config
        while self.tree.edges[child_idx]:
            plan.append(parent_config)
            child_idx = self.tree.get_idx_for_config(parent_config)
            # new parent
            parent_idx = self.tree.edges[child_idx] 
            parent_config = self.tree.vertices[parent_idx].config
        plan.append(parent_config)
        plan = plan[::-1]

        # Print total number of iterations, path cost, and time
        print(f"Total number of iterations needed to reach goal: {num_iter}")
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
        # TODO: Task 2.4
        vec = np.subtract(rand_config, near_config) 
        vec_mag = np.linalg.norm(vec,2)
        unit_vec = vec / vec_mag

        new_vec = self.step_size * unit_vec
        new_config = near_config + new_vec
        return new_config
    
    ### Custom additions

    def update_best(self):
        # when a new max. coverage vertex is found, we should update all of the attributes related to the best path
        self.best_idx = self.tree.max_coverage_id
        best_vert = self.tree.vertices[self.best_idx]
        self.best_pts = best_vert.inspected_points
        self.best_config = best_vert.config
        return

    def rewire(self):
        print()
        print("Rewiring Initiated...")

        # Step 1: Check which points have been seen in other paths but not in the current best path (if there are any)
        missing_pts = []
        for p in self.points_inspected:
            if list(p) not in self.best_pts:
                missing_pts.append(p)
        
        if len(missing_pts) == 0:
            print("\tNo missing points! Attempting to produce new ones...")
            for i in self.all_insp_pts:
                if tuple(i) not in list(self.points_inspected.keys()):
                    count = 0
                    not_found = True
                    while count < 1000 and not_found: # bias towards configs with better coverage opportunities
                        config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
                        pts = self.planning_env.get_inspected_points(config)
                        if i in pts:
                            missing_pts.append(tuple(i))
                            not_found = False
                            self.points_inspected[tuple(i)] = [config] # check that this line works as expected
                            # Todo: Check that this loop works as intended
                        count += 1
        if len(missing_pts) == 0:
            return False 

        print('\tPoints Seen: ', list(self.points_inspected.keys()))
        print('\tCurrent Best Points: ', [tuple(b) for b in self.best_pts])
        print('\tMissing Points: ', missing_pts)

        # Step 2: For each missing point, see if there exists a non-collisional path to it,
        # and if so calculate the minimum cost for doing so (among all missing points)
        new_config = None; best_dist = 1000
        for m in missing_pts:
            for c in self.points_inspected[m]: # keep going until we find a valid config known to see this point
                if np.allclose(c, self.best_config):
                    continue
                dist = self.planning_env.robot.compute_distance(c, self.best_config)
                if (dist < best_dist) and self.planning_env.edge_validity_checker(c, self.best_config):
                    best_dist = dist
                    new_config = c
    
        # Step 3: If a valid path was found to a missing point's config, then rewire the tree such that the current best path leads
        # to that config (i.e. the vertex corresponding to that config will be updated)
        if new_config is not None:
            print("It's Morbin' Time!")
            print("\tCoverage Pre-Morb: ", self.tree.max_coverage)
        
            insp_pts_new = self.planning_env.get_inspected_points(new_config) # check all points seen by the new config
            insp_pts_newer = self.planning_env.compute_union_of_points(self.best_pts, insp_pts_new) # take the union of the inspected points of the parent and child vertex

            old_idx = self.tree.get_idx_for_config(new_config) # get the index of the vertex in the tree that currently corresponds to that config
            cost = self.planning_env.robot.compute_distance(new_config, self.best_config)

            self.tree.vertices[old_idx].cost = self.tree.vertices[self.best_idx].cost + cost # update the cost of the path to the vertex
            self.tree.vertices[old_idx].inspected_points = insp_pts_newer # update the inspected points seen along the path to the vertex
            self.tree.edges[old_idx] = self.best_idx # update the edge between the current best config and the new config (old vertex)
            
            self.tree.max_coverage = self.planning_env.compute_coverage(inspected_points=insp_pts_newer) # update the max. coverage (this should always increase it)
            self.tree.max_coverage_id = old_idx
            self.update_best()

            print("\tCoverage Post-Morb: ", self.tree.max_coverage, "\n")
            return True
        return False

    # for visualization
    def get_edges_as_locs(self):
        '''
        Return the edges in the tree as a list of pairs of positions
        '''
        locs = []
        for (key, val) in self.tree.edges.items():
            first = self.planning_env.robot.compute_forward_kinematics(self.tree.vertices[val].config)[-1]
            second = self.planning_env.robot.compute_forward_kinematics(self.tree.vertices[key].config)[-1]
            locs.append([first,second])
        return locs
    
    # for visualization
    def get_configs_along_best_path(self, best_config):
        child_idx = self.tree.get_idx_for_config(best_config)
        if child_idx not in self.tree.edges:
            return [best_config]
        
        parent_idx = child_idx
        parent_config = best_config

        best_configs = []
        while self.tree.edges[child_idx]:
            best_configs.append(parent_config)
            child_idx = self.tree.get_idx_for_config(parent_config)
            parent_idx = self.tree.edges[child_idx] 
            parent_config = self.tree.vertices[parent_idx].config
        best_configs.append(parent_config)

        return best_configs
    
    """
                elif method3:
                    closest_pt = None
                    min_dist = 1000

                    unexplored = []
                    for p in self.all_insp_pts:
                        if p not in self.best_pts:
                            unexplored.append(p)
                            dist = np.linalg.norm(np.subtract(p, self.best_loc),2)
                            if dist < min_dist:
                                closest_pt = p
                                min_dist = dist

                    counter = 0; good = False
                    while True: # bias towards configs with better coverage
                        if closest_pt is None:
                            config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
                            break

                        if counter > 1000:
                            pert = np.pi/8
                        else:
                            pert = np.pi/4

                        config = []
                        for i in range (4):
                            ang = self.best_config[i]
                            samp = np.random.uniform(low=max(ang-pert,-np.pi), high=min(ang+pert,np.pi))
                            config.append(samp)
                        config = np.array(config)

                        if counter > 1000:
                            break

                        if not np.allclose(self.tree.get_nearest_config(config)[1], self.best_config):
                            continue

                        pts = env.get_inspected_points(config)
                        new_pts = env.compute_union_of_points(pts, self.best_pts)
                        #if (self.tree.get_nearest_config(config)[1] == best_config) and (len(new_pts) > len(best_pts)):
                        if len(new_pts) > len(self.best_pts): # the new config is closest to the best vertex and adds a new inspection point
                            break

                        ee_pos = env.robot.compute_forward_kinematics(config)[-1]
                        ee_angle = env.robot.compute_ee_angle(config) # angle of the end effector
                        ee_angle_range = np.array([ee_angle - env.robot.ee_fov/2, ee_angle + env.robot.ee_fov/2]) # FOV range of the end effector

                        for p in unexplored:
                            diff_vec = p - ee_pos
                            inspt_pt_angle = env.compute_angle_of_vector(diff_vec)
                            if env.check_if_angle_in_range(inspt_pt_angle, ee_angle_range): # the new config is closest to the best vertex and has a new inspection point in its view
                                good = True
                                break
                        if good:
                            break
                        #diff_vec = closest_pt - ee_pos
                        #inspt_pt_angle = env.compute_angle_of_vector(diff_vec)
                        #print(self.tree.get_nearest_config(config)[1], best_config)
                        #if env.check_if_angle_in_range(inspt_pt_angle, ee_angle_range): # the new config is closest to the best vertex and has a new inspection point in its view
                        #    break
                        counter += 1
    """