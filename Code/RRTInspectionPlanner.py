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

        # set custom parameters
        self.step_size = planning_env.step_size # for extensions
        self.decay = 10**-3 # for bias parameter decay

        # record best vertex info
        self.best_idx = None
        self.best_pts = None
        self.best_config = None

    def plan(self, stats_mode=False):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()
        plan = [] # initialize an empty plan

        # TODO: Task 2.4
        orig_goal_prob = self.goal_prob
        env = self.planning_env

        start_pts = list(env.get_inspected_points(env.start))
        start_idx = self.tree.add_vertex(env.start, inspected_points=start_pts)
        self.points_inspected = {tuple(p):start_idx for p in start_pts} # keeps track of all points seen so far
        
        num_iter = 0; reset_counter = 0; thresh = 1000
        while self.tree.max_coverage < self.coverage:
            rewire_successful = False;num_iter += 1

            #if num_iter % 10000 == 0 and not stats_mode:
            #    print(f"\tIter: {num_iter}, Time: {time.time()-start_time:0.2f} [sec]")
            #    env.visualize_map(config=self.best_config, show_map=True, tree_edges=self.get_edges_as_locs(), 
            #        best_configs=self.get_configs_along_best_path(self.best_config), best_inspected_pts=self.best_pts)
 
            current_coverage = self.tree.max_coverage
            self.update_best()
            
            # Instead of biasing towards a goal vertex, we will sample configurations in an intelligent way (exploitation)
            p = np.random.uniform()
            if p < self.goal_prob: ## Exploitation
                # If we go a certain number of iterations without improving the max. coverage, then we should attempt to rewire the tree
                if (self.tree.max_coverage > 0.35):
                    thresh = 0
                if reset_counter > thresh:
                    reset_counter = 0
                    rewire_successful = self.rewire(stats_mode=stats_mode) # True if the rewire succeeded
                        
                # In this biasing method, we sample configs until one is found which sees a point not yet seen by the best path
                new_cov = -1; count = 0
                if not rewire_successful:
                    while (count < 150) and (new_cov < self.tree.max_coverage): # bias towards configs with better coverage opportunities
                        config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
                        count += 1
                        if not env.config_validity_checker(config): # ensure that the sampled configuration is valid
                            continue
                        pts = env.get_inspected_points(config)
                        new_pts = env.compute_union_of_points(pts, self.best_pts)
                        new_cov = env.compute_coverage(inspected_points=new_pts)
                    
            else: ## Exploration
                config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
                self.goal_prob = min(orig_goal_prob + 1 - np.exp(-self.decay*num_iter), 1 - orig_goal_prob) # add a decay that will decrease exploration over time
                if not env.config_validity_checker(config): # Is the sample in the free space?
                    continue
            
            if not rewire_successful:
                nearest_vert = self.tree.get_nearest_config(config)
                nearest_vert_idx = nearest_vert[0]

                # Partial extensions, if enabled
                if self.ext_mode == 'E2':
                    config = self.extend(nearest_vert[1], config)
                    if config is None:
                        continue
                
                # Does the edge between the sample and its nearest tree node collide with any obstacles?
                if env.edge_validity_checker(config, nearest_vert[1]):
                    pts_prev = self.tree.vertices[nearest_vert_idx].inspected_points
                    pts_new = env.get_inspected_points(config)
                    pts_newer = env.compute_union_of_points(pts_prev, pts_new) # take the union of the inspected points of the parent and child vertex
                    
                    v_new_idx = self.tree.add_vertex(config, inspected_points=pts_newer)

                    # when a new vertex is added to the tree, check if its inspected points are already recorded in the dictionary
                    pts_to_check = [key for key in self.points_inspected]
                    union = env.compute_union_of_points(pts_to_check, pts_new)
                    if len(union) > len(self.points_inspected):
                        for p in union:
                            if tuple(p) not in pts_to_check:
                                self.points_inspected[tuple(p)] = v_new_idx
                    
                    cost = env.robot.compute_distance(config, nearest_vert[1])
                    self.tree.add_edge(nearest_vert_idx, v_new_idx, cost)
                    
            self.update_best()
            if current_coverage < self.tree.max_coverage:
                reset_counter = 0
                if not stats_mode:
                    print(f"New Coverage: {self.tree.max_coverage:.03}, Iteration: {num_iter}, Bias: {self.goal_prob:.03}")
                    env.visualize_map(config=self.best_config, show_map=True, tree_edges=self.get_edges_as_locs(), 
                                  best_configs=self.get_configs_along_best_path(self.best_config), best_inspected_pts=self.best_pts)
                else:
                    if rewire_successful:
                        print(f"\tNew Coverage: {self.tree.max_coverage:.03}, Iteration: {num_iter}, Bias: {self.goal_prob:.03} - with Rewire")
                    else:   
                        print(f"\tNew Coverage: {self.tree.max_coverage:.03}, Iteration: {num_iter}, Bias: {self.goal_prob:.03}")
            else:
                reset_counter += 1

            if time.time()-start_time > 300:
                print("Time ran out...")
                return None, None
            
        self.update_best()

        plan.append(self.best_config)
        child_idx = self.best_idx
        parent_idx = self.tree.edges[child_idx] 
        parent_config = self.tree.vertices[parent_idx].config
  
        while self.tree.edges[child_idx]:
            plan.append(parent_config)
            child_idx = self.tree.get_idx_for_config(parent_config)
            parent_idx = self.tree.edges[child_idx] # new parent
            parent_config = self.tree.vertices[parent_idx].config
        plan.append(parent_config)
        plan = plan[::-1]

        total_cost = self.compute_cost(plan)
        duration = time.time()-start_time

        if stats_mode:
            return np.array(plan), [total_cost,num_iter,duration]

        # Print total number of iterations, path cost, and time
        print(f"Total number of iterations needed to reach goal: {num_iter}")
        print('Total cost of path: {:.2f}'.format(total_cost))
        print('Total time: {:.2f} [sec]'.format(duration))

        return np.array(plan), None

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

        if self.planning_env.config_validity_checker(new_config):
            return new_config
        else:
            return None
    
    ### Custom additions

    def update_best(self):
        # when a new max. coverage vertex is found, we should update all of the attributes related to the best path
        self.best_idx = self.tree.max_coverage_id
        best_vert = self.tree.vertices[self.best_idx]
        self.best_pts = [tuple(b) for b in best_vert.inspected_points]
        self.best_config = best_vert.config
        return

    def rewire(self, stats_mode):
        if not stats_mode:
            print()
            print("Rewiring Initiated...")

        # Step 1: Check which points have been seen in other paths but not in the current best path (if there are any)
        missing_pts = []
        for p in self.points_inspected:
            if p not in self.best_pts:
                missing_pts.append(p)
        if len(missing_pts) == 0:
            return False 

        # Step 2: For each missing point, record the vertex that sees it in a new list and sort this list
        # according to the minimum distance between each vertex and the best vertex in the tree
        v_curr = None; dists = []
        for m in missing_pts:
            v_p_idx = self.points_inspected[m]
            dist = self.planning_env.robot.compute_distance(self.tree.vertices[v_p_idx].config, self.best_config)
            dists.append({'Vertex':v_p_idx, 'Dist':dist})
        
        dists.sort(key=lambda x:x.get('Dist'))
        v_curr_idx = dists[0]['Vertex']
        v_curr = self.tree.vertices[v_curr_idx]
    
        # Step 3: If a valid path was found to a missing point's config, then rewire the tree such that the current best path leads
        # to that config (i.e. the vertex corresponding to that config will be updated)
        if v_curr is not None:
            if not stats_mode:
                print("It's Rewirin' Time!")
                print(f"\tCoverage Pre-Rewire: {self.tree.max_coverage:0.3f}")
            
            i = 0
            while (not self.planning_env.edge_validity_checker(v_curr.config, self.best_config)):                         
                i += 1
                if i > len(dists)-1:
                    return False
                
                v_curr_idx = dists[i]['Vertex'] # get the index of the vertex in the tree that currently corresponds to that config
                v_curr = self.tree.vertices[v_curr_idx]
            
            insp_pts_new = self.planning_env.get_inspected_points(v_curr.config) # check all points seen by the new config
            insp_pts_newer = self.planning_env.compute_union_of_points(self.best_pts, insp_pts_new) # take the union of the inspected points of the parent and child vertex

            cost = self.planning_env.robot.compute_distance(v_curr.config, self.best_config)
            old_cost = v_curr.cost # get the original cost of the rewired vertex

            self.tree.vertices[v_curr_idx].cost = self.tree.vertices[self.best_idx].cost + cost # update the cost of the path to the vertex
            self.tree.vertices[v_curr_idx].inspected_points = insp_pts_newer # update the inspected points seen along the path to the vertex
            self.tree.edges[v_curr_idx] = self.best_idx # update the edge between the current best config and the new config (old vertex)
            
            self.tree.max_coverage = self.planning_env.compute_coverage(inspected_points=insp_pts_newer) # update the max. coverage (this should always increase it)
            self.tree.max_coverage_id = v_curr_idx
            
            self.propagate(v_curr_idx, old_cost) # update the costs and inspected points of all descendents of the rewired vertex
            self.update_best()

            if not stats_mode:
                print(f"\tCoverage Post-Rewire: {self.tree.max_coverage:0.3f}\n")
            return True
        return False
    
    def child_update(self, child, parent, cost_0):
        # record the configuration of the child node
        config = self.tree.vertices[child].config
        cost_1 = self.tree.vertices[child].cost # original child cost

        # update the cost of the child node: (original child cost - original parent cost) + new parent cost
        self.tree.vertices[child].cost = (cost_1 - cost_0) + self.tree.vertices[parent].cost
                
        # record the points seen directly by this configuration
        original_pts = self.planning_env.get_inspected_points(config)

        # update the inspected points seen along the path to the child with the new parent
        self.tree.vertices[child].inspected_points = self.planning_env.compute_union_of_points(original_pts, self.tree.vertices[parent].inspected_points)
        
        # check to see if the coverage improves or is equal (to ensure we end up at a leaf node)
        new_coverage = self.planning_env.compute_coverage(inspected_points=self.tree.vertices[child].inspected_points)
        if new_coverage >= self.tree.max_coverage:
            self.tree.max_coverage = new_coverage
            self.tree.max_coverage_id = child

        return cost_1

    def propagate(self, parent, cost_0):
        """
        This function propagates the costs and inspected points of the descendants of a vertex which has just been rewired
        """
        if parent not in self.tree.edges.values():
            return
        children = []
        for k,v in self.tree.edges.items():
            if v == parent:
                children.append(k)
        for c in children:
            cost_1 = self.child_update(c, parent, cost_0)
            self.propagate(c, cost_1)
        return

    # for visualization
    def get_edges_as_locs(self):
        """
        Return the edges in the tree as a list of pairs of positions
        """
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
    # If missing_pts is still empty at this point, this means that all the inspected points we've seen so far are seen along the best path
        if len(missing_pts) == 0:
            if not stats_mode:
                print("\tNo missing points! Attempting to produce new ones...")
            for i in self.all_insp_pts:
                if tuple(i) not in list(self.points_inspected.keys()):
                    count = 0
                    not_found = True
                    while count < 10000 and not_found: # bias towards configs with better coverage opportunities
                        config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
                        if not self.planning_env.config_validity_checker(config) or not self.planning_env.edge_validity_checker(config, self.best_config): # ensure that the sampled configuration is valid
                            continue
                        pts = self.planning_env.get_inspected_points(config)
                        if i in pts:
                            missing_pts.append(tuple(i))
                            not_found = False
                            self.points_inspected[tuple(i)] = [config] # check that this line works as expected
                        #if not not_found:
                            new_idx = self.tree.add_vertex(config, inspected_points=pts) # fake vertex
                            self.tree.add_edge(new_idx, 0, edge_cost=0) # fake edge
                            # Todo: Check that this loop works as intended
                        count += 1
                    if not_found and not stats_mode:
                        print("\t\tBad Morb :(")
                else:
                    if tuple(i) not in self.best_pts: # This should never occur
                        print("uwu?")
                        print(self.points_inspected)
                        print(tuple(i))
                        print(list(self.best_pts))
            print()
    """
    """
            for c in self.points_inspected[m]: # keep going until we find a valid config known to see this point
                if np.allclose(c, self.best_config):
                    continue
                dist = self.planning_env.robot.compute_distance(c, self.best_config)
                if (dist < best_dist) and self.planning_env.edge_validity_checker(c, self.best_config):
                    best_dist = dist
                    new_config = c
            """
            #c = self.points_inspected[m]
            #dist = self.planning_env.robot.compute_distance(c, self.best_config)
            #dists.append({'Config':c,'Dist':dist})

            #if self.ext_mode == 'E2':
                #    new_config = self.extend(self.best_config, dists[i]['Config'])
                #    if new_config is None:
                #        continue
                #else:
                #new_config = dists[i]['Config']

                #if self.ext_mode == 'E2':
            #    if new_config is not None:
            #        return new_config, True
            #    else:
            #        return None, False