import argparse
from MapEnvironment import MapEnvironment
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner

# custom addition
import numpy as np
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

from TableCreator import update_table

def one_run(map, task, ext_mode, goal_prob, coverage, stats_mode=False):
    # prepare the map
    planning_env = MapEnvironment(json_file=map, task=task)

    # setup the planner
    if task == 'mp':
        planner = RRTMotionPlanner(planning_env=planning_env, ext_mode=ext_mode, goal_prob=goal_prob)
    elif task == 'ip':
        planner = RRTInspectionPlanner(planning_env=planning_env, ext_mode=ext_mode, goal_prob=goal_prob, coverage=coverage)
    else:
        raise ValueError('Unknown task option: %s' % task);
        
    # execute plan
    plan, results = planner.plan(stats_mode=stats_mode)

    return plan, results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='script for testing planners')
    parser.add_argument('-map', '--map', type=str, default='map_mp.json', help='Json file name containing all map information')
    parser.add_argument('-task', '--task', type=str, default='mp', help='choose from mp (motion planning) and ip (inspection planning)')
    parser.add_argument('-ext_mode', '--ext_mode', type=str, default='E1', help='edge extension mode')
    parser.add_argument('-goal_prob', '--goal_prob', type=float, default=0.05, help='probability to draw goal vertex')
    parser.add_argument('-coverage', '--coverage', type=float, default=0.5, help='percentage of points to inspect (inspection planning)')
    parser.add_argument('-stats', '--stats', type=bool, default=False, help='choose whether to enable statistics mode')
    parser.add_argument('-batch', '--batch', type=bool, default=False, help='choose whether to enable batch statistics mode')
    args = parser.parse_args()

    if not args.stats and not args.batch:
        # prepare the map
        planning_env = MapEnvironment(json_file=args.map, task=args.task)

        # setup the planner
        if args.task == 'mp':
            planner = RRTMotionPlanner(planning_env=planning_env, ext_mode=args.ext_mode, goal_prob=args.goal_prob)
        elif args.task == 'ip':
            planner = RRTInspectionPlanner(planning_env=planning_env, ext_mode=args.ext_mode, goal_prob=args.goal_prob, coverage=args.coverage)
        else:
            raise ValueError('Unknown task option: %s' % args.task);

        # execute plan
        plan, results = planner.plan()

        # Visualize the final path.
        planner.planning_env.visualize_plan(plan, ext_mode=args.ext_mode, goal_prob=args.goal_prob, coverage=args.coverage, cost=results[0])
    
    elif args.stats:
        print(f"Initiating statistics mode...")

        costs = []; iters = []; times = []
        for i in range(10):
            print(f"Trial: {i+1}/10")

            results = None
            while results is None:
                _, results = one_run(map=args.map, task=args.task, ext_mode=args.ext_mode, goal_prob=args.goal_prob, coverage=args.coverage, stats_mode=True)

            costs.append(results[0])
            iters.append(results[1])
            times.append(results[2])
            print(f"\tCost: {results[0]:.2f}, Iterations: {results[1]}, Time: {results[2]:.2f} [sec]")

        print("Data acquisition completed")
        avg_cost = np.mean(costs)
        avg_iter = np.mean(iters)
        avg_time = np.mean(times)
        print(f"\tAvg. Cost: {avg_cost:.2f}, Avg. Iterations: {avg_iter:.2f}, Avg. Time: {avg_time:.2f} [sec]")

        update_table(task=args.task, ext_mode=args.ext_mode, goal_prob=args.goal_prob, step_size=0.5, coverage=args.coverage, num_iter=avg_iter, time=avg_time, cost=avg_cost)
    
    else: # batch mode
        print(f"Initiating batch statistics mode...")

        if args.task == 'mp':
            batch = [['E1',0.05],['E2',0.05],['E1',0.2],['E2',0.2]]
            coverage = args.coverage
        else:
            batch = [['E1',0.5],['E2',0.5],['E1',0.75],['E2',0.75]]
            goal_prob = args.goal_prob

        for j in range(len(batch)):
            print(f"\nNew batch initiated with parameters: {batch[j][0],batch[j][1]}")
            costs = []; iters = []; times = []
            for i in range(10):
                print(f"\tTrial: {i+1}/10")

                ext_mode = batch[j][0]
                if args.task == 'mp':
                    goal_prob = batch[j][1]
                else:
                    coverage = batch[j][1]

                results = None
                while results is None:
                    _, results = one_run(map=args.map, task=args.task, ext_mode=ext_mode, goal_prob=goal_prob, coverage=coverage, stats_mode=True)

                costs.append(results[0])
                iters.append(results[1])
                times.append(results[2])
                print(f"\t\tCost: {results[0]:.2f}, Iterations: {results[1]}, Time: {results[2]:.2f} [sec]")

            print("\n\tData acquisition completed")
            avg_cost = np.mean(costs)
            avg_iter = np.mean(iters)
            avg_time = np.mean(times)
            print(f"\tAvg. Cost: {avg_cost:.2f}, Avg. Iterations: {avg_iter:.2f}, Avg. Time: {avg_time:.2f} [sec]")

            update_table(task=args.task, ext_mode=ext_mode, goal_prob=goal_prob, step_size=0.5, coverage=coverage, num_iter=avg_iter, time=avg_time, cost=avg_cost)