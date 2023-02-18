'''Mixture policy approach baseline implementation: https://core.ac.uk/download/pdf/212996663.pdf'''
import numpy as np
import argparse
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2
from linear_scalarization import scalarized_ql
import time

def run_Q_learning(episodes, alpha, epsilon, gamma, init_val, objective):
    """
    Q Learning to optimize ONE objective
    params: objective (int) index of the objective to maximize from
    """
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])
    Q_table = Q_table + init_val
    r_acc = np.zeros([len(env.loc_coords)])
    
    for i in range(1, episodes+1):
        state = env.reset()
        done = False
        
        while not done:
            if np.random.uniform(0, 1) < epsilon: action = env.action_space.sample()
            else: action = np.argmax(Q_table[state])
            
            next, reward, done = env.step(action)
            Q_table[state, action] = (1 - alpha)*Q_table[state, action] + alpha*(reward[objective]+gamma*np.max(Q_table[next]))
            state = next
            r_acc += reward
        print('Episode {}: {}'.format(i, r_acc))
    return Q_table

def greedy(vec, weights):
    '''Helper function'''
    arr = []
    for val in vec: arr.append(np.dot(weights, val))    # linear scalarization
    return np.argmax(arr)

def mixture(episodes, timesteps, alpha, epsilon, gamma, init_val, dimension, weights_arr, interval, run=1, save=True, table=None):
    '''Implementation of mixture policy algorithm'''
    dims = [i for i in range(len(weights_arr))]
    if type(table) == type(None):
        policies = []
        for dim in dims:    # Obtain set of policies
            q = np.full([env.observation_space.n, env.action_space.n, dimension], init_val, dtype=float)
            policies.append(q)
    else:
        policies = table
    
    nsw_data, total_data = [], []
    for i in range(1, episodes+1):
        R_acc = np.zeros(dimension)
        state = env.reset()
        done = False
        count, dim, c = 0, 0, 0
        Q = policies[dim]
        weights = weights_arr[dim]
        
        while not done:
            if count > int(timesteps/dimension/interval):   # determines the period of changing policies
                dim += 1
                if dim >= dimension: dim = 0  # back to first objective after a "cycle"
                Q = policies[dim]
                weights = weights_arr[dim]
                count = 0   # change policy after t/d timesteps
            if np.random.uniform(0, 1) < epsilon: action = env.action_space.sample()
            else: action = greedy(Q[state], weights)
            
            next, reward, done = env.step(action)
            count += 1
            next_action = greedy(Q[next], weights)
            for j in range(len(Q[state, action])):
                Q[state,action][j] = Q[state,action][j] + alpha*(reward[j]+gamma*Q[next,next_action][j]-Q[state,action][j])
            state = next
            R_acc += np.power(gamma, c)*reward
            c += 1
        
        R_acc = np.where(R_acc < 0, 0, R_acc) # Replace the negatives with 0
        nsw_score = np.power(np.product(R_acc), 1/len(R_acc))
        nsw_data.append(nsw_score)
        total_data.append(np.sum(R_acc))
        print('Episode {}\nAccumulated Discounted Reward: {}\nNSW: {}\n'.format(i, R_acc, nsw_score))
    if save == True:
        np.save('taxi_q_tables_V2/mixture_size{}_locs{}_run{}_nsw'.format(env.size, len(env.loc_coords), run), nsw_data)
        np.save('taxi_q_tables_V2/mixture_size{}_locs{}_run{}_total'.format(env.size, len(env.loc_coords), run), total_data)
    return nsw_data, total_data, np.mean(nsw_data), policies
    
def nsw(vec, nsw_lambda): 
    '''Helper function'''
    vec = vec + nsw_lambda
    vec = np.where(vec <= 0, nsw_lambda, vec)  # replace any negative values or zeroes with lambda
    return np.sum(np.log(vec))    # numpy uses natural log

def get_setting(size, num_locs):
    """
    To store environment settings

    Parameters
    ----------
    size : int
        size of the grid world in N x N
    num_locs : int
        number of location destination pairs
    """
    if num_locs == 2:
        loc_coords = [[0,0],[3,2]]
        dest_coords = [[0,4],[3,3]]
    elif num_locs == 3:
        loc_coords = [[0,0],[0,5],[3,2]]
        dest_coords = [[0,4],[5,0],[3,3]]
    elif num_locs == 4:
        loc_coords = [[0,0], [0,5], [3,2], [9,0]]
        dest_coords = [[0,4], [5,0], [3,3], [0,9]]
    elif num_locs == 5:
        loc_coords = [[0,0],[0,5],[3,2],[9,0],[4,7]]
        dest_coords = [[0,4],[5,0],[3,3],[0,9],[8,9]]
    else:
        loc_coords = [[0,0],[0,5],[3,2],[9,0],[8,9],[6,7]]
        dest_coords = [[0,4],[5,0],[3,3],[0,9],[4,7],[8,3]]
    return size, loc_coords, dest_coords

def grid_search(episodes, timesteps, alpha, epsilon, gamma, init_val, dimension, weights_arr, intervals): 
    # search for the optimal interval of switching policies
    arr = []
    for val in intervals:
        start_time = time.time()
        avg_nsw = mixture(episodes, timesteps, alpha, epsilon, gamma, init_val, dimension, weights_arr, val, save=False)
        print('Average NSW{}\n Time Taken: {}\n'.format(avg_nsw, start_time-time.time()))
        arr.append(avg_nsw)
    np.save('mixutre_best_interval', intervals[np.argmax(arr)])
    np.save('mixture_all_interval', arr)

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Mixture Policy on Taxi""")
    prs.add_argument("-f", dest="fuel", type=int, default=10000, required=False, help="Timesteps each trajectory\n")
    prs.add_argument("-ep", dest="episodes", type=int, default=5000, required=False, help="episodes.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.1, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.999, required=False, help="Discount rate\n")
    prs.add_argument("-i", dest="init_val", type=int, default=30, required=False, help="Initial values\n")
    prs.add_argument("-gs", dest="size", type=int, default=10, required=False, help="Grid size\n")
    prs.add_argument("-d", dest="dimension", type=int, default=2, required=False, help="Dimension of reward\n")
    args = prs.parse_args()
    
    size, loc_coords, dest_coords = get_setting(args.size, args.dimension)
    fuel = args.fuel
    env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel, output_path='')
    env.seed(1122)
    
    arr = np.arange(2,102,2)
    arr2 = np.arange(200,1100,100)
    arr_3 = np.concatenate((arr,arr2))
    grid_search(500, args.fuel, args.alpha, args.epsilon, 
                args.gamma, args.init_val, args.dimension, [[0.21, 0.79],[1.0, 0.0]], arr_3)