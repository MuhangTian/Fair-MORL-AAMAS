'''
Linear Scalarization Method in MORL
Implemented according to Section 2.2.1 of https://jmlr.org/papers/volume15/vanmoffaert14a/vanmoffaert14a.pdf
'''
import numpy as np
import argparse
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2
from sympy.utilities.iterables import partitions
from more_itertools import distinct_permutations
from collections import Counter

def greedy(vec, weights):
    '''Helper function'''
    arr = []
    for val in vec: arr.append(np.dot(weights, val))    # linear scalarization
    return np.argmax(arr)

def nsw(vec, nsw_lambda): 
    '''Helper function'''
    vec = vec + nsw_lambda
    vec = np.where(vec <= 0, nsw_lambda, vec)  # replace any negative values or zeroes with lambda
    return np.sum(np.log(vec))    # numpy uses natural log

def scalarized_ql(env, init_val, a_space_n, s_space_n, dim, episodes, weights, alpha, gamma, epsilon, run, save=False):
    """
    Multi-objective Q learning with linear scalarization

    Parameters
    ----------
    env : object
        environment to run the algorithm
    init_val : float
        initial value of Q table
    a_space_n : int
        action space size
    s_space_n : int
        state space size
    dim : int
        dimension of rewards
    episodes : int
        number of episodes
    weights : array
        vector of weights for scalarization
    alpha : float
        learning rate
    gamma : float
        discount rate
    epsilon : float
        exploration rate
    save : bool, optional
        if True, save a .npy file containing nsw score over episodes in pareto directory,
        useful for plotting visualizations, by default False

    Returns
    -------
    array
        average accumulated reward calculated from a set number of episodes by user
    """
    if len(weights) != dim: raise ValueError('Dimension of weights not same as dimension of rewards')
    Q = np.full([s_space_n, a_space_n, dim], init_val, dtype=float)
    nsw_data, Racc, total_data = [], [], []   # for recording performance over time
    
    for i in range(1, episodes+1):
        R_acc = np.zeros(dim)   # for recording performance, does not affect action selection
        state = env.reset()
        done = False
        c = 0
        
        while not done:
            if np.random.uniform(0,1) < epsilon:    # epsilon greedy action selection
                action = env.action_space.sample()
            else:
                action = greedy(Q[state], weights)
                
            next, reward, done = env.step(action) # might also have info at left depending on environment
            next_action = greedy(Q[next], weights)
            for j in range(len(Q[state, action])):
                Q[state,action][j] = Q[state,action][j] + alpha*(reward[j]+gamma*Q[next,next_action][j]-Q[state,action][j])
            state = next
            R_acc += np.power(gamma, c)*reward
            c += 1
        
        R_acc = np.where(R_acc < 0, 0, R_acc) # Replace the negatives with 0
        nsw_score = np.power(np.product(R_acc), 1/len(R_acc))
        nsw_data.append(nsw_score)
        Racc.append(R_acc)
        total_data.append(np.sum(R_acc))
        print('Episode {}\nAccumulated Discounted Reward: {}\nNSW: {}\n'.format(i, R_acc, nsw_score))
    
    mean = np.mean(Racc, axis=0)
    print('Average Accumulated Discounted Reward: {}'.format(mean))
    if save==True: 
        np.save('taxi_q_tables_V2/scalarized_ql_nsw_size{}_locs{}_{}_nsw'.format(env.size, len(env.loc_coords), run), nsw_data)
        np.save('taxi_q_tables_V2/scalarized_ql_nsw_size{}_locs{}_{}_total'.format(env.size, len(env.loc_coords), run), total_data)
    return mean  

def pareto_dominated(pareto, current):
    """
    Check if average accumulated reward for a given weights already has weights that dominate it
    """
    for key in pareto:
        if np.greater(pareto[key], current).all():
            return True
        elif np.less_equal(pareto[key], current).all():
            continue
        elif np.greater_equal(pareto[key], current).all():
            return True
    return False

def remove_dominated(pareto, current):
    '''Remove old weights that are dominated (or same as) current weights'''
    for key in list(pareto):
        if np.greater_equal(current, pareto[key]).all():
            pareto.pop(key)

def grid_search(env, weights_arr, init_val, a_space_n, s_space_n, dim, episodes, alpha, gamma, epsilon, out_name):
    '''Grid search of a set of weights in weights_arr, return a dictionary of Pareto Optimal
    weights paired with its average accumulated reward'''
    all_points = []
    pareto_map = {}     # weights mapped by average accumulated reward, eventually a pareto front
    for weights in weights_arr:
        avg_R_acc = scalarized_ql(env, init_val, a_space_n, s_space_n, dim, episodes, weights, alpha, gamma, epsilon)
        all_points.append(avg_R_acc)
        if len(pareto_map) == 0: pareto_map[str(weights)] = avg_R_acc
        else:
            if pareto_dominated(pareto_map, avg_R_acc) == True:
                continue    # if dominated by other weights, no adding
            else:   # remove old weights dominated (or equal) by current weights (if any), then add
                remove_dominated(pareto_map, avg_R_acc)
                pareto_map[str(weights)] = avg_R_acc
    np.save('all_points_{}'.format(out_name), all_points)
    np.save('pareto_front_{}'.format(out_name), pareto_map)
    print('FINSIH GRID SEARCH')

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

def get_weights(dim, grid):
    '''Helper function'''
    for p in partitions(grid, m=dim):
        p[0] = dim - sum(p.values())
        yield from (distinct_permutations(x / grid for x in Counter(p).elements()))

def simple_grid_search(env, weights_arr, init_val, a_space_n, s_space_n, dim, episodes, alpha, gamma, epsilon, out_name):
    '''grid search to find the optimal weights for scalarization'''
    all_points, all_nsw = [], []
    for weights in weights_arr:
        env.seed(1122)
        avg_R_acc = scalarized_ql(env, init_val, a_space_n, s_space_n, dim, episodes, weights, alpha, gamma, epsilon, run=1)
        all_points.append(avg_R_acc)
        nsw = np.power(np.product(avg_R_acc), 1/len(avg_R_acc))
        all_nsw.append(nsw)
    best = np.argmax(all_nsw)
    best_weight = weights_arr[best]
    np.save('all_nsw_{}_dim{}'.format(out_name, dim), all_nsw)
    np.save('all_points_{}_dim{}'.format(out_name, dim), all_points)
    print('------------------FINSIH GRID SEARCH-------------------')
    print('The best weight is {}'.format(best_weight))

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Linear Scalarization""")
    prs.add_argument("-gs", dest="size", type=int, default=10, required=False, help="Grid size\n")
    prs.add_argument("-d", dest="dimension", type=int, default=2, required=False, help="Dimension of reward\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.999, required=False, help="Discount rate\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.1, required=False, help="Exploration rate.\n")
    prs.add_argument("-ep", dest="episodes", type=int, default=2500, required=False, help="Episodes.\n")
    prs.add_argument("-f", dest="fuel", type=int, default=10000, required=False, help="Timesteps each episode\n")
    prs.add_argument("-i", dest="interval", type=int, default=100, required=False, help="Interval of weights\n")
    args = prs.parse_args()
    
    size, loc_coords, dest_coords = get_setting(args.size, args.dimension)
    env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, args.fuel, '')
    env.seed(1122)
    weights_arr = list(get_weights(args.dimension, args.interval))
    # weights_arr = [0.37, 0.63]
    # for i in range(1, 51):
    #     scalarized_ql(env, 30, env.action_space.n, env.observation_space.n, args.dimension, 
    #                   args.episodes, weights_arr, args.alpha, args.gamma, args.epsilon, i, True)
    simple_grid_search(env, weights_arr, 30, env.action_space.n, env.observation_space.n,
                       args.dimension, args.episodes, args.alpha, args.gamma, args.epsilon, 'GRID_SEARCH')