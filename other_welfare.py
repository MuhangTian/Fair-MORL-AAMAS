import numpy as np
import json
import argparse
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2

def nonlinear_Q(episodes, alpha, epsilon, gamma, nsw_lambda, init_val, dim_factor, tolerance, file_name, mode, non_stationary, run, welfare_f, p=None):
    '''
    Welfare Q learning based on arbitrary welfare function (results in the supplmentary material)
    '''
    Q = np.zeros([fair_env.observation_space.n, fair_env.action_space.n, len(fair_env.loc_coords)], dtype=float)
    Num = np.full(fair_env.observation_space.n, epsilon, dtype=float)   # for epsilon
    Q = Q + init_val
    nsw_data, total_data, vec_data = [], [], []
            
    for i in range(1, episodes+1):
        R_acc = np.zeros(len(fair_env.loc_coords))
        state = fair_env.reset()
        print('Episode {}\nInitial State: {}'.format(i,fair_env.decode(state)))
        done = False
        avg = []
        c = 0
        
        while not done:
            epsilon = Num[state]
            avg.append(epsilon)
            
            if np.random.uniform(0,1) < epsilon:
                action = fair_env.action_space.sample()
            else:
                if non_stationary == True and welfare_f == 'egalitarian':
                    action = egalitarian(R_acc, R_acc+np.power(gamma,c)*Q[state])
                elif non_stationary == True and welfare_f == 'p-welfare':
                    action = p_welfare(R_acc+np.power(gamma,c)*Q[state], p)
                elif non_stationary == True and welfare_f == 'polynomial':
                    action = polynomial(R_acc+np.power(gamma,c)*Q[state], p)
                elif non_stationary == True and welfare_f == 'reciprocal':
                    action = reciprocal(R_acc+np.power(gamma,c)*Q[state], p)
                elif non_stationary == True and welfare_f == 'utilitarian':
                    action = utilitarian(R_acc+np.power(gamma,c)*Q[state])
                else: raise ValueError('Wrong mode')
                    
            next, reward, done = fair_env.step(action)
            
            if mode == 'myopic' and welfare_f == 'egalitarian':
                max_action = egalitarian(R_acc, gamma*Q[next])
            elif mode == 'myopic' and welfare_f == 'p-welfare':
                max_action = p_welfare(gamma*Q[next], p)
            elif mode == 'myopic' and welfare_f == 'polynomial':
                max_action = polynomial(gamma*Q[next], p)
            elif mode == 'myopic' and welfare_f == 'reciprocal':
                max_action = reciprocal(gamma*Q[next], p)
            elif mode == 'myopic' and welfare_f == 'utilitarian':
                max_action = utilitarian(gamma*Q[next])
            else: raise ValueError('Must have a mode')
            
            Q[state, action] = Q[state, action] + alpha*(reward + gamma*Q[next, max_action] - Q[state, action])
            
            Num[state] *= dim_factor  # epsilon diminish over time
            state = next
            R_acc += np.power(gamma,c)*reward
            c += 1
        
        R_acc = np.where(R_acc < 0, 0, R_acc) # Replace the negatives with 0
        nsw_score = np.power(np.product(R_acc), 1/len(R_acc))
        nsw_data.append(nsw_score)
        total = np.sum(R_acc)
        total_data.append(total)
        vec_data.append(R_acc)
        print('Accumulated reward: {}\nAverage Epsilon: {}\nNSW: {}\n'.format(R_acc,np.mean(avg),nsw_score))

    str = 'immd_' if mode == 'immediate' else ''
    if welfare_f == 'p-welfare': welfare_f = '{}-welfare'.format(p)
    if non_stationary == False:
        np.save(file='other_welfare_functions/{}_V2_size{}_locs{}_run{}_{}{}_no_gamma'.format(welfare_f, fair_env.size,len(fair_env.loc_coords), run, str, file_name),
                arr=Q)
        np.save(file='other_welfare_functions/{}_V2_size{}_locs{}_run{}_{}{}_no_gamma_vec'.format(welfare_f, fair_env.size,len(fair_env.loc_coords), run, str, file_name),
                arr=vec_data)
        np.save(file='other_welfare_functions/{}_V2_size{}_locs{}_run{}_{}{}_no_gamma_nsw'.format(welfare_f, fair_env.size,len(fair_env.loc_coords), run, str, file_name),
                arr=nsw_data)
        np.save(file='other_welfare_functions/{}_V2_size{}_locs{}_run{}_{}{}_no_gamma_total'.format(welfare_f, fair_env.size,len(fair_env.loc_coords), run, str, file_name),
                arr=total_data)
    else:
        np.save(file='other_welfare_functions/{}_V2_size{}_locs{}_run{}_{}{}'.format(welfare_f, fair_env.size,len(fair_env.loc_coords), run, str, file_name),
                arr=Q)
        np.save(file='other_welfare_functions/{}_V2_size{}_locs{}_run{}_{}{}_vec'.format(welfare_f, fair_env.size,len(fair_env.loc_coords), run, str, file_name),
                arr=vec_data)
        np.save(file='other_welfare_functions/{}_V2_size{}_locs{}_run{}_{}{}_nsw'.format(welfare_f, fair_env.size,len(fair_env.loc_coords), run, str, file_name),
                arr=nsw_data)
        np.save(file='other_welfare_functions/{}_V2_size{}_locs{}_run{}_{}{}_total'.format(welfare_f, fair_env.size,len(fair_env.loc_coords), run, str, file_name),
                arr=total_data)
    print('FINISH TRAINING')

def argmax_nsw(R, gamma_Q, nsw_lambda):
    '''Helper function'''
    sum = R + gamma_Q
    nsw_vals = [nsw(sum[i], nsw_lambda) for i in range(fair_env.action_space.n)]
    if np.all(nsw_vals == nsw_vals[0]) == True: # if all values are same, random action
        # numpy argmax always return first element when all elements are same
        action = fair_env.action_space.sample()
    else:
        action = np.argmax(nsw_vals)
    return action

def nsw(vec, nsw_lambda): 
    '''Helper function'''
    vec = vec + nsw_lambda
    vec = np.where(vec <= 0, nsw_lambda, vec)  # replace any negative values or zeroes with lambda
    return np.sum(np.log(vec))    # numpy uses natural log

def egalitarian(R_acc, vec):
    '''Helepr function for egalitarian welfare'''
    idx = np.argmin(R_acc)
    arr = []
    for val in vec: arr.append(val[idx])
    return np.argmax(arr)

def p_welfare(vec, p):
    '''Helper for Q welfare with input value of p'''
    arr = []
    for val in vec:
        sum = np.sum(np.power(val, p))
        arr.append(np.power(sum / len(val), 1/p))
    return np.argmax(arr)

def polynomial(vec, p):
    arr = []
    for val in vec: arr.append(np.sum(np.power(val, p)))
    return np.argmax(arr)

def reciprocal(vec, p):
    arr = []
    for val in vec: arr.append(-np.sum(np.power(1/val, p)))
    return np.argmax(arr)

def utilitarian(vec):
    '''Helepr for utilitarian welfare'''
    arr = []
    for val in vec: arr.append(np.sum(val))
    return np.argmax(arr)

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
    
if __name__ == '__main__':
    
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""NSW Q-learning on Taxi""")
    prs.add_argument("-f", dest="fuel", type=int, default=10000, required=False, help="Timesteps each episode\n")
    prs.add_argument("-ep", dest="episodes", type=int, default=5000, required=False, help="episodes.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-aN", dest="alpha_N", type=bool, default=False, required=False, help="Whether use 1/N for alpha\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.1, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.999, required=False, help="Discount rate\n")
    prs.add_argument("-nl", dest="nsw_lambda", type=float, default=1e-4, required=False, help="Smoothing factor\n")
    prs.add_argument("-i", dest="init_val", type=int, default=30, required=False, help="Initial values\n")
    prs.add_argument("-d", dest="dim_factor", type=float, default=0.9, required=False, help="Diminish factor for epsilon\n")
    prs.add_argument("-t", dest="tolerance", type=float, default=1e-5, required=False, help="Loss threshold for Q-values between each episode\n")
    prs.add_argument("-n", dest="file_name", type=str, default='', required=False, help="name of .npy\n")
    prs.add_argument("-mode", dest="mode", type=str, default='myopic', required=False, help="Action selection modes\n")
    prs.add_argument("-gs", dest="size", type=int, default=10, required=False, help="Grid size of the world\n")
    prs.add_argument("-num_locs", dest="num_locs", type=int, default=2, required=False, help="Number of Locations\n")
    prs.add_argument("-ns", dest="non_stat", type=bool, default=True, required=False, help="Whether non-stationary policy\n")
    prs.add_argument("-welf", dest="welfare_f", type=str, default='reciprocal', required=False, help="Which Welfare Function\n")
    prs.add_argument("-p", dest="p", type=float, default=1, required=False, help="p of p-welfare function\n")
    args = prs.parse_args()
    
    size, loc_coords, dest_coords = get_setting(args.size, args.num_locs)
    fair_env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, args.fuel, 
                            output_path='Taxi_MDP/NSW_Q_learning/run_', fps=4)
    fair_env.seed(1122)
    
    for i in range(1,51):   # Do 50 runs, and then take the average
        nonlinear_Q(episodes=args.episodes, alpha=args.alpha, epsilon=args.epsilon, mode=args.mode, gamma=args.gamma, 
                           nsw_lambda=args.nsw_lambda, init_val=args.init_val, non_stationary=args.non_stat, 
                           dim_factor=args.dim_factor, tolerance=args.tolerance, file_name=args.file_name, run=i, welfare_f=args.welfare_f, p=args.p)
    
    
    


    