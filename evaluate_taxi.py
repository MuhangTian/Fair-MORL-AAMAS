'''Evaluation of the agent by playing the game using trained Q table, with different algorithms'''
import numpy as np
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2

def argmax_nsw(R, gamma_Q, nsw_lambda):
    '''Helper function'''
    sum = R + gamma_Q
    nsw_vals = [nsw(sum[i], nsw_lambda) for i in range(env.action_space.n)]
    if np.all(nsw_vals == nsw_vals[0]) == True: # if all values are same, random action
        # numpy argmax always return first element when all elements are same
        action = env.action_space.sample()
    else:
        action = np.argmax(nsw_vals)
    return action

def nsw(vec, nsw_lambda): 
    '''Helper function'''
    vec = vec + nsw_lambda
    vec = np.where(vec <= 0, nsw_lambda, vec)  # replace any negative values or zeroes with lambda
    return np.sum(np.log(vec))    # numpy uses natural log

def eval_nsw(Q: np.ndarray, name: str, mode: list, taxi_loc: list=None, pass_dest: list=None, 
             episodes: int=20, nsw_lambda: float=0.01, check_dest: bool=False, 
             render: bool=True, gamma: float=0.999, update: bool=False):
    """
    Evaluate the performance of welfare Q-learning based on trained Q table

    Parameters
    ----------
    Q : np.ndarray
        trained Q table
    name : str
        name of the file to be stored
    mode : list
        mode which controls whether to run stationary or non-stationary policy
    taxi_loc : list, optional
        initial location of taxi in the environment, by default None
    pass_dest : list, optional
        initial location of passengers in the environment, by default None
    episodes : int, optional
        number of episodes to run, by default 20
    nsw_lambda : float, optional
        smoothing factor, by default 0.01
    check_dest : bool, optional
        whether to check destination, by default False
    render : bool, optional
        whether to visualize the agent running in the grid world, by default True
    gamma : float, optional
        discount factor, by default 0.999
    update : bool, optional
        whether to continuing updating Q table during evaluation, that is
        whether to continue learning during evaluation, by default False
    """

    Racc = []
    if render == True: check_dest = False
    if check_dest == True:
        for i in range(env.size):
            for j in range(env.size):
                done = False
                R_acc = np.zeros(len(env.loc_coords))
                state = env.reset([i,j])
                c = 0
                print('Initial State: {}'.format(env.decode(state)))
                
                while not done:   
                    if mode[0] == 'non-stationary':               
                        action = argmax_nsw(R_acc, np.power(gamma, c)*Q[state], nsw_lambda)
                    elif mode[0] == 'stationary':
                        action = argmax_nsw(0, np.power(gamma,c)*Q[state], nsw_lambda)
                    next, reward, done = env.step(action)
                    if update == True:
                        if mode[1] == 'myopic':
                            max_action = argmax_nsw(0, gamma*Q[next], nsw_lambda)
                        elif mode[1] == 'immediate':
                            max_action = argmax_nsw(reward, gamma*Q[next], nsw_lambda)
                        Q[state, action] = Q[state, action] + 0.1*(reward + gamma*Q[next, max_action] - Q[state, action])
                    state = next
                    R_acc += np.power(gamma, c)*reward
                    c += 1

                nsw_score = nsw(R_acc, nsw_lambda)
                Racc.append(R_acc)
                print('Accumulated Reward: {}\nNSW: {}\n'.format(R_acc, nsw_score))
    else:
        for i in range(1, episodes+1):
            env._clean_metrics()
            done = False
            R_acc = np.zeros(len(env.loc_coords))
            pass_loc = None if pass_dest == None else 1
            state = env.reset(taxi_loc, pass_loc, pass_dest)
            c = 0
            if render == True: env.render()
            
            while not done:
                if mode[0] == 'non-stationary':               
                    action = argmax_nsw(R_acc, Q[state], nsw_lambda)
                elif mode[0] == 'stationary':
                    action = argmax_nsw(0, Q[state], nsw_lambda)
                next, reward, done = env.step(action)
                if update == True:
                    if mode[1] == 'myopic':
                        max_action = argmax_nsw(0, gamma*Q[next], nsw_lambda)
                    elif mode[1] == 'immediate':
                        max_action = argmax_nsw(reward, gamma*Q[next], nsw_lambda)
                    Q[state, action] = Q[state, action] + 0.1*(reward + gamma*Q[next, max_action] - Q[state, action])
                if render == True: env.render()
                state = next
                R_acc += np.power(gamma, c)*reward
                c += 1
            
            print('Accumulated Reward, episode {}: {}\n'.format(i, R_acc))
            Racc.append(R_acc)
        #env._output_csv()
        np.save('Experiments/{}.npy'.format(name), Racc)
    
    print("Average Accumulated Reward: {}\nFINSIH EVALUATE NSW Q LEARNING\n".format(np.mean(Racc, axis=0)))

def eval_ql(Q: np.ndarray, taxi_loc: list=None, pass_dest: list=None, 
            episodes: int=20, render: bool=False, gamma: float=0.999):
    """
    Evaluate Q-learning algorithm based on trained Q table

    Parameters
    ----------
    Q : np.ndarray
        trained Q table
    taxi_loc : list, optional
        initial location of taxi in the grid world, by default None
    pass_dest : list, optional
        initial locations of passengers in the grid world, by default None
    episodes : int, optional
        number of episodes to run, by default 20
    render : bool, optional
        whether to visualize the running environment while evaluating, by default False
    gamma : float, optional
        discount factor, by default 0.999
    """
    Racc = []
    for i in range(1, episodes+1):
        done = False
        R_acc = np.zeros(len(env.loc_coords))
        pass_loc = None if pass_dest == None else 1
        state = env.reset(taxi_loc, pass_loc, pass_dest)
        c = 0
        if render==True: env.render()
        while not done:
            action = np.argmax(Q[state])
            next, reward, done = env.step(action)
            if render==True: env.render()
            state = next
            R_acc += np.power(gamma, c)*reward
            c += 1
        
        Racc.append(R_acc)
        print('Episode {}: {}'.format(i, R_acc))
    # env._output_csv()
    print('Average Discounted Accumulated Reward :{}\n'.format(np.mean(Racc, axis=0)))
    return print("FINISH EVALUATE Q LEARNING\n")

def get_setting(size: int, num_locs: int):
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
    size, loc_coords, dest_coords = get_setting(10,4)
    fuel = 10000
    env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel, '', 15)
    env.seed(1122)  # make sure to use same seed as we used in learning
    update = True
    mode = ['stationary', 'myopic']
    # Q = np.load('Experiments/taxi_q_tables_V2/NSW_Penalty_V2_size10_locs2_stat_1.npy')

    # eval_nsw(Q, taxi_loc=[9,2], nsw_lambda=1e-4, mode=mode, gamma=0.999,
    #         episodes=50, render=False,  check_dest=True, name='', update=update)
    # check_all_locs(Q, eval_steps=4000, gamma=0.999, nsw_lambda=1e-4, nsw=True, thres=50, update=update, mode=mode)
    Q = np.load('Experiments/taxi_q_tables_V2/QL_Penalty_size10_locs4.npy')
    eval_ql(Q, taxi_loc=None, pass_dest=None, episodes=50, render=False)
    