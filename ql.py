import numpy as np
import argparse
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2

def run_Q_learning(episodes=20, alpha=0.1, epsilon=0.1, gamma=0.99, init_val=0, tolerance=1e-10):
    '''Implementation of vanilla Q-learning'''
    Q_table = np.zeros([nonfair_env.observation_space.n, nonfair_env.action_space.n])
    Q_table = Q_table + init_val
    loss_data, nsw_data = [], []
    
    for i in range(1, episodes+1):
        R_acc = np.zeros(len(nonfair_env.loc_coords))
        state = nonfair_env.reset()
        print('Episode {}\nInitial State: {}'.format(i, nonfair_env.decode(state)))
        old_table = np.copy(Q_table)
        done = False
        c = 0
        
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = nonfair_env.action_space.sample()
            else:
                if np.all(Q_table[state] == Q_table[state][0]) == True: 
                    # if all values same, choose randomly, since np.argmax returns 0 when values are same
                    action = nonfair_env.action_space.sample()
                else:
                    action = np.argmax(Q_table[state])
            
            next, reward, done = nonfair_env.step(action)
            R_acc += np.power(gamma, c)*reward
            reward = np.sum(reward)     # turn vector reward into scalar
            Q_table[state, action] = (1 - alpha)*Q_table[state, action] + alpha*(reward+gamma*np.max(Q_table[next]))
            state = next
            c += 1

        loss = np.sum(np.abs(Q_table - old_table))
        loss_data.append(loss)
        nsw_score = nsw(R_acc, nsw_lambda=1e-4)
        nsw_data.append(nsw_score)
        print('Accumulated reward: {}\nLoss: {}\nNSW: {}\n'.format(R_acc,loss,nsw_score))
        if loss < tolerance:
            loss_count += 1
            if loss_count == 10: break  # need to be smaller for consecutive loops to satisfy early break
        else: loss_count = 0
        
    np.save(file='taxi_q_tables_V2/QL_Penalty_size{}_locs{}'.format(nonfair_env.size, len(nonfair_env.loc_coords)),
            arr=Q_table)
    np.save(file='taxi_q_tables_V2/QL_Penalty_size{}_locs{}_loss'.format(nonfair_env.size, len(nonfair_env.loc_coords)),
            arr=loss_data)
    np.save(file='taxi_q_tables_V2/QL_Penalty_size{}_locs{}_nsw'.format(nonfair_env.size, len(nonfair_env.loc_coords)),
            arr=nsw_data)
    print('FINISH TRAINING Q LEARNING')

def nsw(vec, nsw_lambda): 
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

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-learning on Taxi""")
    prs.add_argument("-f", dest="fuel", type=int, default=10000, required=False, help="Timesteps each trajectory\n")
    prs.add_argument("-ep", dest="episodes", type=int, default=10000, required=False, help="episodes.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.1, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.999, required=False, help="Discount rate\n")
    prs.add_argument("-i", dest="init_val", type=int, default=30, required=False, help="Initial values\n")
    prs.add_argument("-gs", dest="size", type=int, default=5, required=False, help="Grid size\n")
    prs.add_argument("-d", dest="dimension", type=int, default=2, required=False, help="Dimension of reward\n")
    args = prs.parse_args()
    
    size, loc_coords, dest_coords = get_setting(args.size, args.dimension)
    fuel = args.fuel
    nonfair_env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel, output_path='')
    run_Q_learning(episodes=args.episodes, alpha=args.alpha, 
                   epsilon=args.epsilon, gamma=args.gamma, init_val=args.init_val)
