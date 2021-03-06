import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import operator
import numpy as np
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import itertools


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.initQvalue = env.qvalue   # initial Qvalue if the value does not exist in the table      
        
        
        self.Qtable = {'start':{None:0}}
        self.statePrev = 'start'
        self.actionPrev = None
        self.QvaluePrev = self.initQvalue
        self.initDeadline = 0
        self.dtime = 0
        self.level_1 = 0
        self.level_2 = 0
        self.level_3 = 0
        self.level_4 = 0
        self.qvalue_change = np.empty([0, 0])  # array of changes in qvalues
        self.trial_number = 1
        self.alpha_start = 0.8    # learning rate =  1 = full learning, 0 = no learning
        self.gamma_start = 1.0    # discount rate = Gamma of 1 emphasizes long term rewards, 0 for short term rewards
        self.epsilon_start = 0.3  # epsilon = how often to choose a random action, 1 = always, 0 = never
        self.alpha = env.alpha
        self.gamma = env.gamma
        self.epsilon = env.epsilon
        self.rewards = OrderedDict()
        self.total_reward = 0
        self.penalty_count = 0
        self.penalties = OrderedDict()


    def reset(self, destination=None):
        self.planner.route_to(destination)
        
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.initDeadline = None
        self.dtime = 0

        self.alpha = self.alpha_start/self.trial_number
        self.gamma = self.gamma_start/self.trial_number
        self.epsilon = self.epsilon_start/self.trial_number     
        
#        self.alpha = self.alpha - (self.alpha_start/100)
#        self.gamma = self.gamma - (self.gamma_start/100)
#        self.epsilon = self.epsilon - (self.epsilon_start/100)
        
        self.rewards[self.trial_number] = self.total_reward
        self.total_reward = 0
        
        self.penalties[self.trial_number] = self.penalty_count
        self.penalty_count = 0

        self.trial_number += 1
        
        # Secondary params
#        if self.trial_number <=90:
#            self.alpha = 0.9   # learning rate =  1 = full learning, 0 = no learning
#            self.gamma = 0.9   # discount rate = Gamma of 1 emphasizes long term rewards, 0 for short term rewards
#            self.epsilon = 0.9  # epsilon = how often to choose a random action, 1 = always, 0 = never
#        else:
#            self.alpha = 0   
#            self.gamma = 0 
#            self.epsilon = 0




    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        if self.dtime == 0:
            self.initDeadline = deadline
            self.dtime += 1
            
#            # scale deadline into 4 ordinal values
#            deadline_range = range(1,self.initDeadline+1,1)
#            self.level_1 = int(np.percentile(deadline_range, 100))
#            self.level_2 = int(np.percentile(deadline_range, 75))
#            self.level_3 = int(np.percentile(deadline_range, 50))
#            self.level_4 = int(np.percentile(deadline_range, 25))
#            
#        if self.level_1 >= deadline > self.level_2:
#            deadline = 'level_1'
#        elif self.level_2 >= deadline > self.level_3:
#            deadline = 'level_2'
#        elif self.level_3 >= deadline > self.level_4:
#            deadline = 'level_3'
#        elif deadline <= self.level_4:
#            deadline = 'level_4'
#        
#        gamma = self.gamma
#        gamma = gamma - (self.gamma/self.initDeadline)


#        if deadline >= 0:
#            deadline = 'met_deadline'
#        else:
#            deadline = 'missed_deadline'

        
        # TODO: Update state
        tmpinputs = inputs    ## create another dictionary with all inputs bc inputs needs to be modified
        del tmpinputs['right']    ## remove the input 'right' because it is not relevant to our state
        state_list = [tmpinputs[i] for i in tmpinputs.keys()]    ## make a list out of the values from the inputs dictionary
        [state_list.append(a) for a in [self.next_waypoint]]    ## deadline,  append deadline and next_waypoint to the state list
        self.state = tuple(state_list)    ## creates tuple of states as (Light, Oncoming, Left, Deadline, Next waypoint)
        
        
        # TODO: Select action according to your policy
        possible_actions = self.env.valid_actions
        randaction = random.choice(possible_actions)
        rand_num = random.random()
        action_prob = 1-self.epsilon
        if self.state in self.Qtable:
            sdict = self.Qtable[self.state]
            if len(sdict) == len(possible_actions) and rand_num < action_prob:    
                action = max(sdict.iteritems(), key=operator.itemgetter(1))[0]
            else:
                missing_actions = list(set(possible_actions)-set(sdict.keys()))
                if len(missing_actions) > 0:
                    action = random.choice(missing_actions)
                else:
                    action = randaction
        else:
            self.Qtable[self.state] = {randaction:self.initQvalue}    # Initialize a nested dictionary for new state, give default value too
            action = randaction
        

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward = self.total_reward + reward 
        if reward < 0:
            self.penalty_count += 1 
        
        # TODO: Learn policy based on state, action, reward
        if action not in self.Qtable[self.state]:
            self.Qtable[self.state][action] = self.initQvalue    # set default value for action

        Qvalue = self.Qtable[self.state][action]
        Qvalueupdate = (1-self.alpha)*self.QvaluePrev + self.alpha*(reward + (self.gamma*Qvalue))
        self.Qtable[self.statePrev][self.actionPrev] = Qvalueupdate    # Set updated Qvalue in dictionary
        
        ''' 
        Q(s,a) = (1-self.alpha)*Q(s,a) + self.alpha*(R' + self.gamma*Q(s',a'))
        
        All learning: Q = (1-1)*1 + 1*(2 + (1*1.5))
        No learning: Q = (1-0)*1 + 0*(2 + (0*1.5))
        '''        
        
        
        
        # Store values for next iteration
        self.statePrev = self.state
        self.actionPrev = action
        self.QvaluePrev = self.Qtable[self.statePrev][self.actionPrev]
        
        
        # append the change in Qvalues to numpy array
        qvalue_diff = abs(Qvalueupdate - self.QvaluePrev)
        self.qvalue_change = np.append(self.qvalue_change, qvalue_diff)
        
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #print '\nQtable:', self.Qtable

def run(alpha, gamma, epsilon, qvalue):
    """Run the agent for a finite number of trials."""

    num_trials = 100
    num_dummies = 3
    
    # Set up environment and agent
    e = Environment(alpha, gamma, epsilon, qvalue)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=.0000001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=num_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    return sim.record, a.Qtable, num_trials, a.rewards, a.penalties # returns an ordered dictionary with {trail number:{win/lose:% time remaining}}
                        # example: win with 25% of the time remaining, took 75% of the time to get there

if __name__ == '__main__':  
    
    def brute_force(all_combinations):
        result_table = pd.DataFrame()
        #all_combinations = all_combinations[:10]    # use this for debugging
        print '\n\n\nApplying brute force....\n\n\n'
        time.sleep(3)    
        
        for combo in all_combinations:
            #starttime = time.time()
            rec, Qtable, num_trials, rewards, penalties = run(combo[0], combo[1], combo[2], combo[3])
            win_lose = []
            time_rem = []
            for trial in rec:
                win_lose.append(rec[trial][0])
                time_rem.append(rec[trial][1])
            last10_success_rate = win_lose[-10:].count('Win')/float(len(win_lose[-10:]))
            last10_num_penalties = np.cumsum(penalties.values()[-10:])[9]
            last10_med_time_remaining = np.median(time_rem[-10:])
            success_rate = win_lose.count('Win')/float(len(win_lose))
            num_penalties = np.cumsum(penalties.values())[len(penalties)-1]
            med_time_remaining = np.median(time_rem)
            
            s1 = pd.Series([combo[0], combo[1], combo[2], combo[3], 
                            last10_success_rate, last10_num_penalties, last10_med_time_remaining, 
                            success_rate, num_penalties, med_time_remaining], 
                           index=['alpha', 'gamma', 'epsilon', 'qvalue', 
                           'last10_success_rate', 'last10_num_penalties', 'last10_med_time_remaining',
                           'success_rate', 'num_penalties', 'med_time_remaining'])
            result_table = result_table.append(s1, ignore_index=True)
            #print 'time:', time.time() - starttime
        return result_table
    
    # normal run
    combo = tuple([0.8, 0.3, 0.8, 5.0])    # alpha, gamma, epsilon, qvalue
    rec, Qtable, num_trials, rewards, penalties = run(combo[0], combo[1], combo[2], combo[3]) 
        
    
    
    
'''
## write table to file
result_table.to_csv("C:/Users/kenny/Google Drive/Personal/Udacity/ML Nanodegree/P4 - Smart Cab/original code.csv", 
                    index = False, encoding='utf-8')

## initial brute force
alpha = gamma = epsilon = np.arange(0,1,0.1)
qvalue = range(0, 11)
all_combinations = list(itertools.product(alpha, gamma, epsilon, qvalue))


## work with 1% of best answers, 
# narrow down by: last10 success rate first, 
# run each of these 3x and average
r_table = pd.read_csv("C:/Users/kenny/Google Drive/Personal/Udacity/ML Nanodegree/P4 - Smart Cab/original code.csv")
sub = r_table[r_table.last10_success_rate==1]
sub_sort = sub.sort_values(by=['last10_success_rate', 'last10_num_penalties', 'last10_med_time_remaining'], 
                ascending=[False, True, False])
sub_sort = sub_sort[:100]
for i in sub_sort.iterrows():
    optimal_params = zip(sub_sort.alpha, sub_sort.gamma, sub_sort.epsilon, sub_sort.qvalue)
    

## doing things here
r_table = pd.read_csv("C:/Users/kenny/Google Drive/Personal/Udacity/ML Nanodegree/P4 - Smart Cab/original code.csv")
sub = r_table[r_table.last10_success_rate==1]
sub_sort = sub.sort_values(by=['last10_success_rate', 'last10_num_penalties', 'last10_med_time_remaining'], 
            ascending=[False, True, False])
sub_sort = sub_sort[:100]
for i in sub_sort.iterrows():
    optimal_params = zip(sub_sort.alpha, sub_sort.gamma, sub_sort.epsilon, sub_sort.qvalue)

tot_count = 1
combine_result = pd.DataFrame()
for rounds in range(3):
    final_result = brute_force(optimal_params)
    final_result.to_csv("C:/Users/kenny/Google Drive/Personal/Udacity/ML Nanodegree/P4 - Smart Cab/optimal" + str(rounds) + ".csv", 
                    index = False, encoding='utf-8')
    tot_count += 1
    combine_result = pd.concat([combine_result,final_result], ignore_index=True)
    
    
## find max of 100 best params
result_sort = combine_result.sort_values(by=['last10_success_rate', 'last10_num_penalties', 'last10_med_time_remaining'], 
                ascending=[False, True, False])
'''


# plot rewards
x = rewards.keys()
y = rewards.values()
u = penalties.values()
plt.figure(figsize=(8,4))
trewards = plt.scatter(x,y, color = 'b', label = 'total rewards')
tpenalties = plt.scatter(x,u, color = 'r', label = 'total penalties')
plt.legend(handles=[trewards, tpenalties], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.plot(x,y, color = 'b')
plt.plot(x,u, color = 'r')
plt.show()

# plot win/lose and percentage of time left
a = rewards.keys()
a1 = a[:-10]
a2 = a[-10:]
b = [1 if rec[i][0] == 'Win' else 0 for i in rec]
c = [rec[i][1] for i in rec]
d = b[:-10]
e = b[-10:]
plt.figure(figsize=(8,4))
perc = plt.scatter(a,c, color = 'r', label = '% time rem.')
win = plt.scatter(a1,d, color = 'b', label = 'Win/lose')
last = plt.scatter(a2,e, color = 'g', label = 'Last 10')
plt.legend(handles=[win, perc, last], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
    

'''
try:
-with mapping the qtable as I go (current)
-creating init qvalues for all action when a new state is reached (uses recs from reviewer- under 'magic nums')
-- estimating this will give superior results





'''

