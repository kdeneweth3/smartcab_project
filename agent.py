import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import operator
import numpy as np
import time
from collections import OrderedDict


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.QvaluePrev = 0
        self.initQvalue = 0   # initial Qvalue if the value does not exist in the table 
        self.Qtable = {'start':{None:0}}
        self.statePrev = 'start'
        self.actionPrev = None
        self.QvaluePrev = 0
        self.initDeadline = 0
        self.dtime = 0
        self.level_1 = 0
        self.level_2 = 0
        self.level_3 = 0
        self.level_4 = 0
        self.qvalue_change = np.empty([0, 0])  # array of changes in qvalues
        self.trial_number = 0


    def reset(self, destination=None):
        self.planner.route_to(destination)
        
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.initDeadline = None
        self.dtime = 0
        self.trial_number += 1
        
        # Change params HERE
        self.alpha = 0.5/self.trial_number    # learning rate =  1 = full learning, 0 = no learning
        self.gamma = 0.5/self.trial_number    # discount rate = Gamma of 1 emphasizes long term rewards, 0 for short term rewards
        self.epsilon = 0.5/self.trial_number  # epsilon = how often to choose a random action, 1 = always, 0 = never


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        if self.dtime == 0:
            self.initDeadline = deadline
            self.dtime += 1
#            
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

        if deadline >= 0:
            deadline = 'met_deadline'
        else:
            deadline = 'missed_deadline'

        
        
        # TODO: Update state
        tmpinputs = inputs    ## create another dictionary with all inputs bc inputs needs to be modified
        del tmpinputs['right']    ## remove the input 'right' because it is not relevant to our state
        state_list = [tmpinputs[i] for i in tmpinputs.keys()]    ## make a list out of the values from the inputs dictionary
        [state_list.append(a) for a in [deadline, self.next_waypoint]]    ##  append deadline and next_waypoint to the state list
        self.state = tuple(state_list)    ## creates tuple of states as (Light, Oncoming, Left, Deadline, Next waypoint)
        
        
        # TODO: Select action according to your policy
        possible_actions = [None, 'forward', 'left', 'right']
        randaction = random.choice(possible_actions)
        rand_num = random.random()
        action_prob = 1-self.epsilon
        if self.state in self.Qtable:
            sdict = self.Qtable[self.state]
            if len(sdict) == 4 and rand_num < action_prob:    
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

        # TODO: Learn policy based on state, action, reward
        if action in self.Qtable[self.state]:
            pass
        else:
            self.Qtable[self.state][action] = 0    # set default value for action

        Qvalue = self.Qtable[self.state][action]
        Qvalueupdate = (1-self.alpha)*self.QvaluePrev + self.alpha*(reward + (self.gamma*Qvalue))
        self.Qtable[self.statePrev][self.actionPrev] = Qvalueupdate    # Set updated Qvalue in dictionary
        #Q(s,a) = (1-self.alpha)*Q(s,a) + self.alpha*(R' + self.gamma*Q(s',a'))
        
        
        # Store values for next iteration
        self.statePrev = self.state
        self.actionPrev = action
        self.QvaluePrev = self.Qtable[self.statePrev][self.actionPrev]
        
        
        # append the change in Qvalues to numpy array
        qvalue_diff = abs(Qvalueupdate - self.QvaluePrev)
        self.qvalue_change = np.append(self.qvalue_change, qvalue_diff)
        
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #print '\nQtable:', self.Qtable

def run():
    """Run the agent for a finite number of trials."""

    num_trials = 100
    
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=.0000001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=num_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    return sim.record, a.Qtable, num_trials # returns an ordered dictionary with {trail number:{win/lose:% time remaining}}
                        # example: win with 25% of the time remaining, took 75% of the time to get there

if __name__ == '__main__':
    
    resultdict = OrderedDict()
    diclist = ['win-25', 'win-50', 'win-75', 'win-100', 'win-total', 'lose']
    for d in diclist:
        resultdict[d] = 0

    for x in range(3):
        rec, Qtable, num_trials = run()
    
        for i in rec:
            if rec[i][0] == 'Win':
                resultdict['win-total'] += 1
                if i <= num_trials*0.25:
                    resultdict['win-25'] += 1
                elif num_trials*0.25 < i <= num_trials*0.50:
                    resultdict['win-50'] += 1
                elif num_trials*0.50 < i <= num_trials*0.75:
                    resultdict['win-75'] += 1
                elif num_trials*0.75 < i <= num_trials*1.0:
                    resultdict['win-100'] += 1
            else:
                resultdict['lose'] += 1
    print '\n', resultdict
    

'''
Run = 100 trials 3 times

--- Parameters 1 --- 
alpha = 1
gamma = 1
epsilon = 1
empty qvalue = 0

OrderedDict([('win-25', 16), ('win-50', 18), ('win-75', 16), ('win-100', 16), 
             ('win-total', 66), ('lose', 234)])


--- Parameters 2 --- 
alpha = 1
gamma = 1
epsilon = 1
empty qvalue = 3

OrderedDict([('win-25', 20), ('win-50', 19), ('win-75', 14), ('win-100', 24), 
             ('win-total', 77), ('lose', 223)])


--- Parameters 3 --- 
alpha = 0.5
gamma = 0.5
epsilon = 0.5
empty qvalue = 0

OrderedDict([('win-25', 11), ('win-50', 19), ('win-75', 25), ('win-100', 26), 
             ('win-total', 81), ('lose', 219)])
0.27

OrderedDict([('win-25', 278), ('win-50', 285), ('win-75', 317), ('win-100', 313), 
             ('win-total', 1193), ('lose', 1807)])
0.40

OrderedDict([('win-25', 2194), ('win-50', 2279), ('win-75', 2314), ('win-100', 2271), 
             ('win-total', 9058), ('lose', 20942)])
0.30

OrderedDict([('win-25', 10682), ('win-50', 0), ('win-75', 0), ('win-100', 0), 
             ('win-total', 10682), ('lose', 7292)])
almost 18,000 trials = 0.59


OrderedDict([('win-25', 12), ('win-50', 2), ('win-75', 0), ('win-100', 4), ('win-total', 18), ('lose', 9982)])
10,000 trials with deadline as binary = poor!


--- Parameters 4 --- 
alpha = 0.1
gamma = 0.1
epsilon = 0.1
empty qvalue = 0

OrderedDict([('win-25', 10), ('win-50', 15), ('win-75', 17), ('win-100', 17), 
             ('win-total', 59), ('lose', 241)])


--- Parameters 5 --- 
alpha = 0.5
gamma = 0.5
epsilon = 0.5
empty qvalue = 0
no deadline in state, 30 runs

OrderedDict([('win-25', 124), ('win-50', 117), ('win-75', 109), ('win-100', 119), 
             ('win-total', 469), ('lose', 2531)])


--- Parameters 6 --- 
alpha = 1
gamma = 0.5
epsilon = 0.5
empty qvalue = 0

OrderedDict([('win-25', 17), ('win-50', 18), ('win-75', 5), ('win-100', 18), 
             ('win-total', 58), ('lose', 242)])



--- Parameters 7 --- 
alpha = 0.5
gamma = 1
epsilon = 0.5
empty qvalue = 0

OrderedDict([('win-25', 21), ('win-50', 23), ('win-75', 19), ('win-100', 16), 
             ('win-total', 79), ('lose', 221)])


--- Parameters 8 --- 
alpha = 0.5
gamma = 0.5
epsilon = 1
empty qvalue = 0

OrderedDict([('win-25', 19), ('win-50', 18), ('win-75', 16), ('win-100', 15), 
             ('win-total', 68), ('lose', 232)])


--- Parameters 8 --- 
alpha = 1
gamma = 1
epsilon = 0.5
empty qvalue = 0

OrderedDict([('win-25', 17), ('win-50', 10), ('win-75', 14), ('win-100', 11), 
             ('win-total', 52), ('lose', 248)])



'''






