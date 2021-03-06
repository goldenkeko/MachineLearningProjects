""" Author Kelsey Odenthal

Various small portions of the code was referenced from stack
exchange for assistance with various syntax difficulties"""

import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
import pygame
from pygame.locals import *
import collections
import numpy as np

DEBUG = False

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=True, epsilon=1, alpha=0.9):
        print "init start"
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = defaultdict(dict)          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        #t = self.t
        self.decay = 0.001
        self.trials = 0
        #q_values = self.Q[self.state]
        print "init end"

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """
        print "reset start"
        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
 
        self.trials += 1
        if testing:
            self.epsilon = 0
            self.alpha = 0
        else:
            self.epsilon = (math.cos(self.alpha * self.trials))
            print "epsilon" + str(self.epsilon)

        print "reset end"
        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """
        print "build_state start"
        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent        
#        state = (waypoint, inputs, deadline)
        def to_string(string):
            if string is None:
                return 'Nope'
            else:
                return str(string)

        state = to_string(waypoint) + "_" + inputs['light'] + "_" + to_string(inputs['left']) + "_" + to_string(inputs['right']) + "_" + to_string(inputs['oncoming'])        
        print state

        if self.learning:
            self.Q[state] = self.Q.get(state, {None:0.0, 'forward':0.0, 'left':0.0, 'right':0.0})
#            print "Q" + str(self.Q)
#            print "Q state" + str(self.Q[state])
        print "build state end"
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """
        print "get maxQ start"
        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

        max_Q_action = max(self.Q[state], key = lambda x: self.Q[state][x])
#        for action in self.Q[state]:
        maxQ = self.Q[state][max_Q_action]
        print "max: " + str(maxQ)
        print "get maxQ end"
        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """
        print "createQ start"
        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        #deadline = self.env.get_deadline(self)
        #inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
#        print "next waypoint: " + str(self.next_waypoint)
        if self.learning == True:
            if state not in self.Q:
#            print "self.learning = true"
#            self.Q[state]
#            self.Q.update(state)
#            print "state.. " + str(state)
#            self.Q = None
                self.Q[state] = self.Q.get(state, {None:0.0, 'forward':0.0, 'left':0.0, 'right':0.0})
#            print "Q: " + str(self.Q)
#        else:
            
            
            

        print "createQ end"
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """
        print "choose action start"
        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        print "next waypoint " + str(self.next_waypoint)
        action = random.choice(self.valid_actions)
#        print "original valid actions " + str(self.valid_actions)
        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        p = random.randrange(0,1)
        if self.learning:
            if self.epsilon > p:
                self.epsilon -= self.decay
                print "bigger epsilon " + str(self.epsilon)
                action = random.choice(self.valid_actions)

#                print "p: " + str(p)
#                print "epsilon: " + str(self.epsilon)
#                print "valid actions: " + str(self.valid_actions)
#                print "working to explore"
            else:
                print "small epsilon " + str(self.epsilon)
#                action = self.max_action(state)[0]
                valid_actions = []
#                q = [self.Q(state,action) for action in self.valid_actions]                
                current_maxQ = self.get_maxQ(state)
#                count = q.count(current_maxQ)
#                print count
                print "current max: " + str(current_maxQ)
                for actions in self.Q[state]:
#                    print "Q state: " + str(self.Q[state])
#                    print "Q state action: " + str(self.Q[state][action])
                    if current_maxQ == self.Q[state][actions]:                        
                        valid_actions.append(actions)
#                        action = random.choice(current_maxQ)
                        print "valid actions" + str(valid_actions)
                action = random.choice(valid_actions)
#        else:
#            action = random.choice(self.valid_actions)
#        print "Action: " + str(action)       
        print "choose action end"
        #else:
            #get_maxQ(self.valid_actions)
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """
        print "learn start"
        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        print "alpha: " + str(self.alpha)
        
        
        if self.learning:
#            self.Q[state][action] = ((1-self.alpha) * self.Q[state][action]) + self.alpha *(reward + self.decay * self.Q[state][action])
            self.Q[state][action] = (1-self.alpha) * self.Q[state][action] + self.alpha*reward            
            print "state action" + str(self.Q[state][action])            
        print "learn end"   
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """
        print "update start"
        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn
        print "update end"
        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """
    print "run start"
    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose = True, num_dummies = 100, grid_size = (8,6))    
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning = True, epsilon = 1.0, alpha = 0.9)
#    agent = env.create_agent(LearningAgent, learning = True)
   
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline = True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, display = True, update_delay = 0.01, optimized = True, log_metrics = True)
#    sim = Simulator(env, update_delay = 0.01, optimized = False, log_metrics = True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(tolerance = 0.07, n_test=10)
#    sim.run(n_test = 10)    
    print "run end"

if __name__ == '__main__':
    run()
