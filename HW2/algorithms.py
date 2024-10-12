import numpy as np
import json
from collections import deque

from gridworld import GridWorld

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """
       

    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0  
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state
        
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  

        next_state, reward, done = self.grid_world.step(action)  
        if done:
            self.episode_counter +=1
        return next_state, reward, done
        

class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction
        
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        current_state = self.grid_world.reset()
        
        episode = []
        # s => list of Gt
        returns = [[] for idx in range(self.state_space)]        
        
        while self.episode_counter < self.max_episode:
            # collect the episode data
            next_state, reward, done = self.collect_data()
            episode.append((current_state, reward))
            current_state = next_state
            
            if done:                
                # cache the first-visit
                first_visit_update = {}
                G = 0.0
                for i in range(len(episode)-1, -1, -1):
                    state, reward = episode[i]
                    G = self.discount_factor * G + reward
                    # first-visit in reverse order
                    first_visit_update[state] = G
                
                # update the state value
                for state, value in first_visit_update.items():
                    returns[state].append(value)
                    self.values[state] = sum(returns[state]) / len(returns[state])
                
                episode = []
                


class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld,learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        # TODO: Update self.values with TD(0) Algorithm
        current_state = self.grid_world.reset()
        
        while self.episode_counter < self.max_episode:
            next_state, reward, done = self.collect_data()
            
            td_target = reward
            if done == False:
                td_target += self.discount_factor * self.values[next_state]
                
            self.values[current_state] = self.values[current_state] + self.lr * (td_target - self.values[current_state])
            current_state = next_state
            
            


class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with N-step TD Algorithm
        current_state = self.grid_world.reset()
        
        episode = []
        while self.episode_counter < self.max_episode:
            next_state, reward, done = self.collect_data()
            episode.append((current_state, reward))
            current_state = next_state
            
            if done:
                # n-step look ahead
                for idx, (state, reward) in enumerate(episode):
                    G = 0.0
                    for j in range(idx, min(idx+self.n, len(episode))):
                        G += (self.discount_factor ** (j-idx)) * episode[j][1]
                    if idx + self.n < len(episode):
                        G += (self.discount_factor ** self.n) * self.values[episode[idx+self.n][0]]
                    td_error = (G - self.values[state])
                    self.values[state] = self.values[state] + self.lr * td_error
                episode = []

# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms 
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stocastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)                              # deterministic policy

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon
        
        self.rng = np.random.default_rng(1)

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        
        G = 0.0
        for i in range(len(state_trace)-1, -1, -1):
            state, action, reward = state_trace[i], action_trace[i], reward_trace[i]
            G = self.discount_factor * G + reward
            # every visit
            self.q_values[state][action] = self.q_values[state][action] + self.lr * (G - self.q_values[state][action])
        

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy
        
        # get max Q value index of each state
        self.get_policy_index()
        # update the policy
        for state in range(self.state_space):
            for action in range(self.action_space):
                if action == self.policy_index[state]:
                    self.policy[state][action] = 1 - self.epsilon + self.epsilon/self.action_space
                else:
                    self.policy[state][action] = self.epsilon/self.action_space


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        
        state_trace   = []
        action_trace  = []
        reward_trace  = []
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            
            # collect episode
            action_probs = self.policy[current_state]  
            action = self.rng.choice(self.action_space, p=action_probs)  
            next_state, reward, done = self.grid_world.step(action) 
            state_trace.append(current_state)
            action_trace.append(action)
            reward_trace.append(reward)
            current_state = next_state
            
            if done:
                self.policy_evaluation(state_trace, action_trace, reward_trace)
                self.policy_improvement()
                state_trace, action_trace, reward_trace = [], [], []
                iter_episode += 1

class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon
        
        self.rng = np.random.default_rng(1)

    def get_policy(self, state: int) -> int:
        """Get the policy based on epsilon-greedy"""
        best_q_i = self.q_values[state].argmax()
        for action in range(self.action_space):
            if action == best_q_i:
                self.policy[state][action] = 1 - self.epsilon + self.epsilon/self.action_space
            else:
                self.policy[state][action] = self.epsilon/self.action_space
        return self.policy[state]
    
    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        
        q_target = r
        if is_done == False:
            q_target += self.discount_factor * self.q_values[s2][a2]
        self.q_values[s][a] = self.q_values[s][a] + self.lr * (q_target - self.q_values[s][a])

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()

        action_probs = self.get_policy(current_state)  
        current_action = self.rng.choice(self.action_space, p=action_probs)  
        
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            
            next_state, reward, done = self.grid_world.step(current_action) 
            next_action = self.rng.choice(self.action_space, p=self.get_policy(next_state))  
            
            self.policy_eval_improve(current_state, current_action, reward, next_state, next_action, done)
            current_state = next_state
            current_action = next_action
            
            if done:
                iter_episode += 1
            

class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size
        
        self.rng = np.random.default_rng(1)
        
    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append((s, a, r, s2, d))
        if len(self.buffer) > self.buffer.maxlen:
            self.buffer.popleft()

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        
        batch_list = self.rng.choice(len(self.buffer), size=self.sample_batch_size)
        arr = []
        for idx in batch_list:
            arr.append(self.buffer[idx])
        return np.array(arr)

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        q_target = r
        if is_done == False:
            q_target += self.discount_factor * self.q_values[s2].max()
        self.q_values[s][a] = self.q_values[s][a] + self.lr * (q_target - self.q_values[s][a])

        # update the policy
        best_q_i = self.q_values[s].argmax()
        for action in range(self.action_space):
            if action == best_q_i:
                self.policy[s][action] = 1 - self.epsilon + self.epsilon/self.action_space
            else:
                self.policy[s][action] = self.epsilon/self.action_space
    
    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        transition_count = 0 
        
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            action_probs = self.policy[current_state]
            action = self.rng.choice(self.action_space, p=action_probs)
            next_state, reward, done = self.grid_world.step(action) 
            self.add_buffer(current_state, action, reward, next_state, done)
            transition_count += 1
            
            # update the Q value
            if transition_count % self.update_frequency == 0:
                for s, a, r, s2, d in self.sample_batch():
                    self.policy_eval_improve(int(s), int(a), r, int(s2), bool(d))
            
            current_state = next_state
            if done:
                iter_episode += 1
            