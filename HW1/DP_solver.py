import numpy as np

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        p_a = 1
        q_state_value = 0.0
        next_state, reward, done = self.grid_world.step(state, action)
        q_state_value = reward + self.discount_factor * p_a * self.get_values()[next_state] * (1-done)
        return q_state_value


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        
        v_state_value = 0.0
        for action in range(self.grid_world.get_action_space()):
            v_state_value += self.policy[state][action] * self.get_q_value(state, action)        
        return v_state_value
        
    def evaluate(self) -> np.ndarray:
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step

        v_update_value = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            v_update_value[state] = self.get_state_value(state)
        return v_update_value

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        
        diff = 1
        while diff > self.threshold:
            v_update_value = self.evaluate()
            diff = max(abs(self.values - v_update_value))
            self.values = v_update_value


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        
        v_state_value = 0.0
        v_state_value += self.get_q_value(state, self.policy[state])        
        return v_state_value

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        
        v_update_value = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            v_update_value[state] = self.get_state_value(state)
        return v_update_value

    def policy_improvement(self) -> bool:
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        
        old_action = np.zeros(self.grid_world.get_action_space())
        policy_stable = True
        for state in range(self.grid_world.get_state_space()):
            old_action = self.policy[state]
            new_action = np.zeros(self.grid_world.get_action_space())
            
            for action in range(self.grid_world.get_action_space()):
                new_action[action] = self.get_q_value(state, action) 
                
            new_action_index = np.argmax(new_action)
            self.policy[state] = new_action_index
            
            if old_action != new_action_index :
                policy_stable = False
        
        return policy_stable
                

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        
        diff = 1
        while diff > self.threshold:
            v_update_value = self.policy_evaluation()
            diff = max(abs(self.values - v_update_value))
            self.values = v_update_value
            
            policy_stable = self.policy_improvement()
            if policy_stable == True:
                break
            


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        v_state_value = 0.0
        for action in range(self.grid_world.get_action_space()):
            v_state_value += self.policy[state][action] * self.get_q_value(state, action)        
        return v_state_value

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        raise NotImplementedError


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        raise NotImplementedError
