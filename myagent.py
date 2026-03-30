import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()
        # Flattened 6x6 board = 36 inputs
        
        # Actor Head
        self.actor_head = nn.Linear(36, 1296)
        
        # Critic Head
        self.critic_head = nn.Linear(36, 1)

    def forward(self, x):
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value

class ACAgent:
    def __init__(self, gamma=0.99, lr=3e-4):
        self.gamma = gamma
        self.network = ActorCriticNetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr) # both w and theta start with the same learning rate for simplicity

    def get_action(self, observation_dict, action_override=None):
        """
        Action selection method that also returns the log probability of the action and the value estimate for the current state.
        If action_override is provided, it will be used instead of sampling from the policy.
        """
        # Convert observation to tensors
        obs_tensor = torch.FloatTensor(observation_dict["observation"]).flatten().unsqueeze(0)
        mask_tensor = torch.BoolTensor(observation_dict["action_mask"]).unsqueeze(0)
        
        # Forward pass
        logits, value = self.network(obs_tensor)
        
        # Apply mask safely using masked_fill to prevent in-place modification errors
        logits = logits.masked_fill(~mask_tensor, -1e9)
        
        # Softmax to get action probabilities and create distribution
        dist = Categorical(logits=logits)
        
        # recompute instead of store previous action (new weights)
        if action_override is None:
            action_tensor = dist.sample() # action selection
        else:
            action_tensor = torch.tensor([action_override])
            
        log_prob = dist.log_prob(action_tensor)
        
        return action_tensor.item(), log_prob, value

    def update(self, prev_obs_dict, prev_action, reward, current_obs_dict, done, I_factor=1.0):
        """
        one-step Actor-Critic update
        """
        # recompute forward pass for previous state with current weights
        _, log_prob, value = self.get_action(prev_obs_dict, action_override=prev_action)
        
        # bootstrap next-state value
        if not done:
            with torch.no_grad(): # don't use gradients from this forward pass to update network weights
                _, _, next_value = self.get_action(current_obs_dict)
        else:
            # terminal states have value 0
            next_value = torch.tensor([[0.0]])

        # TD error
        delta = reward + self.gamma * next_value.item() - value.item()

        critic_loss = -delta * value # -delta for gradient ascent
        actor_loss = -I_factor * delta * log_prob
        total_loss = actor_loss + critic_loss # combine losses for single backward pass (pytorch will handle the gradients correctly)
        
        self.optimizer.zero_grad() # Clear old gradients before backward pass
        total_loss.backward() # Backwards pass to calculate gradients
        self.optimizer.step() # Enact weight update

        return total_loss.item()

    def save(self, filepath):
        """Saves the network weights to a file."""
        torch.save(self.network.state_dict(), filepath)

    def load(self, filepath):
        """Loads the network weights from a file."""
        self.network.load_state_dict(torch.load(filepath, weights_only=True))