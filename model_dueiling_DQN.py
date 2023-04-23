import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc3_units=32,fc4_units=16):

        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int), fc2_units (int), fc3_units (int), fc4_units (int): Number of nodes in 1-4 hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size=action_size
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.val = nn.Linear(fc4_units, 1)
        self.adv = nn.Linear(fc4_units, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x_val = self.val(x).expand(x.size(0),self.action_size)
        x_adv =self.adv(x)
        
        return x_val+x_adv-x_adv.mean(1).unsqueeze(1).expand(x.size(0),self.action_size)
