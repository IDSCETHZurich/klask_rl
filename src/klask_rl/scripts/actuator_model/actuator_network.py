import torch
import torch.nn as nn

class ActuatorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ActuatorNetwork, self).__init__()
        # Following the excerpt: 3 hidden layers of 32 units each
        # Use softsign activation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        # softsign activation: y = x/(1+|x|)
        self.activation = nn.Softsign()

    def forward(self, x_commands, x_states):
        if x_states is not None:
            x = torch.cat([x_commands, x_states], dim=1)
        else:
            x = torch.cat([x_commands], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc_out(x)
        return x
    