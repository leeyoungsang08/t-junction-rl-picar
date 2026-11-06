import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP DQN with an optional Dueling head (Wang et al., ICML 2016).

    Non-dueling state_dict keys remain compatible with the original V1/V2 models.
    """

    def __init__(self, hidden=(128, 128), state_dim=12, actions=2, dueling=False):
        super().__init__()
        self.dueling = bool(dueling)
        layers, last = [], state_dim
        for size in hidden:
            layers += [nn.Linear(last, size), nn.ReLU()]
            last = size
        if self.dueling:
            self.layers = nn.Sequential(*layers)
            self.value = nn.Linear(last, 1)
            self.advantage = nn.Linear(last, actions)
        else:
            layers.append(nn.Linear(last, actions))
            self.layers = nn.Sequential(*layers)

    def forward(self, state):
        features = self.layers(state)
        if not self.dueling:
            return features
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=-1, keepdim=True)
