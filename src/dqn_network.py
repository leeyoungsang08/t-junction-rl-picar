import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, hidden=(128, 128), state_dim=12, actions=2):
        super().__init__()
        layers, last = [], state_dim
        for size in hidden:
            layers += [nn.Linear(last, size), nn.ReLU()]
            last = size
        layers.append(nn.Linear(last, actions))
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)
