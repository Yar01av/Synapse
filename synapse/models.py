from torch import nn, flatten

from synapse.util import get_output_size


class DQNNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self._model = nn.Sequential(nn.Linear(input_shape, 150),
                                    nn.ReLU(),
                                    nn.Linear(150, 120),
                                    nn.ReLU(),
                                    nn.Linear(120, n_actions))

    def forward(self, x):
        return self._model(x)


class REINFORCENetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self._model = nn.Sequential(nn.Linear(input_shape, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, n_actions))

    def forward(self, x):
        return self._model(x)


class A2CNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self._base = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU()
        )

        self._value_head = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._policy_head = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        #base_output = self._base(x)
        return self._policy_head(x), self._value_head(x)


class A2CConvNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self._base = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, stride=4, kernel_size=8),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = get_output_size(self._base, input_shape)
        self._value_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self._policy_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        base_output = self._base(x.float()/256)
        base_output = flatten(base_output, start_dim=1)

        return self._policy_head(base_output), self._value_head(base_output)


class A3CNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self._base = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU()
        )

        self._value_head = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._policy_head = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        # base_output = self._base(x)
        return self._policy_head(x), self._value_head(x)