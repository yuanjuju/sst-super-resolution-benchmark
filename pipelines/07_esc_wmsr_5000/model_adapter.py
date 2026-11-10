"""Same model constructions as the respective source1000-step experiments."""
import sys
import torch
from screen_helpers import BASE


class WMSR01(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, image):
        return self.network(image * 255.) / 255.


def make_model(name, scale):
    if name == 'ESC':
        from factory_esc import create_model
        return create_model(scale)
    assert name == 'WMSR'
    sys.path.insert(0, str(BASE))
    from extended_models.factory import create_model
    network, _ = create_model({'model': 'WMSR', 'scale': scale, 'channels': 64})
    return WMSR01(network)


def prepare_model(model, name):
    if name == 'ESC':
        from factory_esc import cache_rpe_indices
        return cache_rpe_indices(model, 'cuda')
    return model
