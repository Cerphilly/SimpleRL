def copy_weight(network, target_network):
    target_network.load_state_dict(network.state_dict())


def soft_update(network, target_network, tau):
    for v1, v2 in zip(network.parameters(), target_network.parameters()):
        v2.data.copy_(tau*v2.data + (1-tau)*v1.data)