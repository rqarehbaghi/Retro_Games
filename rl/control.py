"""Optional model-based afterstate control backup over an observed decision set.

The piece/chance outcome is sampled from the real environment. Only the action
choice is maximized over simulator candidates. Execution-model errors can bias
this target; keep sampled observed-transition training as the default/control.
"""
import random
import numpy as np
import torch
import torch.nn.functional as F


class CandidateReplay:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError('Replay capacity must be positive')
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, candidates=None, terminal_reward=None):
        if not candidates and terminal_reward is None:
            raise ValueError('An empty candidate set is not evidence of death')
        features = np.asarray([c['afterstate'] for c in candidates], dtype=np.float32) if candidates else None
        rewards = np.asarray([c.get('immediate_reward', 0.) for c in candidates], dtype=np.float32) if candidates else None
        item = (np.array(state, copy=True), features, rewards, terminal_reward)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)


def candidate_targets(agent, batch):
    """Double-network action selection/evaluation; one target per real state."""
    lengths = [0 if row[1] is None else len(row[1]) for row in batch]
    chunks = [row[1] for row in batch if row[1] is not None]
    with torch.no_grad():
        if chunks:
            feats = torch.as_tensor(np.concatenate(chunks), device=agent.device)
            online = agent.val_net(feats)
            target = agent.target_net(feats)
        values = []
        offset = 0
        for row, count in zip(batch, lengths):
            if not count:
                values.append(float(row[3]))
                continue
            rewards = torch.as_tensor(row[2], device=agent.device)
            choice = torch.argmax(rewards + agent.gamma * online[offset:offset+count])
            values.append(float(rewards[choice] + agent.gamma * target[offset+choice]))
            offset += count
    return torch.tensor(values, device=agent.device, dtype=torch.float32)


def update_candidates(agent, replay, batch_size):
    batch = random.sample(replay.buffer, batch_size)
    states = torch.as_tensor(np.asarray([r[0] for r in batch]), dtype=torch.float32, device=agent.device)
    targets = candidate_targets(agent, batch)
    loss = F.smooth_l1_loss(agent.val_net(states), targets)
    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.val_net.parameters(), 1.)
    agent.optimizer.step()
    return float(loss.item())
