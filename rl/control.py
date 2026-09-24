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
    def __init__(self, capacity, reward_key='immediate_reward',
                 selection_reward_key=None):
        if capacity <= 0:
            raise ValueError('Replay capacity must be positive')
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.reward_key = reward_key
        # The reward optimized by the value target can deliberately differ
        # from fixed guidance used by the behaviour policy.  Keep the two
        # separate: use selection rewards to choose the backed-up action, then
        # use task rewards to evaluate that choice.  When no separate key is
        # supplied, preserve the historical one-reward behaviour.
        self.selection_reward_key = selection_reward_key or reward_key

    def push(self, state, candidates=None, terminal_reward=None):
        if not candidates and terminal_reward is None:
            raise ValueError('An empty candidate set is not evidence of death')
        features = np.asarray([c['afterstate'] for c in candidates], dtype=np.float32) if candidates else None
        rewards = np.asarray([c.get(self.reward_key, c.get('immediate_reward', 0.))
                              for c in candidates], dtype=np.float32) if candidates else None
        selection_rewards = np.asarray([
            c.get(self.selection_reward_key, c.get(self.reward_key,
                                                    c.get('immediate_reward', 0.)))
            for c in candidates
        ], dtype=np.float32) if candidates else None
        terminals = np.asarray([c.get('terminal', False) for c in candidates], dtype=bool) if candidates else None
        item = (np.array(state, copy=True), features, rewards, terminal_reward,
                terminals, selection_rewards)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity
        return item

    @staticmethod
    def correct(item, index, reward, observed=None, terminal=False,
                selection_reward=None):
        """Replace the EXECUTED candidate's prediction with its measured outcome."""
        item[2][index] = reward
        item[4][index] = terminal
        if selection_reward is not None:
            item[5][index] = selection_reward
        if observed is not None:
            item[1][index] = observed

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
                values.append(float(getattr(agent, 'value_reward_scale', 1.0)) * float(row[3]))
                continue
            rewards = torch.as_tensor(row[2], device=agent.device)
            selection_rewards = torch.as_tensor(row[5], device=agent.device)
            live = torch.as_tensor(~row[4], device=agent.device)
            online_values = online[offset:offset+count].masked_fill(~live, 0.)
            target_values = target[offset:offset+count].masked_fill(~live, 0.)
            scaled_rewards = float(getattr(agent, 'value_reward_scale', 1.0)) * rewards
            scaled_selection = (float(getattr(agent, 'value_reward_scale', 1.0))
                                * selection_rewards)
            choice = torch.argmax(scaled_selection + agent.gamma * online_values)
            values.append(float(scaled_rewards[choice] + agent.gamma * target_values[choice]))
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
    if hasattr(agent, 'update_target_network'):
        agent.update_target_network()
    return float(loss.item())
