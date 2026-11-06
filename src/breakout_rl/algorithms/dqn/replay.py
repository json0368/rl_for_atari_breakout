"""Memory-efficient, uniform frame replay."""
import numpy as np
import torch

class ReplayBuffer:
    """Uniform replay with one uint8 frame per transition.

    FrameStackObservation repeats the reset frame. Life/game boundaries are
    kept explicitly, including the true successor of terminated/truncated steps.
    Three overwritten prefix frames preserve stacks at the oldest ring entry.
    One million transitions use about 6.6 GiB rather than 33 GiB of image arrays.
    """
    def __init__(self, capacity, observation_shape, seed):
        from collections import deque
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity, self.position, self.size, self.total = capacity, 0, 0, 0
        self.stack_size = observation_shape[0]
        self.frames = np.empty((capacity, *observation_shape[1:]), dtype=np.uint8)
        self.starts = np.empty(capacity, dtype=np.bool_)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.history = deque(maxlen=self.stack_size - 1)
        self.end_frames = {}
        self.previous_done = True
        self.latest_next_frame = None
        self.rng = np.random.RandomState(seed)

    @property
    def image_nbytes(self):
        return self.frames.nbytes

    def add(self, obs, next_obs, action, reward, terminated, truncated=False):
        i, t = self.position, self.total
        if self.size == self.capacity:
            old_t = t - self.capacity
            self.history.append((old_t, self.frames[i].copy(), bool(self.starts[i])))
            self.end_frames.pop(old_t, None)
        self.frames[i] = obs[-1]
        self.starts[i] = self.previous_done
        self.actions[i], self.rewards[i], self.terminated[i] = action, reward, terminated
        self.latest_next_frame = np.array(next_obs[-1], copy=True)
        if terminated or truncated:
            self.end_frames[t] = self.latest_next_frame.copy()
        self.previous_done = bool(terminated or truncated)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total += 1

    def _frame(self, t):
        if t < self.total - self.size:
            for old_t, frame, start in self.history:
                if old_t == t:
                    return frame, start
            raise IndexError("Frame history outside replay window")
        return self.frames[t % self.capacity], self.starts[t % self.capacity]

    def _observation(self, t):
        frames = []
        for j in range(self.stack_size):
            frame, start = self._frame(t - j)
            frames.append(frame)
            if start:
                frames.extend([frame] * (self.stack_size - len(frames)))
                break
        return np.stack(frames[::-1])

    def sample(self, batch_size, device):
        slots = self.rng.randint(self.size, size=batch_size)
        # Map ring slots to absolute transition IDs.
        oldest = self.total - self.size
        ids = oldest + ((slots - oldest) % self.capacity) if self.size == self.capacity else slots
        obs = np.stack([self._observation(int(t)) for t in ids])
        next_frames = []
        for t in ids:
            t = int(t)
            if t in self.end_frames:
                frame = self.end_frames[t]
            elif t == self.total - 1:
                frame = self.latest_next_frame
            else:
                frame = self._frame(t + 1)[0]
            next_frames.append(frame)
        next_obs = np.concatenate((obs[:, 1:], np.stack(next_frames)[:, None]), axis=1)
        return tuple(torch.as_tensor(x, device=device) for x in (
            obs, next_obs, self.actions[slots], self.rewards[slots], self.terminated[slots]))
