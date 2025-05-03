import gym
import numpy as np
from collections import deque
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import functional as TF


class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma_init=2.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.sigma_init = sigma_init
        self.mu_weight = nn.Parameter(torch.empty(out_f, in_f))
        self.mu_bias = nn.Parameter(torch.empty(out_f))
        self.sigma_weight = nn.Parameter(torch.empty(out_f, in_f))
        self.sigma_bias = nn.Parameter(torch.empty(out_f))
        self.register_buffer('weight_epsilon', torch.empty(out_f, in_f))
        self.register_buffer('bias_epsilon', torch.empty(out_f))
        self.reset_parameters()
        self.reset()
    def reset_parameters(self):
        bound = 1 / (self.in_f ** 0.5)
        bound = 1.0 / (self.in_f ** 0.5)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(self.sigma_init / (self.in_f ** 0.5))
        self.sigma_bias.data.fill_(  self.sigma_init / (self.out_f ** 0.5))

    @staticmethod
    def _scale_noise(dim):
        x = torch.randn(dim)
        return x.sign().mul_(x.abs().sqrt_())

    def reset(self):
        eps_in = self._scale_noise(self.in_f)
        eps_out = self._scale_noise(self.out_f)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.weight_epsilon
            bias = self.mu_bias + self.sigma_bias * self.bias_epsilon
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)

class DuelingCNN(nn.Module):
    def __init__(self, in_c, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 48, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=1)
 
        with torch.no_grad():
            dummy = torch.zeros(1, in_c, 84, 90)
            flat_dim = self.encode(dummy).shape[1]
        hidden = 768
        self.val_h = NoisyLinear(flat_dim, hidden)
        self.val_o = NoisyLinear(hidden, 1)
        self.adv_h = NoisyLinear(flat_dim, hidden)
        self.adv_o  = NoisyLinear(hidden, n_actions)
    def encode(self, x):
        x = x.float().div(255.0)          
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(start_dim=1)
    def forward(self, obs):
        features = self.encode(obs)          
        value_hidden = F.relu(self.val_h(features))
        state_value = self.val_o(value_hidden)
        adv_hidden = F.relu(self.adv_h(features))
        action_adv = self.adv_o(adv_hidden)
        q_values = state_value + action_adv - action_adv.mean(dim=1, keepdim=True)
        return q_values
    def noise_resampling(self):
        for layer in (self.val_h, self.val_o, self.adv_h, self.adv_o):
            layer.reset()

class FrameBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def initialize(self, frame: np.ndarray):
        self.buffer.clear()
        for _ in range(self.capacity):
            self.buffer.append(frame)

    def append(self, frame: np.ndarray):
        self.buffer.append(frame)

    def stack(self) -> np.ndarray:
        return np.stack(self.buffer, axis=0)


def preprocess_frame(raw_obs: np.ndarray, processor: transforms.Compose) -> np.ndarray:
    tensor = processor(raw_obs).squeeze(0)
    return tensor.numpy()


class Agent:
    def __init__(self):
        # expose action_space exactly as in your stub
        self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))

        # device & model (defaults to CPU)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = DuelingCNN(in_c=4, n_actions=self.action_space.n).to(self._device)
        checkpoint = torch.load("./my_model_1.pth", map_location=self._device)
        self._model.load_state_dict(checkpoint['model'])
        self._model.eval()

        # frame preprocessing pipeline
        self._processor = self._frame_to_tensor

        # frame stacking + skip logic
        self._frame_buffer   = FrameBuffer(capacity=4)
        self._skip_interval  = 4 - 1
        self._skip_remaining = 0
        self._last_action    = 0
        self._initialized    = False
    
    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        pil_img = transforms.ToPILImage()(frame)
        gray = TF.rgb_to_grayscale(pil_img, num_output_channels=1)
        resized = TF.resize(gray, (84, 90))
        return TF.to_tensor(resized)
    
    def act(self, observation):
        frame = preprocess_frame(observation, self._processor)

        if not self._initialized:
            self._frame_buffer.initialize(frame)
            self._initialized = True

        if self._skip_remaining > 0:
            self._skip_remaining -= 1
            return self._last_action

        buf = self._frame_buffer
        buf.append(frame)
        state = buf.stack()
        tensor_state = torch.tensor(state, device=self._device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self._model(tensor_state)

        action = int(q_vals.argmax(dim=1).item())
        self._last_action    = action
        self._skip_remaining = self._skip_interval
        return action