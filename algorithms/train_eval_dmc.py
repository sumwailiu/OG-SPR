import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Comment out if the machine does not support egl
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import spectral_norm as SN
from gymnasium import spaces
from utils.buffer import PriorityReplayBuffer
import copy


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    gpu_id: int = 0
    seed: int = 123
    torch_deterministic: bool = True
    cuda: bool = True
    bit_depth: int = 5
    domain: str = "cheetah"
    task: str = "run"
    total_timesteps: int = 1000000
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    enc_lr: float = 3e-4
    dec_lr: float = 3e-4
    buffer_size: int = int(5e5)
    gamma: float = 0.99
    batch_size: int = 256
    policy_noise: float = 0.2
    exploration_noise: float = 0.1
    learning_starts: int = 10000
    eval_per_steps: int = 5000
    n_step_return: int = 3
    latent_horizon: int = 5
    policy_frequency: int = 1
    target_update_freq: int = 250
    noise_clip: float = 0.3
    n_bins: int = 51
    grad_clip_norm: float = 20
    aux_task: bool = True
    obs_coef: float = 0.1 # auxiliary loss weight for Next-Observation Prediction
    rew_coef: float = 1.0 # auxiliary loss weight for Short-Term Value Prediction
    latent_coef: float = 5.0 # auxiliary loss weight for Latent Self-Prediction
    prioritized: bool = True # 是否启用 prioritized replay buffer
    log_root: str = "/root"
    exp_id: str = ""


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def dmc_evaluate(
    domain: str,
    task: str,
    eval_episodes: int,
    actor: nn.Module,
    critic: nn.Module,
    encoder: nn.Module,
    projector,
    bit_depth,
    device: torch.device = torch.device("cpu"),
):
    envs = DMCVisual2GymWrapper(domain, task, action_repeat=2,
                                     symbolic=False, bit_depth=bit_depth, image_size=(84, 84))
    envs = FrameStackWrapper(envs, num_stack=3)

    actor.eval()
    critic.eval()
    encoder.eval()

    obs, ep_ret, ep_len = envs.reset(), 0, 0
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            obs_processed = torch.Tensor(obs).to(device)
            actions = actor(projector(encoder.cnn_forward(obs_processed.float())))
            actions = actions.cpu().numpy().clip(envs.action_space.low, envs.action_space.high)

        next_obs, rewards, terminations, truncations = envs.step(actions)
        ep_ret += rewards
        ep_len += 1
        obs = next_obs
        if terminations or truncations:
            print(f"eval_episode={len(episodic_returns)}, episodic_return={ep_ret}")
            episodic_returns.append(ep_ret)
            obs, ep_ret, ep_len = envs.reset(), 0, 0

    actor.train()
    critic.train()
    encoder.train()
    return episodic_returns


class TwoHot:
    def __init__(self, device: torch.device, lower: float=-10, upper: float=10, num_bins: int=101):
        self.bins = torch.linspace(lower, upper, num_bins, device=device)
        self.bins = self.bins.sign() * (self.bins.abs().exp() - 1) # symexp
        self.num_bins = num_bins

    def transform(self, x: torch.Tensor):
        diff = x - self.bins.reshape(1,-1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, 1, keepdim=True)

        lower = self.bins[ind]
        upper = self.bins[(ind+1).clamp(0, self.num_bins-1)]
        weight = (x - lower)/(upper - lower)

        two_hot = torch.zeros(x.shape[0], self.num_bins, device=x.device)
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind+1).clamp(0, self.num_bins), weight)
        return two_hot

    def inverse(self, x: torch.Tensor):
        return (F.softmax(x, dim=-1) * self.bins).sum(-1, keepdim=True)

    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = F.log_softmax(pred, dim=-1)
        target = self.transform(target)
        return -(target * pred).sum(-1, keepdim=True)


class RandomShiftsAug(torch.jit.ScriptModule):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    @torch.jit.script_method
    def forward(self, x):
        n, c, h, w = x.size()

        padding = (self.pad, self.pad, self.pad, self.pad)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=torch.float32)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1) # 统一取扩大图的左上角区域

        # 每张图生成独立的、随机的(dx, dy)
        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=torch.float32)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Actor(torch.jit.ScriptModule):
    def __init__(self, env, act_dim, feature_dim, hidden_dim):
        super().__init__()

        self.l1 = nn.Linear(feature_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, act_dim)

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.apply(weight_init)

    @torch.jit.script_method
    def shallow_outputs(self, x):
        x = self.ln_activ(self.l1(x))
        return x

    @torch.jit.script_method
    def ln_activ(self, x):
        x = F.layer_norm(x, (x.shape[-1],))
        return F.relu(x)

    @torch.jit.script_method
    def forward(self, x):
        x = self.ln_activ(self.l1(x))
        x = self.ln_activ(self.l2(x))
        x = self.l3(x)
        x = torch.tanh(x)
        return x * self.action_scale + self.action_bias

    @torch.jit.script_method
    def train_forward(self, x):
        x = self.ln_activ(self.l1(x))
        x = self.ln_activ(self.l2(x))
        pre_activ = self.l3(x)
        x = torch.tanh(pre_activ)
        return x * self.action_scale + self.action_bias, pre_activ


class Critic(torch.jit.ScriptModule):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()

        self.q1_fc1 = nn.Linear(feature_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc4 = nn.Linear(hidden_dim, 1)

        self.q2_fc1 = nn.Linear(feature_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc4 = nn.Linear(hidden_dim, 1)

        self.Q1 = nn.Sequential(
            self.q1_fc1,
            nn.ELU(inplace=True),
            self.q1_fc2,
            nn.ELU(inplace=True),
            self.q1_fc3,
            nn.ELU(inplace=True),
            self.q1_fc4
        )

        self.Q2 = nn.Sequential(
            self.q2_fc1,
            nn.ELU(inplace=True),
            self.q2_fc2,
            nn.ELU(inplace=True),
            self.q2_fc3,
            nn.ELU(inplace=True),
            self.q2_fc4
        )

        self.apply(weight_init)

    @torch.jit.script_method
    def ln_elu(self, x):
        x = F.layer_norm(x, (x.shape[-1],))
        return F.elu(x)

    @torch.jit.script_method
    def forward(self, h_action):
        q1 = self.ln_elu(self.q1_fc1(h_action))
        q1 = self.ln_elu(self.q1_fc2(q1))
        q1 = self.ln_elu(self.q1_fc3(q1))
        q1 = self.q1_fc4(q1)

        q2 = self.ln_elu(self.q2_fc1(h_action))
        q2 = self.ln_elu(self.q2_fc2(q2))
        q2 = self.ln_elu(self.q2_fc3(q2))
        q2 = self.q2_fc4(q2)
        return q1, q2


class Encoder(torch.jit.ScriptModule):
    def __init__(self, in_channels, repr_dim, feature_dim, act_dim, shared_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(repr_dim, feature_dim)
        self.activ = F.elu
        self.shared_fc1 = nn.Linear(int(feature_dim + act_dim), shared_dim)
        self.shared_fc2 = nn.Linear(shared_dim, shared_dim)
        self.shared_fc3 = nn.Linear(shared_dim, 512)

        self.conv_layers = nn.Sequential(
            self.conv1,
            nn.ELU(),
            self.conv2,
            nn.ELU(),
            self.conv3,
            nn.ELU(),
            self.conv4,
            nn.ELU(),
        )
        self.apply(weight_init)

    @torch.jit.script_method
    def cnn_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0 - 0.5
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.ln_activ(self.fc1(x))
        return x

    @torch.jit.script_method
    def ln_activ(self, x):
        x = F.layer_norm(x, (x.shape[-1],))
        return self.activ(x)

    @torch.jit.script_method
    def forward(self, obs_embed: torch.Tensor, action: torch.Tensor):
        h_action = torch.cat([obs_embed, action], dim=-1)
        h_action = self.ln_activ(self.shared_fc1(h_action))
        h_action = self.ln_activ(self.shared_fc2(h_action))
        h_action = self.shared_fc3(h_action)
        return h_action


class Decoder(nn.Module):

    def __init__(self, inp_channels, latent_dim: int = 50, out_shape: tuple[int, int, int] = (3, 84, 84)) -> None:
        super().__init__()
        c, h, w = out_shape
        assert (h, w) == (84, 84)

        self.inp_channels = inp_channels
        self.fc = SN(nn.Linear(latent_dim, self.inp_channels))

        self.deconv1 = nn.ConvTranspose2d(self.inp_channels, 64, kernel_size=7, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, c, kernel_size=6, stride=2)

        self.deconvs = nn.Sequential(
            self.deconv1,
            nn.ReLU(),
            self.deconv2,
            nn.ReLU(),
            self.deconv3,
            nn.ReLU(),
            self.deconv4
        )
        self.apply(weight_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, self.inp_channels, 1, 1)
        x = self.deconvs(x)
        return x


class RewardPredictor(torch.jit.ScriptModule):
    def __init__(self, hidden_size1, hidden_size2, n_bins):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, n_bins)
        self.activ = F.elu
        self.apply(weight_init)

    @torch.jit.script_method
    def ln_activ(self, x):
        x = F.layer_norm(x, (x.shape[-1],))
        return self.activ(x)

    @torch.jit.script_method
    def forward(self, x):
        x = self.ln_activ(self.fc1(x))
        x = self.fc2(x)
        return x


class LatentProjector(torch.jit.ScriptModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        self.apply(weight_init)

    @torch.jit.script_method
    def ln_activ(self, x):
        x = F.layer_norm(x, (x.shape[-1],))
        return F.elu(x)

    @torch.jit.script_method
    def forward(self, x):
        x = self.l1(x)
        return x


def multi_step_reward(reward: torch.Tensor, not_done: torch.Tensor, discount: float, n_step: int):
    ms_reward = 0
    scale = 1
    ms_rewards = []
    scales = []
    for i in range(n_step):
        ms_reward += scale * reward[:, i]
        scale *= discount * not_done[:, i]
        ms_rewards.append(ms_reward.clone())
        scales.append(scale.clone())
    return ms_rewards, scales


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    return (F.mse_loss(x, y, reduction='none') * mask).mean()


def maybe_augment_state(state: torch.Tensor, next_state: torch.Tensor, pixel_obs: bool, use_augs: bool):
    if pixel_obs and use_augs:
        if len(state.shape) != 5: state = state.unsqueeze(1)
        batch_size, horizon, history, height, width = state.shape

        # Group states before augmenting.
        both_state = torch.concatenate([state.reshape(-1, history, height, width), next_state.reshape(-1, history, height, width)], 0)
        both_state = shift_aug(both_state)

        state, next_state = torch.chunk(both_state, 2, 0)
        state = state.reshape(batch_size, horizon, history, height, width)
        next_state = next_state.reshape(batch_size, horizon, history, height, width)

        if horizon == 1:
            state = state.squeeze(1)
            next_state = next_state.squeeze(1)
    return state, next_state


def train_encoder(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                  reward: torch.Tensor, not_done: torch.Tensor, env_terminates: bool):
    with torch.no_grad():
        encoder_target = target_projector(target_encoder.cnn_forward(
            next_state.reshape(-1, *state_shape)  # Combine batch and horizon
        )).reshape(state.shape[0], -1, 512)  # Separate batch and horizon

    pred_zs = latent_projector(encoder.cnn_forward(state[:, 0]))
    prev_not_done = 1  # In subtrajectories with termination, mask out losses after termination.
    encoder_loss = 0  # Loss is accumluated over latent_horizon.

    for i in range(args.latent_horizon):  # 这就是SPR
        pred_zs = latent_projector2(encoder(pred_zs, action[:, i]))

        # Mask out states past termination.
        dyn_loss = masked_mse(pred_zs, encoder_target[:, i], prev_not_done)

        encoder_loss = encoder_loss + 1.0 * dyn_loss
        prev_not_done = not_done[:, i].reshape(-1, 1) * prev_not_done  # Adjust termination mask.

    return encoder_loss


def shift_aug(image: torch.Tensor, pad: int=4):
    batch_size, _, height, width = image.size()
    image = F.pad(image, (pad, pad, pad, pad), 'replicate')
    eps = 1.0 / (height + 2 * pad)

    arange = torch.linspace(-1.0 + eps, 1.0 - eps, height + 2 * pad, device=image.device, dtype=torch.float)[:height]
    arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)

    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(0, 2 * pad + 1, size=(batch_size, 1, 1, 2), device=image.device, dtype=torch.float)
    shift *= 2.0 / (height + 2 * pad)
    return F.grid_sample(image, base_grid + shift, padding_mode='zeros', align_corners=False)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{args.exp_id}"

    writer_path = f"{args.log_root}/{args.domain}-{args.task}/{run_name}"
    writer = SummaryWriter(writer_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    eval_results_path = writer_path + f"/ep_returns_{args.seed}.npz"
    print(f"log_root: {args.log_root}")
    print(f"log_writer_path: {writer_path}")
    print(f"eval_results_path: {eval_results_path}")
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = f"{args.gpu_id}"
    os.environ["EGL_DEVICE_ID"] = f"{args.gpu_id}"

    from utils.dmc2gym import DMCVisual2GymWrapper, FrameStackWrapper

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # env setup
    envs = DMCVisual2GymWrapper(args.domain, args.task, seed=args.seed, action_repeat=2,
                                     symbolic=False, bit_depth=args.bit_depth, image_size=(84, 84))
    envs = FrameStackWrapper(envs, num_stack=3)
    observation_space_v2 = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
    envs.action_space.seed(args.seed)
    act_dim = int(np.array(envs.action_space.shape).prod())
    act_min = torch.from_numpy(envs.action_space.low).to(device)
    act_max = torch.from_numpy(envs.action_space.high).to(device)

    aug = RandomShiftsAug(pad=4)
    encoder = Encoder(in_channels=9, feature_dim=512, repr_dim=1568, shared_dim=580,
                          act_dim=act_dim).to(device)
    decoder = Decoder(inp_channels=128, out_shape=(3, 84, 84), latent_dim=512).to(device)
    latent_projector = LatentProjector(in_dim=512, out_dim=512).to(device)
    latent_projector2 = LatentProjector(in_dim=512, out_dim=512).to(device)
    target_projector = copy.deepcopy(latent_projector)
    target_encoder = copy.deepcopy(encoder)

    actor = Actor(env=envs, act_dim=act_dim, feature_dim=512, hidden_dim=512).to(device)
    critic = Critic(hidden_dim=512, feature_dim=512).to(device)
    target_critic = copy.deepcopy(critic)
    target_actor = copy.deepcopy(actor)

    # actor = torch.compile(actor)
    # critic = torch.compile(critic)
    # target_critic = torch.compile(target_critic)
    # target_actor = torch.compile(target_actor)
    # decoder = torch.compile(decoder)
    # encoder = torch.compile(encoder)
    # target_encoder = torch.compile(target_encoder)

    target_critic.load_state_dict(critic.state_dict())
    target_actor.load_state_dict(actor.state_dict())
    target_encoder.load_state_dict(encoder.state_dict())
    target_projector.load_state_dict(latent_projector.state_dict())

    two_hot = TwoHot(device=device, num_bins=65)
    reward_predictor = RewardPredictor(hidden_size1=512, hidden_size2=512, n_bins=65).to(device)

    enc_optimizer = optim.AdamW(list(encoder.parameters()) + list(latent_projector.parameters()) +
                                list(latent_projector2.parameters())
                                , lr=args.enc_lr, weight_decay=1e-4)

    dec_optimizer = optim.AdamW(decoder.parameters(), lr=args.dec_lr, weight_decay=1e-4)
    q_optimizer = optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=1e-4)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.actor_lr, weight_decay=1e-4)
    rew_optimizer = optim.AdamW(reward_predictor.parameters(), lr=args.critic_lr, weight_decay=1e-4)
    # 用于gard_clip
    critic_param_list = list(critic.parameters())
    enc_param_list = list(encoder.parameters()) + list(latent_projector.parameters()) + list(latent_projector2.parameters())
    dec_param_list = list(decoder.parameters())
    rew_param_list = list(reward_predictor.parameters())
    projector_param_list = list(latent_projector.parameters())

    state_shape = (9, 84, 84)
    rb = PriorityReplayBuffer(
        pixel_obs=True,
        max_size=args.buffer_size,
        obs_shape=(3, 84, 84),
        history=3,
        horizon=max(args.n_step_return, args.latent_horizon),
        action_dim=act_dim,
        prioritized=args.prioritized,
        device=device,
        initial_priority=1.0,
        batch_size=args.batch_size,
    )

    # Start the game
    obs, ep_ret, ep_len = envs.reset(), 0, 0
    cur_max_avg_return = -np.inf
    q_val_discount = args.gamma ** args.n_step_return
    training_steps = 0
    min_priority = 1.0
    alpha = 0.4
    # the buffer used to record the average evaluation returns for ease of accessing results without tensorboard
    eval_buffer = {"steps": [],
                   "ep_return": []
                   }
    for global_step in range(0, args.total_timesteps + 1):
        if global_step <= args.learning_starts:
            actions = np.array([envs.action_space.sample() for _ in range(1)])
        else:
            with torch.no_grad():
                obs_processed = torch.Tensor(obs).to(device)
                actions = actor(latent_projector(encoder.cnn_forward(obs_processed.float())))
                # add noise
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.action_space.low, envs.action_space.high)

        next_obs, rewards, terminations, truncations = envs.step(actions)
        ep_ret += rewards
        ep_len += 1

        rb.add(obs.squeeze(0), actions[0], next_obs.squeeze(0), rewards, terminations, truncations)
        obs = next_obs

        if terminations or truncations:
            print(
                f"global_step={global_step} ({(global_step / args.total_timesteps * 100):.4g}%), episodic_return={ep_ret}")
            writer.add_scalar("charts/episodic_return", ep_ret, global_step)
            writer.add_scalar("charts/episodic_length", ep_len, global_step)

            obs, ep_ret, ep_len = envs.reset(), 0, 0

        # ALGO LOGIC: training.
        if global_step >= args.learning_starts:
            training_steps += 1
            #########################
            # Update Target Nets
            #########################
            if (training_steps - 1) % args.target_update_freq == 0:
                target_critic.load_state_dict(critic.state_dict())
                target_actor.load_state_dict(actor.state_dict())
                target_encoder.load_state_dict(encoder.state_dict())
                target_projector.load_state_dict(latent_projector.state_dict())

            #########################
            # Update Critic
            #########################
            state, actions, next_state, reward, not_done = rb.sample(args.n_step_return, include_intermediate=True)
            action = actions[:, 0]
            data_observations = state[:, 0].float()
            data_next_observations = next_state[:, -1].float()
            data_target_observations = next_state[:, 0][:, -3:].float()
            data_observations_stack = torch.cat([data_observations, data_target_observations], dim=1)
            data_observations_stack = aug(data_observations_stack)
            data_next_observations = aug(data_next_observations)
            data_observations = data_observations_stack[:, 0:9, :]
            target_observations = data_observations_stack[:, -3:, :] / 255. - 0.5
            obs_targets = data_observations / 255. - 0.5

            cum_rewards, term_discounts = multi_step_reward(reward, not_done, args.gamma, args.n_step_return)
            cum_reward = cum_rewards[args.n_step_return - 1]
            term_discount = term_discounts[args.n_step_return - 1]

            obs_embeddings = encoder.cnn_forward(data_observations)
            with torch.no_grad():
                next_obs_embeddings = target_projector(target_encoder.cnn_forward(data_next_observations))

            with torch.no_grad():
                clipped_noise = (torch.randn_like(action, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (
                            target_actor(next_obs_embeddings) + clipped_noise).clamp(
                    act_min, act_max
                )
                qf1_next_target, qf2_next_target = target_critic(target_encoder(next_obs_embeddings, next_state_actions))

                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = cum_reward + term_discount * min_qf_next_target

            latent = encoder(obs_embeddings, action)
            latent_obs_embeddings = latent_projector(obs_embeddings)
            latent_h_a = encoder(latent_obs_embeddings, action)
            qf1_a_values, qf2_a_values = critic(latent_h_a)
            qfs_a_values = torch.cat([qf1_a_values, qf2_a_values], dim=1)
            qf_loss = F.smooth_l1_loss(qfs_a_values, next_q_value.expand(-1, 2))

            #########################
            # Auxiliary Tasks
            #########################
            if args.aux_task:
                state, actions, next_state, reward, not_done = rb.sample(args.latent_horizon, include_intermediate=True)
                state, next_state = maybe_augment_state(state, next_state, True, True)
                self_pred_loss = train_encoder(state, actions, next_state, reward, not_done, rb.env_terminates)

                pred_obs_nxt = decoder(latent)
                pred_rew_logits = reward_predictor(latent)
                aux_loss_batch = torch.sum((pred_obs_nxt - target_observations) ** 2, dim=(1, 2, 3))
                aux_loss = torch.mean(aux_loss_batch)

                pred_rew_loss = two_hot.cross_entropy_loss(pred_rew_logits, cum_reward).squeeze(1)
                pred_rew_loss = pred_rew_loss.mean()
                total_aux_loss = args.obs_coef * aux_loss + args.rew_coef * pred_rew_loss + \
                                 args.latent_coef * self_pred_loss
                total_loss = qf_loss + total_aux_loss
            else:
                total_loss = qf_loss

            # optimize the model
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            q_optimizer.zero_grad()
            rew_optimizer.zero_grad()
            total_loss.backward()

            nn.utils.clip_grad_norm_(critic_param_list + enc_param_list + dec_param_list + rew_param_list, args.grad_clip_norm, norm_type=2)
            q_optimizer.step()
            dec_optimizer.step()
            enc_optimizer.step()
            rew_optimizer.step()
            #########################
            # Update Actor
            #########################
            if global_step % args.policy_frequency == 0:
                actor_outs, pre_activ = actor.train_forward(latent_obs_embeddings.detach())
                Q1, Q2 = critic(encoder(latent_obs_embeddings.detach(), actor_outs))
                qfs_vals = torch.cat([Q1, Q2], dim=1)
                actor_loss = -qfs_vals.mean() + 1e-5 * pre_activ.pow(2).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()

                actor_optimizer.step()

            if args.prioritized:
                priority = (qfs_a_values - next_q_value.expand(-1, 2)).abs().max(1).values
                priority = priority.clamp(min=min_priority).pow(alpha)
                rb.update_priority(priority)

        if (global_step > args.learning_starts) & (global_step % args.eval_per_steps == 0):
            episodic_returns = dmc_evaluate(
                domain=args.domain,
                task=args.task,
                eval_episodes=10,
                actor=actor,
                encoder=encoder,
                critic=critic,
                device=device,
                projector=latent_projector,
                bit_depth=args.bit_depth
            )
            episodic_return_mean = np.hstack(episodic_returns).mean()
            episodic_return_std = np.hstack(episodic_returns).std()
            writer.add_scalar("eval/episodic_return_mean", episodic_return_mean, global_step)
            writer.add_scalar("eval/episodic_return_std", episodic_return_std, global_step)
            eval_buffer['steps'].append(global_step)
            eval_buffer['ep_return'].append(episodic_return_mean)
            print(f"cur_avg_return: {episodic_return_mean}")

    envs.close()
    writer.close()

for k in eval_buffer.keys():
    eval_buffer[k] = np.array(eval_buffer[k])
