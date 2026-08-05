import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
import torch.nn.functional as F
import torch.optim as optim
import tyro

from torch.utils.tensorboard import SummaryWriter
from utils.buffer import PriorityReplayBuffer
import copy
import torchvision.transforms as T
from utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    gpu_id: int = 0
    seed: int = 123
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    bit_depth: int = 5
    env_id: str = "AssaultNoFrameskip-v4"
    total_timesteps: int = 100000
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    enc_lr: float = 3e-4
    dec_lr: float = 3e-4
    buffer_size: int = int(100000)
    gamma: float = 0.99
    num_envs: int = 1
    batch_size: int = 256
    policy_noise: float = 0.2
    exploration_noise: float = 0.2
    learning_starts: int = 2000
    eval_per_steps: int = 5000
    policy_frequency: int = 1
    target_update_freq: int = 250
    noise_clip: float = 0.3
    n_bins: int = 51
    grad_clip_norm: float = 20
    latent_horizon: int = 5
    n_step_return: int = 3
    aux_task: bool = True
    obs_coef: float = 1.0
    rew_coef: float = 1.0
    latent_coef: float = 5.0
    prioritized: bool = False
    intensity_aug: bool = False
    updates_per_step: int = 2
    log_root: str = "/root"
    exp_id: str = ""


def atari_evaluate(
    make_env,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    encoder,
    actor,
    projector,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    actor.eval()
    encoder.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        obs_processed = torch.Tensor(obs).to(device)
        actions = actor(projector(encoder.cnn_forward(obs_processed.float())))
        actions = actions.argmax(dim=-1).cpu().numpy()

        next_obs, _, _, _, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    # print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    episodic_returns.append(info["episode"]["r"])
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")

        obs = next_obs
    actor.train()
    encoder.train()
    return episodic_returns


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=108000)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, max_episode_steps=108000)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


def make_eval_env(env_id, seed, idx, capture_video, run_name):
    """
    评估的时候不再进行reward clipping
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=108000)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, max_episode_steps=108000)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


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


class RewardPredictor(torch.jit.ScriptModule):
    def __init__(self, hidden_size1, hidden_size2, n_bins):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, n_bins)
        self.activ = F.elu
        # ----------------- 权重初始化 -----------------
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


class Actor(torch.jit.ScriptModule):
    def __init__(self, act_dim, feature_dim, hidden_dim):
        super().__init__()

        self.l1 = nn.Linear(feature_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, act_dim)
        # self.activ = partial(F.gumbel_softmax, tau=10.0)

        self.apply(weight_init)

    @torch.jit.script_method
    def ln_activ(self, x):
        x = F.layer_norm(x, (x.shape[-1],))
        return F.relu(x)

    @torch.jit.script_method
    def forward(self, x):
        x = self.ln_activ(self.l1(x))
        x = self.ln_activ(self.l2(x))
        x = self.l3(x)
        x = F.gumbel_softmax(x, tau=10.0)
        return x

    @torch.jit.script_method
    def train_forward(self, x):
        x = self.ln_activ(self.l1(x))
        x = self.ln_activ(self.l2(x))
        pre_activ = self.l3(x)
        x = F.gumbel_softmax(pre_activ, tau=10.0)
        return x, pre_activ


class Critic(torch.jit.ScriptModule):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()

        self.q1_fc3 = nn.Linear(feature_dim, hidden_dim)
        self.q1_fc3_ = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc4_ = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc4 = nn.Linear(hidden_dim, 1)

        self.q2_fc3 = nn.Linear(feature_dim, hidden_dim)
        self.q2_fc3_ = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc4_ = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc4 = nn.Linear(hidden_dim, 1)

        self.Q1 = nn.Sequential(
            self.q1_fc3,
            nn.ELU(inplace=True),
            self.q1_fc3_,
            nn.ELU(inplace=True),
            self.q1_fc4_,
            nn.ELU(inplace=True),
            self.q1_fc4
        )

        self.Q2 = nn.Sequential(
            self.q2_fc3,
            nn.ELU(inplace=True),
            self.q2_fc3_,
            nn.ELU(inplace=True),
            self.q2_fc4_,
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
        q1 = self.ln_elu(self.q1_fc3(h_action))
        q1 = self.ln_elu(self.q1_fc3_(q1))
        q1 = self.ln_elu(self.q1_fc4_(q1))
        q1 = self.q1_fc4(q1)

        q2 = self.ln_elu(self.q2_fc3(h_action))
        q2 = self.ln_elu(self.q2_fc3_(q2))
        q2 = self.ln_elu(self.q2_fc4_(q2))
        q2 = self.q2_fc4(q2)
        return q1, q2


class Encoder(torch.jit.ScriptModule):

    def __init__(self, in_channels, repr_dim, feature_dim, act_dim, shared_dim):
        """
        参数
        ----
        in_channels : 输入图像通道数 (默认 3)
        img_size    : (H, W)。若传入，则在 __init__ 阶段即可推算 fc 尺寸；
                      若为 None，则在首次 forward 时 “懒初始化” fc 和 LayerNorm
        out_dim     : 线性层输出维度 (默认 50)
        """
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

        # ----------------- 卷积堆叠 -----------------
        self.conv_layers = nn.Sequential(
            self.conv1,           # conv1
            nn.ELU(),
            self.conv2,           # conv2
            nn.ELU(),
            self.conv3,           # conv3
            nn.ELU(),
            self.conv4,           # conv4
            nn.ELU(),
        )

        self.my_modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.shared_fc1,
                           self.shared_fc2, self.shared_fc3]
        # ----------------- 权重初始化 -----------------
        self.apply(weight_init)

    # --------------------------------------------------
    # 前向传播
    # --------------------------------------------------
    @torch.jit.script_method
    def cnn_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0 - 0.5
        x = self.conv_layers(x)                           # (N, 32, H', W')
        x = torch.flatten(x, 1)                           # (N, *)
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


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    # @torch.jit.script_method
    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)  # torch.Size([192, 1, 1, 1])的正态分布随机变量
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise  # (T, B)维度上随机独立


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


def realign(x):
    return F.one_hot(x.argmax(1), x.shape[1]).float()


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
        # reward_loss = (two_hot.cross_entropy_loss(pred_r, reward[:, i]) * prev_not_done).mean()
        # done_loss = masked_mse(pred_d, 1. - not_done[:, i].reshape(-1, 1), prev_not_done) if env_terminates else 0

        encoder_loss = encoder_loss + 1.0 * dyn_loss
        prev_not_done = not_done[:, i].reshape(-1, 1) * prev_not_done  # Adjust termination mask.

    # encoder_optimizer.zero_grad()
    # encoder_loss.backward()
    # encoder_optimizer.step()
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
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.exp_id}"
    writer_path = f"{args.log_root}/{args.env_id}/{run_name}"
    writer = SummaryWriter(writer_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    metric_path = writer_path + f"/ep_return_{args.seed}.npz"
    print(f"log_root: {args.log_root}")
    print(f"writer_path: {writer_path}")
    print(f"metric_path: {metric_path}")
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    print(f"gpu: {args.gpu_id}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    act_dim = envs.single_action_space.n
    # aug = RandomShiftsAug(pad=4)
    encoder = Encoder(in_channels=4, feature_dim=512, repr_dim=1568, shared_dim=580,
                          act_dim=act_dim).to(device)
    decoder = Decoder(inp_channels=128, out_shape=(1, 84, 84), latent_dim=512).to(device)
    latent_projector = LatentProjector(in_dim=512, out_dim=512).to(device)
    latent_projector2 = LatentProjector(in_dim=512, out_dim=512).to(device)
    target_projector = copy.deepcopy(latent_projector)
    target_encoder = copy.deepcopy(encoder)

    actor = Actor(act_dim=int(act_dim), feature_dim=int(512), hidden_dim=int(512)).to(device)
    critic = Critic(hidden_dim=512, feature_dim=512).to(device)
    target_critic = copy.deepcopy(critic)
    target_actor = copy.deepcopy(actor)

    reward_done_predictor = RewardPredictor(hidden_size1=512, hidden_size2=512, n_bins=65).to(device)
    two_hot = TwoHot(device=device, num_bins=65)

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

    enc_optimizer = optim.AdamW(list(encoder.parameters()) + list(latent_projector.parameters()) +
                                list(latent_projector2.parameters())
                                , lr=args.enc_lr, weight_decay=1e-4)
    dec_optimizer = optim.AdamW(decoder.parameters(), lr=args.dec_lr, weight_decay=1e-4)
    q_optimizer = optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=1e-4)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.actor_lr, weight_decay=1e-4)
    rew_optimizer = optim.AdamW(reward_done_predictor.parameters(), lr=args.dec_lr, weight_decay=1e-4)
    # 用于gard_clip
    critic_param_list = list(critic.parameters())
    enc_param_list = list(encoder.parameters()) + list(latent_projector.parameters()) + list(latent_projector2.parameters())
    dec_param_list = list(decoder.parameters())
    rew_param_list = list(reward_done_predictor.parameters())
    projector_param_list = list(latent_projector.parameters())

    state_shape = (4, 84, 84)
    rb = PriorityReplayBuffer(
        pixel_obs=True,
        max_size=args.buffer_size,
        obs_shape=(1, 84, 84),
        history=4,
        horizon=max(args.n_step_return, args.latent_horizon),
        action_dim=act_dim,
        prioritized=args.prioritized,
        device=device,
        initial_priority=1.0,
        batch_size=args.batch_size
    )
    # 图像增强
    aug = nn.Sequential(nn.ReplicationPad2d(4), T.RandomCrop((84, 84)), Intensity(scale=0.05)) if args.intensity_aug else RandomShiftsAug(4)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    cur_max_avg_return = -np.inf
    training_steps = 0
    min_priority = 1.0
    alpha = 0.4
    eval_buffer = {"steps": [],
                   "ep_return": []
                   }

    print('='*15)
    print(f"Date: {time.asctime()}")
    print('=' * 15)
    for global_step in range(0, args.total_timesteps + 1):
        # ALGO LOGIC: put action logic here
        if global_step <= args.learning_starts:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                obs_processed = torch.Tensor(obs).to(device)
                actions = actor(latent_projector(encoder.cnn_forward(obs_processed.float())))
                # add noise
                actions += torch.randn_like(actions) * args.exploration_noise
                actions = actions.argmax(dim=-1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        # 当terminations=True, next_obs是下个ep的obs0，真正的final_obs在infos里
        rb.add(obs.squeeze(0), actions[0], real_next_obs.squeeze(0), rewards[0], terminations[0], truncations[0])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            for update in range(args.updates_per_step):
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
                data_target_observations = next_state[:, 0][:, [-1]].float()
                data_observations_stack = torch.cat([data_observations, data_target_observations], dim=1)
                data_observations_stack = aug(data_observations_stack)
                data_next_observations = aug(data_next_observations)
                data_observations = data_observations_stack[:, 0:4, :]
                target_observations = data_observations_stack[:, [-1], :] / 255. - 0.5

                cum_rewards, term_discounts = multi_step_reward(reward, not_done, args.gamma, args.n_step_return)
                cum_reward = cum_rewards[args.n_step_return-1]
                term_discount = term_discounts[args.n_step_return-1]

                obs_embeddings = encoder.cnn_forward(data_observations)
                with torch.no_grad():
                    next_obs_embeddings = target_projector(target_encoder.cnn_forward(data_next_observations))

                with torch.no_grad():
                    clipped_noise = (torch.randn_like(action, device=device) * args.policy_noise).clamp(
                        -args.noise_clip, args.noise_clip)

                    next_state_actions = realign((target_actor(next_obs_embeddings) + clipped_noise))

                    qf1_next_target, qf2_next_target = target_critic(target_encoder(next_obs_embeddings, next_state_actions))

                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value = cum_reward + term_discount * min_qf_next_target

                h_action = encoder(obs_embeddings, action)
                latent_obs_embeddings = latent_projector(obs_embeddings)
                latent_h_a = encoder(latent_obs_embeddings, action)
                qf1_a_values, qf2_a_values = critic(latent_h_a)
                qfs_a_values = torch.cat([qf1_a_values, qf2_a_values], dim=1)
                qf_loss = F.smooth_l1_loss(qfs_a_values, next_q_value.expand(-1, 2))
                #########################
                # Aux Task
                #########################
                if args.aux_task:
                    pred_obs_nxt = decoder(h_action)
                    pred_rew = reward_done_predictor(h_action)
                    pred_rew_loss = two_hot.cross_entropy_loss(pred_rew, cum_reward)

                    state, actions, next_state, reward, not_done = rb.sample(args.latent_horizon, include_intermediate=True)
                    state, next_state = maybe_augment_state(state, next_state, True, True)
                    self_pred_loss = train_encoder(state, actions, next_state, reward, not_done, rb.env_terminates)

                    aux_loss_batch = torch.sum((pred_obs_nxt - target_observations) ** 2, dim=(1, 2, 3))
                    aux_loss1 = torch.mean(aux_loss_batch)
                    aux_loss2 = torch.mean(pred_rew_loss)

                    total_aux_loss = args.obs_coef * aux_loss1 + args.rew_coef * aux_loss2 + args.latent_coef * self_pred_loss
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
                if training_steps % args.policy_frequency == 0:
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
            if (global_step % 100000 == 0):
                eval_episodes = 100
            else:
                eval_episodes = 10

            episodic_returns = atari_evaluate(
                make_env=make_eval_env, env_id=args.env_id,
                eval_episodes=eval_episodes, run_name=f"{run_name}-eval", device=device,
                encoder=encoder, actor=actor, projector=latent_projector
            )
            episodic_return_mean = np.hstack(episodic_returns).mean()
            episodic_return_std = np.hstack(episodic_returns).std()
            writer.add_scalar("eval/episodic_return_mean", episodic_return_mean, global_step)
            writer.add_scalar("eval/episodic_return_std", episodic_return_std, global_step)
            eval_buffer['steps'].append(global_step)
            eval_buffer['ep_return'].append(episodic_return_mean)
            if (global_step % 100000 == 0):
                writer.add_scalar("eval/episodic_return_mean2", episodic_return_mean, global_step)
                writer.add_scalar("eval/episodic_return_std2", episodic_return_std, global_step)
            print(f"cur_avg_return: {episodic_return_mean}")
            if episodic_return_mean > cur_max_avg_return:
                cur_max_avg_return = episodic_return_mean
            print(f"cur_max_avg_return: {cur_max_avg_return}")
            writer.add_scalar("eval/cur_max_avg_return", cur_max_avg_return, global_step)

    envs.close()
    writer.close()

for k in eval_buffer.keys():
    eval_buffer[k] = np.array(eval_buffer[k])


np.savez(metric_path, **eval_buffer)
print("take time: {}".format(time.time() - start_time))
