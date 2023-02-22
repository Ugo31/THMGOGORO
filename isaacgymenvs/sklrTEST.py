import isaacgym

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview4
from skrl.utils import set_seed


# set the seed for reproducibility
set_seed(42)

class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(256, 128),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(128, 32),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(32, self.num_actions),
                                 )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}
    
class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 256),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(256, 128),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(128, 32),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}
    

# Load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="Gogoro")   # preview 3 and 4 use the same loader
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)


models_sac = {}
models_sac["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
models_sac["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_2"] = Critic(env.observation_space, env.action_space, device)
for model in models_sac.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    

cfg_sac = SAC_DEFAULT_CONFIG.copy()
cfg_sac["gradient_steps"] = 1
cfg_sac["batch_size"] = 64
cfg_sac["random_timesteps"] = 0
cfg_sac["discount_factor"] = 0.99

cfg_sac["actor_learning_rate"] = 1e-7
cfg_sac["critic_learning_rate"] = 1e-7

#cfg_sac["learning_rate_scheduler"] = KLAdaptiveRL
# cfg_sac["learn_entropy"] = True
# cfg_sac["entropy_learning_rate"] = 1e-4
# cfg_sac["initial_entropy_value"] = 0.3


# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_sac["experiment"]["write_interval"] = 25
cfg_sac["experiment"]["checkpoint_interval"] = 1000




agent_sac = SAC(models=models_sac,
                memory=memory,
                cfg=cfg_sac,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000000000000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_sac)

# start training
trainer.train()