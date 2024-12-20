import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DQN, PPO, SAC

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

class LinearLR:
    def __init__(self, initial_value, final_value, schedule_timesteps):
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_timesteps = schedule_timesteps

    def __call__(self, progress):
        # progress: The current training progress (0.0 to 1.0)
        lr = self.initial_value + progress * (self.final_value - self.initial_value)
        return lr

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        # 定義 CNN 層，處理 16x4x4 的輸入
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=1),  # 32x3x3
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=1, padding=1),  # 64x2x2
            nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),  # 128x1x1
            # nn.ReLU(),
            nn.Flatten()
        )
        
        # 計算 CNN 輸出維度，展平後應該是 128
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, 16, 4, 4)).shape[1]  # 這裡 shape[1] 是展平後的向量大小
        
        # 全連接層將輸出調整為指定的 features_dim
        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations):
        # return self.cnn(observations)
        return self.linear(self.cnn(observations))

class DoubleCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super(DoubleCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # Should be 16 (input channels)
        
        # First branch of convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=(1, 2), stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(4, 3), stride=1),
            nn.ReLU()
        )
        
        # Second branch of convolutions
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=(3, 4), stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(2, 4), stride=1),
            nn.ReLU()
        )
        
        # Linear layers (fully connected)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 128, features_dim),  # Adjust based on final flattened size
            nn.ReLU(),
        )

    def forward(self, observations):
        # Apply the two convolutional branches
        conv1_out = self.conv1(observations)
        conv2_out = self.conv2(observations)
        
        # Concatenate the outputs
        concat_out = torch.cat((conv1_out, conv2_out), dim=1)  # Concatenate along the channel axis
        
        # Pass through fully connected layers
        return self.fc(concat_out)


# Set hyper params (configurations) for training
num_train_envs = 10
# Custom policy kwargs with a larger, deeper MLP
policy_kwargs = dict(
    features_extractor_class=DoubleCNNFeatureExtractor,  # 自定義特徵萃取器
    features_extractor_kwargs=dict(features_dim=256),  # 特徵輸出維度
    net_arch=[64]
)
epoch_num = 500
timesteps_per_epoch = 20000
n_steps = 15
# lr_schedule = LinearLR(initial_value=1e-4, final_value=1e-5, schedule_timesteps=epoch_num * timesteps_per_epoch)
my_config = {
    "run_id": f"A2C_{num_train_envs}_{epoch_num}_{timesteps_per_epoch}_newframe_DCNN_64_p(-100)_cnt3_weight_high_lr",

    "algorithm": A2C,
    "policy_network": "MlpPolicy",
    "save_path": "models/sample_model",

    "epoch_num": epoch_num,
    "timesteps_per_epoch": timesteps_per_epoch,
    "eval_episode_num": 10,
    "learning_rate": 1e-4,
    "n_steps": n_steps,
    # "vf_coef": 0.25,
    # "gamma": 0.95,
}

def make_env():
    env = gym.make('2048-v0')
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()

        # Interact with env using old Gym API
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        avg_highest += info[0]['highest']
        avg_score   += info[0]['score']

    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num
        
    return avg_score, avg_highest

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    for epoch in range(config["epoch_num"]):

        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=WandbCallback(
                gradient_save_freq=100,
                verbose=1,
            ),
        )

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_score, avg_highest = eval(eval_env, model, config["eval_episode_num"])
        
        print("Avg_score:  ", avg_score)
        print("Avg_highest:", avg_highest)
        print()
        wandb.log(
            {"avg_highest": avg_highest,
             "avg_score": avg_score}
        )
        
        ### Save best model
        if current_best < avg_highest:
            print("Saving Model")
            current_best = avg_highest
            save_path = config["save_path"]
            # model.save(f"{save_path}/{epoch}")
            model.save(f"{save_path}/1")
            # model.save(f"{save_path}/{str(my_config)}")

        # print("---------------")


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable `wandb` logging)
    run = wandb.init(
        project="assignment_3",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=my_config["run_id"],
    )

    # Create training environment 
    train_env = DummyVecEnv([make_env for _ in range(num_train_envs)])

    # Create evaluation environment 
    eval_env = DummyVecEnv([make_env])  

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"], 
        train_env, 
        verbose=1,
        tensorboard_log=my_config["run_id"],
        learning_rate=my_config["learning_rate"],
        n_steps = my_config["n_steps"],
        # vf_coef = my_config["vf_coef"],
        # gamma = my_config["gamma"],
        policy_kwargs=policy_kwargs  # Include custom network architecture
    )

    # load pretained model
    # model = my_config["algorithm"].load(f"{my_config['save_path']}/A2C_10_100_20000_newframe_CNN_128_128_128_128_p(-100)_cnt3_weight_high_281", env=train_env)
    # model.n_steps = my_config["n_steps"]
    # model.vf_coef = my_config["vf_coef"]
    # model.gamma = my_config["gamma"]
    # model.learning_rate = my_config["learning_rate"]
    # print(f"Model loaded. Current learning rate: {model.learning_rate}")

    train(eval_env, model, my_config)
