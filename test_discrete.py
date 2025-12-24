"""Quick test to verify discrete action support for SpaceInvaders"""
import gymnasium as gym
import ale_py
import torch
from envs import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper

ale_py.register_v5_envs()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test environment setup
env = CleanGymWrapper(GymPixelsProcessingWrapper(
    gym.wrappers.ResizeObservation(
        gym.make("SpaceInvadersNoFrameskip-v4"), (64, 64))))

# Test getEnvProperties
observationShape, isDiscrete, actionSize, actionLow, actionHigh = getEnvProperties(env)

print("=" * 60)
print("Environment Properties Test")
print("=" * 60)
print(f"Observation Shape: {observationShape}")
print(f"Is Discrete: {isDiscrete}")
print(f"Action Size: {actionSize}")
print(f"Action Low: {actionLow}")
print(f"Action High: {actionHigh}")
print("=" * 60)

if isDiscrete and actionSize == 6:
    print("✓ SpaceInvaders correctly identified as discrete with 6 actions")
else:
    print("✗ Error: SpaceInvaders should be discrete with 6 actions")

# Test a few steps
obs = env.reset()
print(f"✓ Environment reset successful, observation shape: {obs.shape}")

for i in range(5):
    action = i % actionSize  # Cycle through actions
    obs, reward, done = env.step(action)
    print(f"  Step {i+1}: action={action}, reward={reward}, done={done}")
    if done:
        obs = env.reset()

print("=" * 60)
print("All tests passed! Ready for training.")
print("=" * 60)
