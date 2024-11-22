import gymnasium as gym
from stable_baselines3 import DQN
import ale_py


def main():
    gym.register_envs(ale_py)
    # Create environment with rendering enabled
    env = gym.make("ALE/Breakout-v5", render_mode="human")

    # Load the trained model
    try:
        model = DQN.load("models/policy.zip")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have run train.py first and the model file exists.")
        return

    # Run several episodes
    n_episodes = 5
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            # Use the model to predict the best action (greedy policy)
            action, _ = model.predict(obs, deterministic=True)

            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {episode + 1} - Total Reward: {total_reward} - Steps: {steps}")

    env.close()


if __name__ == "__main__":
    main()