import pandas as pd
import numpy as np
import time
import re

from Environment.gens.TA_Gen import TAStreamer
from Environment.envs.indicator_1 import Indicator_1
from Agent.duelling_dqn import DDDQNAgent

def initialize_environment(filename, train_test_split, trading_fee, time_fee):
    """Initialize the trading environment."""
    generator = TAStreamer(filename=filename, mode='train', split=train_test_split)
    episode_length = round(int(len(pd.read_csv(filename)) * train_test_split), -1)
    environment = Indicator_1(data_generator=generator, trading_fee=trading_fee, time_fee=time_fee, episode_length=episode_length)
    return environment, episode_length

def initialize_agent(environment, episodes, memory_size, gamma, epsilon_min, batch_size, train_interval, learning_rate, episode_length, filename, train_test):
    """Initialize the trading agent."""
    action_size = len(Indicator_1._actions)
    state = environment.reset()
    state_size = len(state)
    try:
        symbol = re.findall(r'Data\\([^_]+)', filename)[0]
    except:
        symbol = ""
    agent = DDDQNAgent(state_size=state_size, action_size=action_size, memory_size=memory_size, episodes=episodes, episode_length=episode_length, train_interval=train_interval, gamma=gamma, learning_rate=learning_rate, batch_size=batch_size, epsilon_min=epsilon_min, train_test=train_test, symbol=symbol)
    return agent


def warm_up_agent(agent, environment, memory_size, display, start_time):
    """Warm up the agent with initial observations."""
    for _ in range(memory_size):
        action = agent.act(environment.reset())
        next_state, reward, done, _ = environment.step(action)
        agent.observe(environment.reset(), action, reward, next_state, done, warming_up=True)
    if display:
        print('Completed memory allocation:', time.time() - start_time)

def train_agent(agent, environment, episodes, episode_length, display):
    """Train the agent over a specified number of episodes."""
    metrics = {'loss': [], 'reward': [], 'epsilon': []}
    for ep in range(episodes):
        start_time = time.time()
        state, total_reward = environment.reset(), 0
        loss_list = []

        for _ in range(episode_length):
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            loss = agent.observe(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if loss is not None:
                loss_list.append(loss)  # 直接添加损失值

        # 计算平均损失
        avg_loss = np.mean(loss_list) if loss_list else 0
        metrics['loss'].append(avg_loss)
        metrics['reward'].append(total_reward)
        metrics['epsilon'].append(agent.epsilon)

        if display:
            print(f"Episode: {ep}, Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.2f}, Time: {time.time() - start_time:.2f}")

    return pd.DataFrame(metrics)


def update_metrics(metrics, loss_temp, val_loss_temp, rew, epsilon):
    """Update training metrics after each episode."""
    metrics['loss'].append(np.mean(loss_temp))
    metrics['val_loss'].append(np.mean(val_loss_temp))
    metrics['reward'].append(rew)
    metrics['epsilon'].append(round(epsilon, 2))

def print_episode_summary(ep, rew, epsilon, loss, start_time, loss_temp, val_loss_temp):
    """Print a summary of the episode."""
    print("Ep: {} | Reward: {:.2f} | Epsilon: {:.2f} | Loss: {:.4f} | Runtime: {:.2f}".format(ep, rew, epsilon, loss.history["loss"][0], time.time() - start_time))
    print("Avg Loss: {:.2f} | Avg Val Loss: {:.2f}".format(np.mean(loss_temp), np.mean(val_loss_temp)))

def test_agent(agent, environment, render_show):
    """Test the agent and collect trading data."""
    done = False
    state = environment.reset()
    trade_data = {'rewards': [], 'trades': [], 'q_values': [], 'states': [], 'actions': []}

    while not done:
        action, q_values = agent.act(state, test=True)
        state, reward, done, info = environment.step(action)
        update_trade_data(trade_data, reward, environment, render_show, q_values, state, action)

        if 'status' in info and info['status'] == 'Closed plot':
            done = True
        else:
            if render_show:
                environment.render()

    print('Total Reward: {:.2f}'.format(sum(trade_data['rewards'])))
    return trade_data

def update_trade_data(trade_data, reward, environment, render_show, q_values, state, action):
    """Update trade data during testing."""
    trade_data['rewards'].append(reward)
    calc_returns = environment.return_calc(render_show)
    if calc_returns:
        trade_data['trades'].append(calc_returns)
    trade_data['q_values'].append(q_values)
    trade_data['states'].append(state)
    trade_data['actions'].append(action)

def save_results(metrics_df, trades_df, action_policy_df, save_results):
    """Save trading and performance metrics to files."""
    if save_results:
        metrics_df.to_csv(r'./Results/perf_metrics.csv')
        pd.DataFrame(trades_df).to_csv(r'./Results/trade_list.csv')
        pd.DataFrame({'q_values': action_policy_df['q_values'], 'state': action_policy_df['states'], 'action': action_policy_df['actions']}).to_pickle(r'./Results/action_policy.pkl')

def World(filename=None, train_test='train', episodes=10, train_test_split=0.75, trading_fee=.0001, 
          time_fee=.001, memory_size=3000, gamma=0.96, epsilon_min=0.01, batch_size=64, train_interval=10, 
          learning_rate=0.001, render_show=True, display=True, save_results=False):
    start_time = time.time()
    environment, episode_length = initialize_environment(filename, train_test_split, trading_fee, time_fee)
    agent = initialize_agent(environment, episodes, memory_size, gamma, epsilon_min, batch_size, train_interval, learning_rate, episode_length, filename, train_test)
    
    if train_test == 'train':
        warm_up_agent(agent, environment, memory_size, display, start_time)
        metrics_df = train_agent(agent, environment, episodes, episode_length, display)
        agent.save_model()
    else:
        agent.load_model()

    if train_test == 'test':
        trade_data = test_agent(agent, environment, render_show)
    else:
        trade_data = {}

    if display:
        print("All done in {:.2f} seconds".format(time.time() - start_time))

    return {
        "metrics_df": metrics_df if train_test == 'train' else None,
        "trades_df": trade_data.get('trades', None),
        "action_policy_df": trade_data.get('q_values', None),
        "reward_list": trade_data.get('rewards', None)
    }

if __name__ == "__main__":
    World(filename=r'./Data\\AAP_data.csv', save_results=False, episodes=10, display=True, train_test='test')
