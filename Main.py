import pandas as pd
import os
import random
import numpy as np
import time

from Environment.gens.TA_Gen import TAStreamer
from Environment.envs.indicator_1 import Indicator_1

from Agent.duelling_dqn import DDDQNAgent

if __name__ == "__main__":
    start = time.time()
    # Instantiating the environmnent
    train_test = 'train'
    # train_test = 'test'

    episodes = 10
    train_test_split = 0.75
    trading_fee = .0001
    time_fee = .001
    render_show = False
    # render_show = True
    gen_type = 'C'
    trading_type = 'T'


    # os.chdir("..") #  to go back to the main directory

    if gen_type == 'C':
        filename = r'./Data/XOM_OHLC.csv'
        generator = TAStreamer(filename=filename, mode='train', split=train_test_split)
        episode_length = round(int(len(pd.read_csv(filename))*train_test_split), -1)

    if gen_type != 'C':
        episode_length = 400

    if trading_type == 'T':
        environment = Indicator_1(data_generator=generator,
                                  trading_fee=trading_fee,
                                  time_fee=time_fee,
                                  episode_length=episode_length)
        action_size = len(Indicator_1._actions)

    state = environment.reset()
    memory_size = 3000
    state_size = len(state)
    gamma = 0.96
    epsilon_min = 0.01
    batch_size = 64
    train_interval = 10
    learning_rate = 0.001

    agent = DDDQNAgent(state_size=state_size,
                     action_size=action_size,
                     memory_size=memory_size,
                     episodes=episodes,
                     episode_length=episode_length,
                     train_interval=train_interval,
                     gamma=gamma,
                     learning_rate=learning_rate,
                     batch_size=batch_size,
                     epsilon_min=epsilon_min,
                     train_test=train_test)

    # Warming up the agent
    if (train_test == 'train'):
        for _ in range(memory_size):
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            agent.observe(state, action, reward, next_state, done, warming_up=True)
        print('completed mem allocation: ', time.time() - start)

    # Training the agent
    loss_list=[]
    val_loss_list=[]
    reward_list=[]
    epsilon_list=[]
    if train_test == "train":
        best_loss = 9999
        best_reward = 0
        for ep in range(episodes):
            ms = time.time()
            state = environment.reset()
            rew = 0
            loss_list_temp = []
            val_loss_list_temp = []

            for _ in range(episode_length):
                action = agent.act(state)
                next_state, reward, done, _ = environment.step(action)
                loss = agent.observe(state, action, reward, next_state,
                                     done)  # loss would be none if the episode length is not % by 10
                state = next_state
                rew += reward
                if(loss):
                    loss_list_temp.append(round(loss.history["loss"][0],3))
                    val_loss_list_temp.append(round(loss.history["val_loss"][0],3))
            print("Ep:" + str(ep)
                  + "| rew:" + str(round(rew, 2))
                  + "| eps:" + str(round(agent.epsilon, 2))
                  + "| loss:" + str(round(loss.history["loss"][0], 4))
                  + "| runtime:" + str(time.time() - ms))
            print("Loss=", str(np.mean(loss_list_temp)), " Val_Loss=", str(np.mean(val_loss_list_temp)))
            loss_list.append(np.mean(loss_list_temp))
            val_loss_list.append(np.mean(val_loss_list_temp))
            reward_list.append(rew)
            epsilon_list.append(round(agent.epsilon, 2))

        agent.save_model()
        metrics_df=pd.DataFrame({'loss':loss_list,'val_loss':val_loss_list,'reward':reward_list,'epsilon':epsilon_list})
        metrics_df.to_csv(r'./Results/perf_metrics.csv')

    if(gen_type=='C'):
        agent.load_model()
        generator = TAStreamer(filename=filename, mode='test', split=train_test_split)
        environment = Indicator_1(data_generator=generator,
                                  trading_fee=trading_fee,
                                  time_fee=time_fee,
                                  episode_length=episode_length,)

    done = False
    state = environment.reset()
    q_values_list=[]
    state_list=[]
    action_list=[]
    reward_list=[]
    trade_list=[]

    while not done:
        action, q_values = agent.act(state, test=True)
        state, reward, done, info = environment.step(action)
        if 'status' in info and info['status'] == 'Closed plot':
            done = True
        else:
            # print(reward)
            reward_list.append(reward)

            calc_returns=environment.return_calc(render_show)
            if calc_returns:
                trade_list.append(calc_returns)

            if(render_show):
                environment.render()


        q_values_list.append(q_values)
        state_list.append(state)
        action_list.append(action)

    print(sum(reward_list))

    trades_df=pd.DataFrame(trade_list)
    trades_df.to_csv(r'./Results/trade_list.csv')

    action_policy_df = pd.DataFrame({'q_values':q_values_list,'state':state_list,'action':action_list})
    action_policy_df.to_pickle(r'./Results/action_policy.pkl')


    print("All done:", str(time.time() - start))
