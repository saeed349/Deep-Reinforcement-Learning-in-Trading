# import pdb
# from __future__ import absolute_import
import pandas as pd
import math

import random
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers import Dense, Lambda, Layer, Input, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K




from Environment.gens.TA_Gen import TAStreamer
from Environment.envs.indicator_1 import Indicator_1


class DDDQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 episodes,
                 episode_length,
                 train_interval=100,
                 Update_target_frequency=100,
                 memory_size=2000,
                 gamma=0.95,
                 learning_rate=0.001,
                 batch_size=64,
                 epsilon_min=0.01,
                 train_test='train'
                 ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = (self.epsilon - epsilon_min) \
                                 * train_interval / (episodes * episode_length)  # linear decrease rate
        self.learning_rate = learning_rate
        self.train_interval = train_interval
        self.Update_target_frequency = Update_target_frequency
        self.batch_size = batch_size
        self.brain = self._build_brain()
        self.brain_ = self._build_brain()
        self.i = 0
        self.train_test = train_test

    def save_model(self):
        self.brain.save('DQN_Agent.h5')

    def load_model(self):
        self.brain = load_model('DQN_Agent.h5')

    def _build_brain(self):
        """Build the agent's brain
        """
        # pdb.set_trace()
        brain = Sequential()
        neurons_per_layer = 24
        activation = "relu"
        brain.add(Dense(neurons_per_layer,
                        input_dim=self.state_size,
                        activation=activation))
        brain.add(Dense(neurons_per_layer*2, activation=activation))
        brain.add(Dense(neurons_per_layer*4, activation=activation))
        brain.add(Dense(self.action_size, activation='linear'))
        layer = brain.layers[-2]  # Ash: Get the second last layer of the model
        nb_action = brain.output._keras_shape[-1]  # Ash: remove the last layer
        # layer y --> shape (nb_action+1), y[:,0] = V(s;theta), y[:,1:] = A(s,a;theta)
        y = Dense(nb_action + 1, activation='linear')(layer.output)
        outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                             output_shape=(nb_action,))(y)  # Ash: Using the max dueling type
        brain = Model(inputs=brain.input, outputs=outputlayer)
        brain.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return brain

    # softmax implementation described in Onenote-FE800-Problems-Random Action-Epsilon
    def act(self, state, test=False):
        """Acting Policy of the DDDQNAgent
        """
        act_values=[]
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon and self.train_test == 'train' and not test:
            action[random.randrange(
                self.action_size)] = 1  # saeed : it would put 1 in either of the 3 action positions randomly
        else:
            state = state.reshape(1, self.state_size)
            act_values = self.brain.predict(state)
            action[np.argmax(act_values[0])] = 1
        if test:
            return action, act_values
        else:
            return action

        # # Softmax Implementation, here (1-epsilon is the temperature), we can set epsilon_max to >1 to give more confidence in the selection.
        # act_values = self.brain.predict(state) * (1-self.epsilon)
        # soft = np.exp(act_values) / np.sum(np.exp(act_values))
        # return (np.random.multinomial(1, soft))

    def updateTargetModel(self):
        self.brain_.set_weights(self.brain.get_weights())

    def observe(self, state, action, reward, next_state, done, warming_up=False):
        """Memory Management and training of the agent
        """
        self.i = (self.i + 1) % self.memory_size
        self.memory[self.i] = (state, action, reward, next_state, done)
        if (self.i == self.memory_size - 1):
            print("Memory Refilled")
        #   saeed : increments of 3000, the agent trains
        if (not warming_up) and (self.i % self.train_interval) == 0:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
            state, action, reward, next_state, done = self._get_batches()
            reward += (self.gamma
                       * np.logical_not(done)
                       *self.brain_.predict(next_state)[range(self.batch_size), (np.argmax(self.brain.predict(next_state), axis=1))])
            q_target = self.brain.predict(state)
            q_target[action[0], action[1]] = reward

            if self.i % self.Update_target_frequency == 0:
                self.updateTargetModel()


            return self.brain.fit(state, q_target,
                                  batch_size=self.batch_size,
                                  epochs=1,
                                  verbose=False,
                                  validation_split=0.1)

    def _get_batches(self):
        """Selecting a batch of memory
           Split it into categorical subbatches
           Process action_batch into a position vector
        """
        batch = np.array(random.sample(self.memory, self.batch_size))
        state_batch = np.concatenate(batch[:, 0]) \
            .reshape(self.batch_size, self.state_size)
        action_batch = np.concatenate(batch[:, 1]) \
            .reshape(self.batch_size, self.action_size)
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3]) \
            .reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]
        # action processing
        action_batch = np.where(action_batch == 1)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


if __name__ == "__main__":
    start = time.time()
    # Instantiating the environmnent
    train_test = 'train'
    # train_test = 'test'

    episodes = 200
    train_test_split = 0.75
    trading_fee = .0001
    time_fee = .001
    # render_show = False
    render_show = True
    gen_type = 'C'
    trading_type = 'T'



    # trading_fee = .2 #0.009 #.0002 #.2 for wavy-gen

    history_length = 2
    profit_taken = 10
    stop_loss = -5



    import os
    # os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
    os.chdir("..") #  to go back to the main directory

    if gen_type == 'C':
        # filename = r'./Data/AAPL_OHLC.csv'
        filename = r'./Data/XOM_OHLC.csv'
        # filename = r'./Data/synth_train.csv'
        generator = TAStreamer(filename=filename, mode='train', split=train_test_split)
        episode_length = round(int(len(pd.read_csv(filename))*train_test_split), -1)
        # episode_length = round(len(pd.read_csv(filename)), -1)
        # episode_length = round(len(pd.read_csv(filename))-15, -1)

    if gen_type != 'C':
        episode_length = 400


    # history_length number of historical states in the observation vector.


    if trading_type == 'T':
        environment = Indicator_1(data_generator=generator,
                                  trading_fee=trading_fee,
                                  time_fee=time_fee,
                                  history_length=history_length,
                                  episode_length=episode_length,
                                  profit_taken=profit_taken,
                                  stop_loss=stop_loss)
        action_size = len(Indicator_1._actions)

    state = environment.reset()
    # Instantiating the agent
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

    if (train_test == 'train'):  # or train_test=='test'
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
            # print("Loss=",str(loss.history["loss"])," Val_Loss=",str(loss.history["val_loss"]))
            print("Loss=", str(np.mean(loss_list_temp)), " Val_Loss=", str(np.mean(val_loss_list_temp)))
            loss_list.append(np.mean(loss_list_temp))
            val_loss_list.append(np.mean(val_loss_list_temp))
            reward_list.append(rew)
            epsilon_list.append(round(agent.epsilon, 2))
            # if(loss_list[-1]<best_loss):
            #     best_loss=loss_list[-1]
            #     agent.save_model()
            #     print("MODEL SAVED AT EPOCH=",ep)
            # if(rew>best_reward):
            #     best_reward=reward
            #     print("MODEL SAVED AT EPOCH=",ep)
            #     agent.save_model()
        agent.save_model()
        metrics_df=pd.DataFrame({'loss':loss_list,'val_loss':val_loss_list,'reward':reward_list,'epsilon':epsilon_list})
        metrics_df.to_csv('perf_metrics.csv')

    # if (train_test == "test"):
    if(gen_type=='C'):
        agent.load_model()
        generator = TAStreamer(filename=filename, mode='test', split=train_test_split)
        environment = Indicator_1(data_generator=generator,
                                  trading_fee=trading_fee,
                                  time_fee=time_fee,
                                  history_length=history_length,
                                  episode_length=episode_length,
                                  profit_taken=profit_taken,
                                  stop_loss=stop_loss)

    done = False
    state = environment.reset()
    q_values_list=[]
    state_list=[]
    action_list=[]
    reward_list=[]

    trade_list=[]
    # render_show=False

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
                # print(trade_list[-1])

            if(render_show):
                environment.render()




        q_values_list.append(q_values)
        state_list.append(state)
        action_list.append(action)

    print(sum(reward_list))

    trades_df=pd.DataFrame(trade_list)
    trades_df.to_csv('trade_list.csv')

    action_policy_df = pd.DataFrame({'q_values':q_values_list,'state':state_list,'action':action_list})
    action_policy_df.to_pickle('action_policy.pkl')


    print("All done:", str(time.time() - start))
