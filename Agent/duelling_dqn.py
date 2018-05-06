
import pandas as pd
import os
import random
import numpy as np
import time
from keras.layers import Dense, Lambda, Layer, Input, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from Agent.agent import Agent





class DDDQNAgent(Agent):
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
                 train_test='train',
                 symbol=""
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
        self.symbol=symbol


    def save_model(self):
        self.brain.save(r'./Saved Models/'+self.symbol+'.h5')

    def load_model(self):
        self.brain = load_model(r'./Saved Models/'+self.symbol+'.h5')

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
        layer = brain.layers[-2]  # Get the second last layer of the model
        nb_action = brain.output._keras_shape[-1]  #  remove the last layer
        y = Dense(nb_action + 1, activation='linear')(layer.output)
        outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                             output_shape=(nb_action,))(y)  #  Using the max dueling type
        brain = Model(inputs=brain.input, outputs=outputlayer)
        brain.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return brain

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

    def updateTargetModel(self):
        self.brain_.set_weights(self.brain.get_weights())

    def observe(self, state, action, reward, next_state, done, warming_up=False):
        """Memory Management and training of the agent
        """
        self.i = (self.i + 1) % self.memory_size
        self.memory[self.i] = (state, action, reward, next_state, done)
        if (self.i == self.memory_size - 1):
            # print("Memory Refilled")
            pass
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


