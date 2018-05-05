# With TP and SL, limited trades, only 1 open position


from __future__ import absolute_import
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from Environment.core import Env
from tgym.utils import calc_spread

np.random.seed(0)

plt.style.use('dark_background')
mpl.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 15,
        "lines.linewidth": 1,
        "lines.markersize": 8
    }
)
logging.basicConfig(filename='dqn.log', level=logging.INFO)
# create logger
logger = logging.getLogger('tx')
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.propagate = False
logger.setLevel(logging.WARNING) # saeed: not to print the logg in console

class Indicator_1(Env):
    """Class for a discrete (buy/hold/sell) spread trading environment.
    """

    _actions = {
        'hold': np.array([1, 0, 0]),
        'buy': np.array([0, 1, 0]),
        'sell': np.array([0, 0, 1])
    }

    _positions = {
        'flat': np.array([1, 0, 0]),
        'long': np.array([0, 1, 0]),
        'short': np.array([0, 0, 1])
    }

    def __init__(self, data_generator, episode_length=1000,
                 trading_fee=0, time_fee=0, profit_taken=20, stop_loss=-10,
                 reward_factor=10000, history_length=2):
        """Initialisation function

        Args:
            data_generator (tgym.core.DataGenerator): A data
                generator object yielding a 1D array of bid-ask prices.
            episode_length (int): number of steps to play the game for
            trading_fee (float): penalty for trading
            time_fee (float): time fee
            history_length (int): number of historical states to stack in the
                observation vector.
        """

        assert history_length > 0
        self._data_generator = data_generator
        self._first_render = True
        self._trading_fee = trading_fee
        self._time_fee = time_fee
        self._episode_length = episode_length
        self.n_actions = 3
        self._prices_history = []
        self._history_length = history_length
        self._tick_buy = 0
        self._tick_sell = 0
        self.tick_mid = 0 # saeed
        self.tick_cci_14 = 0
        self.tick_rsi_14=0
        self.tick_dx_14 = 0
        self._price = 0
        self._round_digits = 4
        self._holding_position = []  # [('buy',price, profit_taken, stop_loss),...]
        self._max_lost = -1000
        self._reward_factor = reward_factor
        self.reset()
        self.TP_render=False
        self.SL_render = False
        self.Buy_render=False
        self.Sell_render=False
        self.current_action="-"
        self.current_reward=0
        self.unr_pnl=0

    def reset(self):
        """Reset the trading environment. Reset rewards, data generator...

        Returns:
            observation (numpy.array): observation of the state
        """
        self._iteration = 0
        self._data_generator.rewind()
        # self._data_generator._iterator_end()
        self._total_reward = 0
        self._total_pnl = 0
        self._current_pnl = 0
        self._position = self._positions['flat']
        # self._entry_price = 0
        # self._exit_price = 0
        self._closed_plot = False
        self._holding_position = []
        self._max_lost = -1000
        for i in range(self._history_length):
            self._prices_history.append(next(self._data_generator))
        self._tick_buy, self._tick_sell,self.tick_mid ,self.tick_rsi_14,self.tick_cci_14= \
            self._prices_history[0][:5]
        # self._tick_buy, self._tick_sell,self.tick_mid ,self.tick_rsi_14,self.tick_cci_14, self.tick_dx_14= \
        #     self._prices_history[0][:6]

        observation = self._get_observation()
        self.state_shape = observation.shape
        self._action = self._actions['hold']
        return observation

    def step(self, action):
        """Take an action (buy/sell/hold) and computes the immediate reward.

        Args:
            action (numpy.array): Action to be taken, one-hot encoded.

        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """

        self._action = action
        self._iteration += 1
        done = False
        info = {}
        if all(self._position != self._positions['flat']):
            reward = -self._time_fee
        self._current_pnl=0

        #-------------------
        instant_pnl=0
        reward = -self._time_fee
        if all(action == self._actions['buy']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['long']
                self._entry_price = self._price = self._tick_buy
                self.Buy_render = True
            elif all(self._position == self._positions['short']):
                self._exit_price = self._exit_price = self._tick_sell
                instant_pnl = self._entry_price - self._exit_price
                self._position = self._positions['flat']
                self._entry_price = 0
                # self.Buy_render = True
                if (instant_pnl > 0):
                    self.TP_render=True
                else:
                    self.SL_render=True

        elif all(action == self._actions['sell']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['short']
                self._entry_price = self._price = self._tick_sell
                self.Sell_render = True
            elif all(self._position == self._positions['long']):
                self._exit_price = self._tick_buy
                instant_pnl = self._exit_price - self._entry_price
                self._position = self._positions['flat']
                self._entry_price = 0
                # self.Sell_render = True
                if (instant_pnl > 0):
                    self.TP_render = True
                else:
                    self.SL_render = True

        else:
            self.Buy_render = self.Sell_render = False
            self.TP_render = self.SL_render = False

        reward += instant_pnl
        self._total_pnl += instant_pnl
        self._total_reward += reward

        try:
            self._prices_history.append(next(self._data_generator))
            self._tick_sell, self._tick_buy, self.tick_mid, self.tick_rsi_14, self.tick_cci_14= \
            self._prices_history[-1][:5]
        except StopIteration:
            done = True
            info['status'] = 'No more data.'

        # Game over logic
        if self._iteration >= self._episode_length:
            done = True
            info['status'] = 'Time out.'
        if reward <= self._max_lost:
            done = True
            info['status'] = 'Bankrupted.'
        if self._closed_plot:
            info['status'] = 'Closed plot'

        observation = self._get_observation()

        return observation, reward, done, info

    def _handle_close(self, evt):
        self._closed_plot = True


    def return_calc(self,render_show=False):
        trade_details= {}
        if self.Sell_render:
            trade_details = {'Trade':'SELL','Price':self._tick_sell,'Time':self._iteration}
        elif self.Buy_render:
            trade_details = {'Trade': 'BUY', 'Price': self._tick_buy, 'Time': self._iteration}
        if self.TP_render:
            trade_details = {'Trade': 'TP', 'Price': self._exit_price, 'Time': self._iteration}
        elif self.SL_render:
            trade_details = {'Trade': 'SL', 'Price': self._exit_price, 'Time': self._iteration}

        if(not render_show):
            self.TP_render=self.SL_render=False
            self.Buy_render=self.Sell_render=False

        return trade_details

    def render(self, savefig=False, filename='myfig'):
        """Matlplotlib rendering of each step.

        Args:
            savefig (bool): Whether to save the figure as an image or not.
            filename (str): Name of the image file.
        """
        if self._first_render:
            self._f, (self._ax, self._ay, self._az, self._at) = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, squeeze=True,
                        gridspec_kw={'height_ratios': [4, 1, 1, 0]},)



            self._ax = [self._ax]
            self._ay = [self._ay]
            self._az = [self._az]
            self._at = [self._at]
            self._f.set_size_inches(12, 6)
            self._first_render = False
            self._f.canvas.mpl_connect('close_event', self._handle_close)


        #  price
        ask, bid, mid, rsi, cci = self._tick_buy, self._tick_sell,self.tick_mid, self.tick_rsi_14, self.tick_cci_14

        self._ax[-1].plot([self._iteration, self._iteration + 1], [mid, mid], color='white')
        self._ay[-1].plot([self._iteration, self._iteration + 1], [cci, cci], color='green')
        self._az[-1].plot([self._iteration, self._iteration + 1], [rsi, rsi], color='blue')
        self._ay[0].set_ylabel('CCI')
        self._az[0].set_ylabel('RSI')

        ymin, ymax = self._ax[-1].get_ylim()
        yrange = ymax - ymin
        if self.Sell_render:
            self._ax[-1].scatter(self._iteration + 0.5, bid + 0.03 *
                                 yrange, color='orangered', marker='v')
        elif self.Buy_render:
            self._ax[-1].scatter(self._iteration + 0.5, ask - 0.03 *
                                 yrange, color='lawngreen', marker='^')
        if self.TP_render:
            self._ax[-1].scatter(self._iteration + 0.5, bid + 0.03 *
                                 yrange, color='gold', marker='.')
        elif self.SL_render:
            self._ax[-1].scatter(self._iteration + 0.5, ask - 0.03 *
                                 yrange, color='maroon', marker='.')


        self.TP_render=self.SL_render=False
        self.Buy_render=self.Sell_render=False

        plt.suptitle('Total Reward: ' + "%.2f" % self._total_reward +
                     '  Total PnL: ' + "%.2f" % self._total_pnl +
                     '  Unrealized Return: ' + "%.2f" % (self.unrl_pnl*100)  + "% "+
                     '  Pstn: ' + ['flat', 'long', 'short'][list(self._position).index(1)] +
                     '  Action: ' + ['flat', 'long', 'short'][list(self._action).index(1)] +
                     '  Tick:' + "%.2f" % self._iteration)
        self._f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.xticks(range(self._iteration)[::5])
        plt.xlim([max(0, self._iteration - 80.5), self._iteration + 0.5])

        plt.subplots_adjust(top=0.85)
        plt.pause(0.00001) # 0.01
        if savefig:
            plt.savefig(filename)

    def _get_observation(self):
        """Concatenate all necessary elements to create the observation.

        Returns:
            numpy.array: observation array.
        """
        if all(self._position==self._positions['flat']):
            self.unrl_pnl=0
        elif all(self._position==self._positions['long']):
            self.unrl_pnl = (self._prices_history[-1][2]-self._price)/self._prices_history[-1][2]
        elif all(self._position==self._positions['short']):
            self.unrl_pnl = (self._price - self._prices_history[-1][2])/self._prices_history[-1][2]

        return np.concatenate(
            [self._prices_history[-1][3:]] +
            [
                np.array([self.unrl_pnl]),
                np.array(self._position)
            ]
        )

    @staticmethod
    def random_action_fun():
        """The default random action for exploration.
        We hold 80% of the time and buy or sell 10% of the time each.

        Returns:
            numpy.array: array with a 1 on the action index, 0 elsewhere.
        """
        return np.random.multinomial(1, [0.8, 0.1, 0.1])
