# Based on: https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/

import random
import gym
import math
import numpy as np
import tensorflow as tf
from collections import deque

class DQNCartPoleSolver():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.state_ = tf.placeholder(tf.float32, shape=[None, 4])
        h = tf.layers.dense(self.state_, units=24, activation=tf.nn.tanh)
        h = tf.layers.dense(h, units=48, activation=tf.nn.tanh)
        self.Q = tf.layers.dense(h, units=2)

        self.Q_ = tf.placeholder(tf.float32, shape=[None, 2])
        loss = tf.losses.mean_squared_error(self.Q_, self.Q)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(0.01, self.global_step, 0.995, 1)
        self.train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=self.global_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done, -1])

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.sess.run(self.Q, feed_dict={self.state_: state}))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done, next_action in minibatch:
            y_target = self.sess.run(self.Q, feed_dict={self.state_: state})
            # y_target[0][action] = reward if done else reward + self.gamma * np.max(self.sess.run(self.Q, feed_dict={self.state_: next_state})[0])
            y_target[0][action] = reward if done else reward + self.gamma * self.sess.run(self.Q, feed_dict={self.state_: next_state})[0][next_action]
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.sess.run(self.train_step, feed_dict={self.state_: np.array(x_batch), self.Q_: np.array(y_batch)})

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                if e % 100 == 0 and not self.quiet:
                    self.env.render()
                action = self.choose_action(state, self.get_epsilon(e))
                # update the "next action" for previous one
                if len(self.memory):
                    self.memory[-1][-1] = action
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials'.format(e, e - 100))
                return e - 100
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            self.replay(self.batch_size)

        if not self.quiet: print('Did not solve after {} episodes'.format(e))
        return e

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
