import csv
import os

import gym
import argparse
import numpy as np
from twod_tmaze import TMaze
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.special import expit
from scipy.special import logsumexp
import dill

class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

class EgreedyPolicy:
    def __init__(self, rng, nfeatures, nactions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.weights = np.zeros((nfeatures, nactions))

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.weights.shape[1]))
        return int(np.argmax(self.value(phi)))

class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = np.zeros((nfeatures, nactions))
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        return int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))

class SigmoidTermination:
    def __init__(self, rng, nfeatures):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi

class IntraOptionQLearning:
    def __init__(self, discount, lr, terminations, weights):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights

    def start(self, phi, option):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi, option)

    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)

    def advantage(self, phi, option=None):
        values = self.value(phi)
        advantages = values - np.max(values)
        if option is None:
            return advantages
        return advantages[option]

    def update(self, phi, option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_option] += self.lr*tderror
        if not done:
            self.last_value = current_values[option]
        self.last_option = option
        self.last_phi = phi

        return update_target

class IntraOptionActionQLearning:
    def __init__(self, discount, lr, terminations, weights, qbigomega):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.qbigomega = qbigomega

    def value(self, phi, option, action):
        return np.sum(self.weights[phi, option, action], axis=0)

    def start(self, phi, option, action):
        self.last_phi = phi
        self.last_option = option
        self.last_action = action

    def update(self, phi, option, action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_phi, self.last_option, self.last_action)
        self.weights[self.last_phi, self.last_option, self.last_action] += self.lr*tderror

        self.last_phi = phi
        self.last_option = option
        self.last_action = action

class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr

    def update(self, phi, option):
        magnitude, direction = self.terminations[option].grad(phi)
        self.terminations[option].weights[direction] -= \
                self.lr*magnitude*(self.critic.advantage(phi, option))

class IntraOptionGradient:
    def __init__(self, option_policies, lr):
        self.lr = lr
        self.option_policies = option_policies

    def update(self, phi, option, action, critic):
        actions_pmf = self.option_policies[option].pmf(phi)
        self.option_policies[option].weights[phi, :] -= self.lr*critic*actions_pmf
        self.option_policies[option].weights[phi, action] += self.lr*critic

class OneStepTermination:
    def sample(self, phi):
        return 1

    def pmf(self, phi):
        return 1.

class FixedActionPolicies:
    def __init__(self, action, nactions):
        self.action = action
        self.probs = np.eye(nactions)[action]

    def sample(self, phi):
        return self.action

    def pmf(self, phi):
        return self.probs

def save_params(args, dir_name):
    f = os.path.join(dir_name, "Params.txt")
    with open(f, "w") as f_w:
        for param, val in sorted(vars(args).items()):
            f_w.write("{0}:{1}\n".format(param, val))

class Parameter_exchange:
    def __init__(self, discount, lr, terminations, qbigomega, qU, weights, policy, option_policies, status_continuity):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.status_continuity = status_continuity
        self.weights = weights
        self.qbigomega = qbigomega
        self.qU = qU
        self.policy = policy
        self.option_policies = option_policies

    def start(self, phi, option):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi, option)

    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)

    def Maximum_Entropy(self, around_status_list):
        state_num_of_option = [0 for i in range(args.noptions)]
        for i in around_status_list:
            option = self.policy.sample([i])
            state_num_of_option[option] += 1
        max_entropy = 0
        all_state_num = np.sum(state_num_of_option)
        for i in state_num_of_option:
            if i != 0:
                max_entropy -= i/all_state_num * np.log2(i/all_state_num)
        return max_entropy

    def update(self, phi, option, action, done, around_status_list):
        status_continuity = 0
        action_certainty = 0
        phi_ = phi[0]
        if not done:
            for i in around_status_list[phi_]['around']:
                status_continuity += self.policy.pmf([i])[option] - self.policy.pmf(phi)[option]
            asl_len = len(around_status_list[phi_]['around'])
            status_continuity = status_continuity/asl_len if asl_len>0 else status_continuity

            self.status_continuity[phi, option] = status_continuity

            max_entropy = self.Maximum_Entropy(around_status_list)
            self.exchange(phi)
            new_max_entropy = self.Maximum_Entropy(around_status_list)
            if np.abs(status_continuity) + 2*(new_max_entropy - max_entropy) <= 0.7:
                self.exchange(phi)

    def exchange(self, phi):
        min_option = np.argsort(self.status_continuity[phi])[0][0]
        max_option = np.argsort(self.status_continuity[phi])[0][-1]

        exchange_qbigomega_weight = self.qbigomega.weights[phi, min_option]
        self.qbigomega.weights[phi, min_option] = self.qbigomega.weights[phi, max_option]
        self.qbigomega.weights[phi, max_option] = exchange_qbigomega_weight

        exchange_qU_weight = self.qU.weights[phi, min_option]
        self.qU.weights[phi, min_option] = self.qU.weights[phi, max_option]
        self.qU.weights[phi, max_option] = exchange_qU_weight

        exchange_op_weight = self.option_policies[min_option].weights[phi]
        self.option_policies[min_option].weights[phi] = self.option_policies[max_option].weights[phi]
        self.option_policies[max_option].weights[phi] = exchange_op_weight

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='TMaze')
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=1e-2)
    parser.add_argument('--epsilon', help="Epsilon-greedy for policy over options", type=float, default=1e-2)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=2000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=100)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=1000)
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_true', default=False)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--primitive', help="Augment with primitive", default=False, action='store_true')

    args = parser.parse_args()

    rng = np.random.RandomState(1234)


    outer_dir = "AggregateOptionCritic"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    xs = np.arange(-.4, 0.401, 0.1)
    xs = np.around(xs, decimals=1)
    xs = xs.tolist()
    ys = np.arange(0.5, -0.201, -0.1)
    ys = np.around(ys, decimals=1)
    ys = ys.tolist()

    actions0 = np.arange(-0.5, 0.6, 1)
    actions0 = np.around(actions0, decimals=2)
    actions0 = actions0.tolist()
    actions1 = np.arange(-0.5, 0.6, 1)
    actions1 = np.around(actions1, decimals=2)
    actions1 = actions1.tolist()

    num_states = len(xs)*len(ys)
    num_actions = len(actions0) * len(actions1)

    history = np.zeros((args.nruns, args.nepisodes, 3), dtype=np.float32)

    for run in range(args.nruns):
        if args.env == "TMaze":
            env = TMaze()
        else:
            env = gym.make(args.env)
        features = Tabular(num_states)
        nfeatures, nactions = len(features), num_actions
        around_status_list = {}
        filedata_list = []

        # The intra-option policies are linear-softmax functions
        option_policies = [SoftmaxPolicy(rng, nfeatures, nactions, args.temperature) for _ in range(args.noptions)]
        if args.primitive:
            option_policies.extend([FixedActionPolicies(act, nactions) for act in range(nactions)])

        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(rng, nfeatures) for _ in range(args.noptions)]
        if args.primitive:
            option_terminations.extend([OneStepTermination() for _ in range(nactions)])

        # E-greedy policy over options
        #policy = EgreedyPolicy(rng, nfeatures, args.noptions, args.epsilon)
        policy = SoftmaxPolicy(rng, nfeatures, args.noptions, args.temperature)
        status_continuity = np.zeros((nfeatures, args.noptions))

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, policy.weights)

        # Learn Qomega separately
        action_weights = np.zeros((nfeatures, args.noptions, nactions))
        comprehensive_evaluation = np.zeros((nfeatures, args.noptions, nactions))
        action_critic = IntraOptionActionQLearning(args.discount, args.lr_critic, option_terminations, action_weights,
                                                   critic)

        # Improvement of the termination functions based on gradients
        termination_improvement = TerminationGradient(option_terminations, critic, args.lr_term)

        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(option_policies, args.lr_intra)

        # Parameter_exchange exchange option based on comprehensive evaluation
        exchange = Parameter_exchange(args.discount, args.lr_critic, option_terminations, critic, action_critic,
                                      comprehensive_evaluation, policy, option_policies, status_continuity)

        for episode in range(args.nepisodes):
            phi_list = []
            if episode == 1500:
                print("********************New Goal***********************")
                if hasattr(env, 'NAME'):  # Switch the goal for TMaze
                    preferred = np.argmax(env.counts)
                    from twod_tmaze import TMaze

                    env = TMaze(change_goal=[-.3, .3]) if not preferred else TMaze(change_goal=[.3, .3])
                else:  # Switch the direction for HalfCheetah
                    env.env.env.reset_task({'direction': -1})

            phi_ = features(env.reset_model())
            phi_ = np.around(phi_, decimals=1)
            phi_ = phi_.tolist()
            phi = int(xs.index(phi_[0][0])*len(ys) + ys.index(phi_[0][1]))
            phi = [phi]
            phi_list.append(int(phi[0]))
            last_phi = phi[0]
            option = policy.sample(phi)
            last_option = option
            action = option_policies[option].sample(phi)
            exchange.start(phi, option)
            critic.start(phi, option)
            action_critic.start(phi, option, action)

            cumreward = 0.
            duration = 1
            option_switches = 0
            avgduration = 0.

            if int(phi[0]) not in around_status_list:
                around_status_list[int(phi[0])] = {'tab':[0, -1], 'around':[]}

            for step in range(args.nsteps):
                div = action // len(actions1)
                mod = action % len(actions1)
                action0 = actions0[div]
                action1 = actions1[mod]
                action = np.array([action0, action1])
                observation, reward, done, _ = env.step(action)
                phi_ = observation
                phi_ = np.around(phi_, decimals=1)
                phi_ = phi_.tolist()
                phi = int(xs.index(phi_[0]) * len(ys) + ys.index(phi_[1]))
                phi = [phi]
                phi_list.append(int(phi[0]))
                if int(phi[0]) not in around_status_list:
                    around_status_list[int(phi[0])] = {'tab': [0, -1], 'around': []}

                # Termination might occur upon entering the new state
                if option_terminations[option].sample(phi):
                    option = policy.sample(phi)
                    option_switches += 1
                    avgduration += (1./option_switches)*(duration - avgduration)
                    duration = 1

                action = option_policies[option].sample(phi)

                # Critic update
                exchange.update(phi, option, action, done, around_status_list)
                update_target = critic.update(phi, option, reward, done)
                action_critic.update(phi, option, action, reward, done)

                if around_status_list[last_phi]['tab'][0] == 0:
                    for i in range(args.noptions):
                        if policy.pmf([last_phi])[i] == np.max(policy.pmf([last_phi])) and np.max(policy.pmf([last_phi])) > 1.01/args.noptions:
                            around_status_list[last_phi]['tab'][0] = 1
                            around_status_list[last_phi]['tab'][1] = i
                            break
                else:
                    last_control_option = around_status_list[last_phi]['tab'][1]
                    for i in range(args.noptions):
                        if i != last_control_option and policy.pmf([last_phi])[i] == np.max(policy.pmf([last_phi])) and np.max(policy.pmf([last_phi])) > 1.01/args.noptions:
                            around_status_list[last_phi]['tab'][1] = i
                            break

                if isinstance(option_policies[option], SoftmaxPolicy):
                    # Intra-option policy update
                    critic_feedback = action_critic.value(phi, option, action)
                    if args.baseline:
                        critic_feedback -= critic.value(phi, option)
                    intraoption_improvement.update(phi, option, action, critic_feedback)

                    # Termination update
                    termination_improvement.update(phi, option)


                cumreward += reward
                duration += 1
                last_phi = int(phi[0])
                last_option = option
                if done:
                    break
            for i in range(len(phi_list)):
                if i < len(phi_list)-1:
                    if i == 0:
                        around_status_list[phi_list[i]]['around'].append(phi_list[i+1])
                    elif i >= 1:
                        around_status_list[phi_list[i]]['around'].append(phi_list[i - 1])
                        around_status_list[phi_list[i]]['around'].append(phi_list[i + 1])
                elif i == len(phi_list)-1:
                    if i == 0:
                        around_status_list[phi_list[i]]['around'].append(phi_list[i+1])
                    elif i >= 1:
                        around_status_list[phi_list[i]]['around'].append(phi_list[i - 1])
            for i in around_status_list:
                around_status_list[i]['around'] = list(set(around_status_list[i]['around']))
            history[run, episode, 0] = step
            history[run, episode, 1] = avgduration
            filedata_list.append([cumreward, avgduration, option_switches, step])
            print('Run {} episode {} steps {} cumreward {} avg. duration {} switches {}'.format(run, episode, step,
                                                                                                cumreward, avgduration,
                                                                                                option_switches))
        filename = "aoc_" + str(run) + ".csv"
        filename = os.path.join(outer_dir, filename)
        with open(filename, mode='w', newline='') as f:
            f_writer = csv.writer(f)
            f_writer.writerow(['reward', 'avg. duration', 'switch_num', 'step'])
            f_writer.writerows(filedata_list)