import numpy as np
import time


class mdp(object):
    def __init__(self, kernel = np.ones((1,1,1,1)), R = [1], action_mask = np.ones((1, 1))):
        """
        :param kernel(s_,r,s,a): joint prob. of transition to state s_ and R[r] when choosing action a at state s
        :param R: R[r] is reward value at index r
        :param action_mask: action_mask[s][a]==1 only if action a is allowed at state s
        """
        assert np.shape(kernel)[0] == np.shape(kernel)[2]
        assert np.shape(kernel)[1] == len(R)
        assert np.shape(action_mask)[1] == np.shape(kernel)[3]
        self.kernel = kernel
        # State space
        self.lenS = np.shape(kernel)[0]
        self.S = np.arange(0, self.lenS)
        # Reward space
        self.lenR = len(R)
        self.R = np.asarray(R)

        # Action space
        self.lenA = np.shape(kernel)[3]
        self.A = np.zeros(self.lenS, dtype=object)
        for s in self.S:
            self.A[s] = (np.argwhere(action_mask[s] == 1)).reshape(self.lenA)

    def get_transition_matrix(self, f = None):
        if f is None:
            f = np.zeros(self.lenS)
        Pf = np.zeros((self.lenS, self.lenS))
        for s in range(self.lenS):
            if f[s] == 0:
                Pf[s][0] = self.kernel[0][0][s][0]
            elif f[s] == 1:
                Pf[s][(s+1)%self.lenS] += self.kernel[(s+1)%self.lenS][int((s+1)/self.lenS)][s][1]
                Pf[s][0] += self.kernel[0][0][s][1]

        return Pf

    def get_reward_vector(self, f = None):#It was hard to make use of kernel array. Assignment 2 didn't specifically mentioned to make use of it, so..
        if f is None:
            f = np.zeros(self.lenS)
        Rf = np.zeros(self.lenS)
        for s in range(self.lenS):
            if f[s] == 0:
                Rf[s] = reward_str[0]
            elif f[s] == 1:
                Rf[s] = (1-noise)*reward_str[s] + noise*reward_str[0]
        return Rf

def cliffwalk(action_str = [1, 1, 1, 1], reward_str = [0, 0, 0, 1], noise = 0.1):
    assert len(action_str) == len(reward_str)
    # State space
    lenS = len(action_str)
    S = np.arange(0,lenS)

    # Action space
    lenA = 2
    A = np.arange(0,lenA)

    # Reward space
    R = np.asarray(list(set(np.concatenate((reward_str, [0])))))
    lenR = len(R)
    kernel = np.zeros((lenS, lenR, lenS, lenA))
    for s in S:
        # action_str[s]: action intended at state 2
        # (action_str[s]+1)%2: action not intended at state 2
        kernel[(s + 1) % lenS][np.argwhere(R==reward_str[s])[0][0]][s][action_str[s]] += 1 - noise
        kernel[0][np.argwhere(R == 0)[0][0]][s][action_str[s]] += noise
        kernel[0][np.argwhere(R == 0)[0][0]][s][(action_str[s]+1)%2] = 1

    env = mdp(kernel, R, np.ones((lenS, lenA)))
    return env

class my_example(object):
    def __init__(self, action_str = [1, 1, 1, 1], reward_str = [1, 0, 0, 2], noise = 0.1):#[[1], [0], [0], [1.1]]
        pass

class policy_iteration (object):
    def __init__(self, env = cliffwalk(), discount = 0.99, init_v = None, init_policy = None):
        self.env = env
        self.discount = discount
        self.elapsed_time = 0.0
        self.elapsed_iter = 0

        if init_v is None:
            self.Vt = np.ones(env.lenS)
        else:
            self.Vt = init_v
        if init_policy is None:
            self.ft = np.zeros(env.lenS)
        else:
            self.ft = init_policy

    def iteration(self):
        done = False
        start_time = time.time()
        # Perform an iteration
        print("Policy iteration start")

        while not done:
            print("elasped iter: ", self.elapsed_iter)
            self.policy_evaluate()
            done = self.policy_improve()
            self.elapsed_iter += 1

        # update statistics
        self.elapsed_time = time.time() - start_time


    def policy_evaluate(self):
        self.Vt = np.matmul(np.linalg.inv(np.eye(env.lenS) - self.discount*env.get_transition_matrix(self.ft)).round(6), env.get_reward_vector(self.ft)).round(6)
        #print(np.linalg.inv(np.eye(env.lenS) - self.discount*env.get_transition_matrix(self.ft)).round(5), env.get_reward_vector(self.ft), self.Vt)
        print("policy evaluated: {}".format(self.elapsed_iter), self.Vt)


    def policy_improve(self):
        #print("Policy {}".format(self.elapsed_iter), self.ft)
        self.ft_ = np.zeros(env.lenS)#next policy
        for s in range(env.lenS):
            temp = np.zeros(env.lenS)
            policy1 = env.get_reward_vector([0,0,0,0])[s] + self.discount*(np.sum(env.get_transition_matrix([0,0,0,0])[s]*self.Vt))
            #print(env.get_reward_vector([0,0,0,0])[s],env.get_transition_matrix([0,0,0,0])[s],self.Vt)
            policy2 = env.get_reward_vector([1,1,1,1])[s] + self.discount*(np.sum(env.get_transition_matrix([1,1,1,1])[s]*self.Vt))
            #print(env.get_reward_vector([1,1,1,1])[s],env.get_transition_matrix([1,1,1,1])[s],self.Vt)

            self.ft_[s] = np.argmax([policy1, policy2])
            #print(policy1, policy2, self.ft_[s])

        done = True
        for s in range(env.lenS):
            if self.ft[s] != self.ft_[s]:#if policy is not converged, done=False
                done = False

        if done == True:
            print("policy iteration done -> Policy:", self.ft)
            return True
        else:
            self.ft = self.ft_
            print("policy improved: {}".format(self.elapsed_iter), self.ft)
            return False

class value_iteration (object):
    def __init__(self, env = cliffwalk(), discount = 0.99, init_v = None):
        self.env = env
        self.discount = discount
        self.elapsed_time = 0.0
        self.elapsed_iter = 0

        if init_v is None:
            self.Vt = np.zeros(env.lenS)
        else:
            self.Vt = init_v

        self.ft = np.zeros(env.lenS)

    def iteration(self):
        # Perform an iteration
        self.Vt = tuple(np.zeros(env.lenS))#Used tuple because of an error that changes data of self.Vt unwantedly.
        self.ft = np.zeros(env.lenS)
        self.ft_next = np.zeros(env.lenS)

        print("Value iteration start: ", self.Vt)
        start_time = time.time()

        while True:
            temp = np.zeros(env.lenS)

            for s in range(env.lenS):
                a = env.get_reward_vector([0,0,0,0])[s] + self.discount * np.sum(env.get_transition_matrix([0,0,0,0])[s]*self.Vt)
                #print(env.get_reward_vector(temp)[s], env.get_transition_matrix(temp)[s], self.Vt)
                b = env.get_reward_vector([1,1,1,1])[s] + self.discount * np.sum(env.get_transition_matrix([1,1,1,1])[s]*self.Vt)
                #print(env.get_reward_vector([1,1,1,1])[s], env.get_transition_matrix([1,1,1,1])[s], self.Vt)
                temp[s] = max([a, b])

            self.elapsed_iter += 1

            print("Elasped iter: {}".format(self.elapsed_iter), self.Vt)

            if np.linalg.norm((np.array(self.Vt) - np.array(temp)), ord=np.inf) < 0.0001:#1e-4 = 0.0001
                self.Vt = temp
                print("Final value:", np.array(self.Vt))
                break

            else:
                self.Vt = tuple(temp)

        for s in range(env.lenS):
            a = (env.get_reward_vector([0,0,0,0])[s] + self.discount * np.sum(env.get_transition_matrix([0,0,0,0])[s] * self.Vt))
            b = (env.get_reward_vector([1,1,1,1])[s] + self.discount * np.sum(env.get_transition_matrix([1,1,1,1])[s] * self.Vt))

            self.ft[s] = np.argmax([a, b])

        print("Value iteration done -> Policy: ", self.ft)

        # update statistics
        self.elapsed_time = time.time() - start_time


if __name__ == '__main__':
    # Example main
    action_str = [1, 1, 1, 1]
    reward_str = [0, 0, 0, 1]
    noise = 0.0
    env = cliffwalk(action_str, reward_str, noise)
    print("get_reward_vector", '\n', env.get_reward_vector([0,0,0,0]), '\n', env.get_reward_vector([1,1,1,1]))
    print("get_transition_matrix",'\n', env.get_transition_matrix([0,0,0,0]), '\n', env.get_transition_matrix([1,1,1,1]))
    pol_iter = policy_iteration(env, 0.99)
    val_iter = value_iteration(env, 0.99)


    pol_iter.iteration()
    val_iter.iteration()
    print("policy iteration num: ", pol_iter.elapsed_iter)
    print("value iteration num: ", val_iter.elapsed_iter)
    print('Policy iteration took %fs'%pol_iter.elapsed_time)
    print('Value iteration took %fs' %val_iter.elapsed_time)



