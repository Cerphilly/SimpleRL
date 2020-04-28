import numpy as np

def dmstate(observation):
    state = []
    for key, value in observation.observation.items():
        state.extend(list(value))

    return state

def dmstep(step):
    next_state = dmstate(step)
    reward = step.reward
    done = step.last()

    return next_state, reward, done

def dmextendstate(observation):
    state = []
    extend1 = [0, 0, 0, 0]
    extend2 = [0, 0]
    extend = [[0,0], [0]]
    i = 0

    for key, value in observation.observation.items():
        state.extend(list(value))
        state.extend(extend[i])
        i += 1

    return state

def dmextendstep(step):
    next_state = dmextendstate(step)
    reward = step.reward
    done = step.last()

    return next_state, reward, done