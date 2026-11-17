import numpy as np
from numba import njit
D=2

def edges(mask):
    idx = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(idx[::2], idx[1::2])

@njit(cache=False)
def decode(emissions, switch, into_dead, out_of_dead):
    n = len(emissions)
    back = np.zeros((n, 3), np.int8)
    score = emissions[0].copy()
    for i in range(1, n):
        new = emissions[i].copy()
        for current in range(3):
            best_score = -1e+30
            for previous in range(3):
                penalty = 0.0
                if current != previous:
                    if current == D:
                        penalty = into_dead
                    elif previous == D:
                        penalty = out_of_dead
                    else:
                        penalty = switch[i, current]
                proposed = score[previous] - penalty
                if proposed > best_score:
                    best_score = proposed
                    back[i, current] = previous
            new[current] += best_score
        score = new
    state = np.empty(n, np.int8)
    state[-1] = np.argmax(score)
    for i in range(n - 1, 0, -1):
        state[i - 1] = back[i, state[i]]
    return state

def apply(probabilities, ball, exit_margin):
    emissions=np.log(np.clip(probabilities,1e-6,1.))
    emissions[exit_margin < -.45,2] += 8.
    observed=np.isfinite(ball).all(axis=1)
    costs=np.full((len(emissions),2),1.)
    costs += 2. * ~observed[:,None]
    labels=decode(emissions,costs,7.,7.)
    for a,b in edges(labels==2):
        if a and b<len(labels) and b-a<10 and not (exit_margin[a:b]<-.45).any():
            labels[a:b]=labels[a-1]
    return labels
