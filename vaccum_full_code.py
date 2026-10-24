# for drawing plots
import matplotlib.pyplot as plt
# for math an random
import numpy as np
# for data frame
import pandas as pd
# for a random seed
import random

# variables because python lacks proper enum support
# uniform picks, no intelligence, random
SIMULATION_TYPE_RANDOM = "random"
# only if dirty such, simple stimulus, response rule (responsive)
SIMULATION_TYPE_REFLEX = "reflex"
# stops if all are clean (no-op, stop), reverse directions
# at the end of the array or the start (has memory)
# pros :
#  1 - remembers dirty rooms
# cons :
#  1 - does not use that info to change direction
SIMULATION_TYPE_MODEL = "model"
# nearest dirty can actually use its memory for heuristics
# it's a proper (belief-state agent)
# pros :
#  1 - no wondering around
# cons :
#  1 - a room it hasn't seen is assumed to be dirty
#   but it may not be dirty, so exploring it blind here
SIMULATION_TYPE_NEAREST_DIRTY_MODEL = "nearest dirty model"
# exploit instead of explore
SIMULATION_TYPE_EXPLOIT = "exploit"
# list of all simulation types
SIMULATION_TYPE_FULL_LIST = [
  SIMULATION_TYPE_RANDOM,
  SIMULATION_TYPE_REFLEX,
  SIMULATION_TYPE_MODEL,
  SIMULATION_TYPE_NEAREST_DIRTY_MODEL,
  SIMULATION_TYPE_EXPLOIT,
]

# class used as a config data pack
class VCConfig:
    
    # initialization method
    def __init__ (self,
            room_amount=2,
            steps=50,
            runs_count=100,
            simulation_type=SIMULATION_TYPE_RANDOM):
        # amount of rooms used in the simulation
        self.room_amount = room_amount
        # steps in one run
        self.steps = steps
        # how many times to run the simulation
        self.runs_count = runs_count
        # the type of simulation to run
        self.simulation_type = simulation_type


# geneirc simulation function
def vc_simulate(seed : int, vcc : VCConfig) -> int:
    rng = np.random.default_rng(seed)
    # D = [(r < 0.5) | r is a random number in [0, 1] interval]
    # a boolean array of length vcc.room_amount
    # to store the states of each room
    # with random values of 50% chance of 0 or 1
    dirt_states = rng.random(vcc.room_amount) < 0.5
    # array of booleans so that the model can remember
    # what rooms are clear, 1 means the room is clean
    # for "model" type
    known_clean = np.zeros(vcc.room_amount, dtype=bool)
    # for "nearest dirty model" (pessimistic start, all dirty)
    known_dirty = np.ones(vcc.room_amount, dtype=bool)
    # for "exploit" model (did it see that room ?)
    seen = np.zeros(vcc.room_amount, dtype=bool)
    # start from position 0
    pos = 0
    # score is 0 at the start
    score = 0
    # direction to move (vector)
    direction = 1
    # repeat as many steps as specified
    for _ in range(vcc.steps):

        # random model agent
        if vcc.simulation_type == SIMULATION_TYPE_RANDOM:
            # completely random
            action = rng.choice(['left', 'right', 'sucks'])
        # reflex type agent
        elif vcc.simulation_type == SIMULATION_TYPE_REFLEX:
            # knows the current is dirty or not
            if dirt_states[pos]:
                action = 'suck'
            # other two choices are picked at random
            else:
              action = rng.choice(['left', 'right'])
        # model-based agent
        elif vcc.simulation_type == SIMULATION_TYPE_MODEL:
            # remember the current state of the room
            # 0 -> room is not dirty then done -> 1
            # 1 -> room is dirty then has to something -> 0
            known_clean[pos] = not dirt_states[pos]
            # knows the current is dirty or not
            if dirt_states[pos]:
                action = 'suck'
            # if all is clear, do nothing
            elif known_clean.all():
                action = 'noop'
            # otherwise if 'right' or 'left' is needed
            else:
                # if it has reached the end
                if pos == vcc.room_amount - 1:
                    # go towards the start
                    direction = -1
                elif pos == 0:
                    # go toward the end
                    direction = 1
                # choose right or left based on the direction
                action = 'right' if direction == 1 else 'left'
        elif vcc.simulation_type == SIMULATION_TYPE_NEAREST_DIRTY_MODEL:
            known_dirty[pos] = dirt_states[pos]
            if dirt_states[pos]:
                action = 'suck'
            elif not known_dirty.any():
                action = 'noop'
            else:
                # find the index of all the rooms that are dirty (state = 1)
                targets = np.where(known_dirty)[0]
                # select the nearest target based on absolute value distance
                # min { abs(each target - current pos) } = min distance index
                tg = targets[np.argmin(np.abs(targets - pos))]
                # if the target is right, go to right
                # else if its at the left, go to left
                # (move towards the target)
                action = 'right' if tg > pos else 'left'
        elif vcc.simulation_type == SIMULATION_TYPE_EXPLOIT:
            # mark as seen for later
            seen[pos] = True
            known_dirty[pos] = dirt_states[pos]
            # if dirty, suck
            if dirt_states[pos]:
                action = 'suck'
            # if any dirty near room exists, go there
            elif known_dirty.any():
                targets = np.where(known_dirty)[0]
                tg = targets[np.argmin(np.abs(targets - pos))]
                action = 'right' if tg > pos else 'left'
            # go to the nearest never visited rooms if there is any
            elif not seen.all():
                targets = np.where(~seen)[0]
                tg = targets[np.argmin(np.abs(targets - pos))]
                action = 'right' if tg > pos else 'left'
            # o nothing
            else:
                action = 'noop'
        else:
            # oops
            print('wrong simulation type')
            exit(-1)

        # if action turns out to suck
        if action == 'suck':
            # the current room should get clean
            dirt_states[pos] = False
        # if it wants to go to the right
        elif action == 'right':
            # go right but don't go out of bounds
            # (index >= len is not allowed)
            pos = min(pos + 1, vcc.room_amount - 1)
        # if it wants to go to the left
        elif action == 'left':
            # go to the left but don't go out of bounds
            # negative index is not allowed (index < 0)
            pos = max(pos - 1, 0)
        # give a mark of +1 per clean room (sum 1s)
        score += (~ dirt_states).sum()
    return int(score)

# get the average of multiple experiments
def calc_avg_runs(seed : int, vcc : VCConfig) -> tuple[float, float]:
    # base seed
    rng = np.random.default_rng(seed)
    # generate 100 seeds in the inteval of [0, 2^32] (32-bit integer)
    seeds = rng.integers(0, 2**32, size=vcc.runs_count)
    # run the experiment multiple times
    scores = np.array([vc_simulate(int(s), vcc) for s in seeds])
    # average them out (mean)
    avg_s = scores.mean()
    # get the average error (+- stderr)
    # ddof=1 for simple standard variance (std)
    avg_err = scores.std(ddof=1) / np.sqrt(vcc.runs_count)
    return (avg_s, avg_err)

# test all simulation types
def test_all(k_rooms : int = 3, n_runs : int = 100, sim_types : [str] = []):
    for s_ty in sim_types:
        # make a shared config for this type
        vcc = VCConfig(room_amount=k_rooms, runs_count=n_runs, simulation_type=s_ty)
        # for each seed, simulate and get a score
        mean, stderr = calc_avg_runs(random.randint(1, 2**32), vcc)
        print(f'[+] simulation type {s_ty} : score mean = {mean:8.2f}, stderr = {stderr:.3f}')

if __name__ == '__main__':
  # run all tests
  ss = SIMULATION_TYPE_FULL_LIST
  test_all(k_rooms=3, n_runs=100, sim_types=ss)
  print('excepted results : ')
  if len(ss) == 0:
    exit(0)
  txt = ss[0]
  for s in ss[1:]:
      txt += f' < {s}'
  print(txt)

