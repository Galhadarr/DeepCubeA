from typing import List, Tuple
import numpy as np
from environments.environment_abstract import Environment, State
from utils import misc_utils


def is_valid_soln(state: State, soln: List[int], env: Environment) -> bool:
    soln_state: State = state
    move: int
    for move in soln:
        soln_state = env.next_state([soln_state], move)[0][0]

    return env.is_solved([soln_state])[0]


def bellman(states: List, curr_h_fn, target_h_fn, env: Environment) -> Tuple[np.ndarray, List[np.ndarray], List[List[State]]]:
    # expand states
    states_exp, tc_l = env.expand(states)
    tc = np.concatenate(tc_l, axis=0)

    # Get cost-to-go of expanded states using current heuristic function
    states_exp_flat, split_idxs = misc_utils.flatten(states_exp)
    curr_ctg_next: np.ndarray = curr_h_fn(states_exp_flat)

    # Get the indices (actions) of the minimum heuristic values
    curr_ctg_next_p_tc = tc + curr_ctg_next
    curr_ctg_next_p_tc_l = np.split(curr_ctg_next_p_tc, split_idxs)
    min_indices = [np.argmin(ctg) for ctg in curr_ctg_next_p_tc_l]

    # Evaluate these actions using the target heuristic function
    best_actions = [states_exp_flat[action] for action in min_indices]
    target_ctg_next = target_h_fn(best_actions)

    # Backup cost-to-go
    is_solved = env.is_solved(states)
    ctg_backup = np.array([tc[idx] + target_ctg_next[i] for i, idx in enumerate(min_indices)]) * np.logical_not(is_solved)

    return ctg_backup, target_ctg_next, states_exp
