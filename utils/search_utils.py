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


def bellman(states: List, heuristic_fn, env: Environment) -> Tuple[np.ndarray, List[np.ndarray], List[List[State]]]:
    # expand states
    states_exp, tc_l = env.expand(states)
    tc = np.concatenate(tc_l, axis=0)

    # get cost-to-go of expanded states
    states_exp_flat, split_idxs = misc_utils.flatten(states_exp)
    ctg_next: np.ndarray = heuristic_fn(states_exp_flat)

    # backup cost-to-go
    ctg_next_p_tc = tc + ctg_next
    ctg_next_p_tc_l = np.split(ctg_next_p_tc, split_idxs)

    is_solved = env.is_solved(states)
    ctg_backup = np.array([np.min(x) for x in ctg_next_p_tc_l]) * np.logical_not(is_solved)

    return ctg_backup, ctg_next_p_tc_l, states_exp


def double_bellman(states: List, curr_h_fn, target_h_fn, env: Environment) -> Tuple[np.ndarray, List[np.ndarray], List[List[State]]]:
    # expand states
    states_exp, tc_l = env.expand(states)
    tc = np.concatenate(tc_l, axis=0)

    # get cost-to-go of expanded states using current heuristic function
    states_exp_flat, split_idxs = misc_utils.flatten(states_exp)
    curr_ctg_next: np.ndarray = curr_h_fn(states_exp_flat)

    # get the indices (actions) of the minimum heuristic values
    curr_ctg_next_l = np.split(curr_ctg_next, split_idxs)
    min_actions = [np.argmin(ctg) for ctg in curr_ctg_next_l]

    # evaluate these actions using the target heuristic function
    best_next_states = [states_exp[idx][action] for idx, action in enumerate(min_actions)]
    target_ctg_next = target_h_fn(best_next_states)

    # add the cost of the edge
    tc_best_actions = np.array([tc_l[idx][action] for idx, action in enumerate(min_actions)])
    target_ctg_next_tc = tc_best_actions + target_ctg_next

    # Backup cost-to-go
    is_solved = env.is_solved(states)
    ctg_backup = np.array(target_ctg_next_tc) * np.logical_not(is_solved)

    return ctg_backup, best_next_states, states_exp
