from typing import List, Tuple
import numpy as np
from utils import nnet_utils, misc_utils
from environments.environment_abstract import Environment, State
from search_methods.gbfs import GBFS
from search_methods.astar import AStar, Node
from torch.multiprocessing import Queue, get_context
import time
import random


def gbfs_update(states: List[State], env: Environment, num_steps: int, target_heuristic_fn, curr_heuristic_fn, eps_max: float, double_update=False):
    eps: List[float] = list(np.random.rand(len(states)) * eps_max)

    gbfs = GBFS(states, env, eps=eps)
    for _ in range(num_steps):
        gbfs.step(target_heuristic_fn, curr_heuristic_fn, double_update=double_update)

    trajs: List[List[Tuple[State, float]]] = gbfs.get_trajs()

    trajs_flat: List[Tuple[State, float]]
    trajs_flat, _ = misc_utils.flatten(trajs)

    is_solved: np.ndarray = np.array(gbfs.get_is_solved())

    states_update: List = []
    cost_to_go_update_l: List[float] = []
    for traj in trajs_flat:
        states_update.append(traj[0])
        cost_to_go_update_l.append(traj[1])

    cost_to_go_update = np.array(cost_to_go_update_l)

    return states_update, cost_to_go_update, is_solved


def astar_update(states: List[State], env: Environment, num_steps: int, heuristic_fn):
    weights: List[float] = list(np.random.rand(len(states)))
    astar = AStar(states, env, heuristic_fn, weights)
    for _ in range(num_steps):
        astar.step(heuristic_fn, 1, verbose=False)

    nodes_popped: List[List[Node]] = astar.get_popped_nodes()
    nodes_popped_flat: List[Node]
    nodes_popped_flat, _ = misc_utils.flatten(nodes_popped)

    for node in nodes_popped_flat:
        node.compute_bellman()

    states_update: List[State] = [node.state for node in nodes_popped_flat]
    cost_to_go_update: np.array = np.array([node.bellman for node in nodes_popped_flat])

    is_solved: np.array = np.array(astar.has_found_goal())

    return states_update, cost_to_go_update, is_solved


def update_runner(num_states: int, back_max: int, update_batch_size: int, heur_fn_i_q, heur_fn_o_q,
                  proc_id: int, env: Environment, result_queue: Queue, num_steps: int, update_method: str,
                  eps_max: float, seed, double_update=False):
    curr_heuristic_fn = nnet_utils.heuristic_fn_queue(heur_fn_i_q, heur_fn_o_q, proc_id, env) if double_update else None
    target_heuristic_fn = nnet_utils.heuristic_fn_queue(heur_fn_i_q, heur_fn_o_q, proc_id, env, use_target=True)

    # print(f"Starting update_runner process {proc_id} seed {seed}")

    np.random.seed(seed)
    random.seed(seed)

    start_idx: int = 0
    while start_idx < num_states:
        end_idx: int = min(start_idx + update_batch_size, num_states)

        # print(f"update_runner: proc_id: {proc_id}. start_idx: {start_idx}. end_idx: {end_idx}. num_states: {num_states}.")

        states_itr, _ = env.generate_states(end_idx - start_idx, (0, back_max))

        # print(f"Generated states in update_runner proc_id: {proc_id}.\nStates: {states_itr}")

        if update_method.upper() == "GBFS":
            states_update, cost_to_go_update, is_solved = gbfs_update(states_itr, env, num_steps, target_heuristic_fn, curr_heuristic_fn, eps_max, double_update=double_update)
        elif update_method.upper() == "ASTAR":
            states_update, cost_to_go_update, is_solved = astar_update(states_itr, env, num_steps, target_heuristic_fn)
        else:
            raise ValueError("Unknown update method %s" % update_method)

        states_update_nnet: List[np.ndaray] = env.state_to_nnet_input(states_update)

        result_queue.put((states_update_nnet, cost_to_go_update, is_solved))

        start_idx: int = end_idx

    result_queue.put(None)


class Updater:
    def __init__(self, env: Environment, num_states: int, back_max: int, heur_fn_i_q, heur_fn_o_qs,
                 num_steps: int, update_method: str, update_batch_size: int = 1000, eps_max: float = 0.0, double_update=False):
        super().__init__()
        ctx = get_context("spawn")
        self.num_steps = num_steps
        num_procs = len(heur_fn_o_qs)

        # initialize queues
        self.result_queue: ctx.Queue = ctx.Queue()

        # num states per process
        num_states_per_proc: List[int] = misc_utils.split_evenly(num_states, num_procs)

        self.num_batches: int = int(np.ceil(np.array(num_states_per_proc)/update_batch_size).sum())

        # initialize processes
        self.procs: List[ctx.Process] = []
        for proc_id in range(len(heur_fn_o_qs)):
            num_states_proc: int = num_states_per_proc[proc_id]
            if num_states_proc == 0:
                continue

            # Generate a unique seed for each process
            seed = random.randint(0, 2 ** 32 - 1)

            proc = ctx.Process(target=update_runner, args=(num_states_proc, back_max, update_batch_size,
                                                           heur_fn_i_q, heur_fn_o_qs[proc_id], proc_id, env,
                                                           self.result_queue, num_steps, update_method, eps_max, seed, double_update))
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

    def update(self):
        states_update_nnet: List[np.ndarray]
        cost_to_go_update: np.ndarray
        is_solved: np.ndarray
        states_update_nnet, cost_to_go_update, is_solved = self._update()

        output_update = np.expand_dims(cost_to_go_update, 1)

        return states_update_nnet, output_update, is_solved

    def _update(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        # process results
        states_update_nnet_l: List[List[np.ndarray]] = []
        cost_to_go_update_l: List = []
        is_solved_l: List = []

        none_count: int = 0
        result_count: int = 0
        display_counts: List[int] = list(np.linspace(1, self.num_batches, 10, dtype=np.int))

        start_time = time.time()

        while none_count < len(self.procs):
            result = self.result_queue.get()
            if result is None:
                none_count += 1
                continue
            result_count += 1

            states_nnet_q: List[np.ndarray]
            states_nnet_q, cost_to_go_q, is_solved_q = result
            states_update_nnet_l.append(states_nnet_q)

            cost_to_go_update_l.append(cost_to_go_q)
            is_solved_l.append(is_solved_q)

            if result_count in display_counts:
                print("%.2f%% (Total time: %.2f)" % (100 * result_count/self.num_batches, time.time() - start_time))

        num_states_nnet_np: int = len(states_update_nnet_l[0])
        states_update_nnet: List[np.ndarray] = []
        for np_idx in range(num_states_nnet_np):
            states_nnet_idx: np.ndarray = np.concatenate([x[np_idx] for x in states_update_nnet_l], axis=0)
            states_update_nnet.append(states_nnet_idx)

        cost_to_go_update: np.ndarray = np.concatenate(cost_to_go_update_l, axis=0)
        is_solved: np.ndarray = np.concatenate(is_solved_l, axis=0)

        for proc in self.procs:
            proc.join()

        return states_update_nnet, cost_to_go_update, is_solved
