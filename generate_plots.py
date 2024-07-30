import pickle
from collections import defaultdict

from matplotlib import pyplot as plt
import os
from scipy.signal import savgol_filter


def print_results(res_path: str):
    with open(res_path, 'rb') as file:
        data = pickle.load(file)

    print(data)


def compare_state_dict():
    import torch

    model_path1 = "saved_models/cube3/current/model_state_dict_180.0.pt"
    model_path2 = "saved_models/cube3/current/model_state_dict_160.0.pt"

    state_dict1 = torch.load(model_path1, map_location=torch.device('cpu'))
    state_dict2 = torch.load(model_path2, map_location=torch.device('cpu'))

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    # Check if they have the same layers
    if keys1 != keys2:
        print("State dictionaries have different layers")
        print("Layers in model 1 but not in model 2:", keys1 - keys2)
        print("Layers in model 2 but not in model 1:", keys2 - keys1)
        return

    for key in keys1:
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        if not torch.equal(param1, param2):
            print(f"Difference found in layer: {key}")
            print(f"Max difference: {torch.max(torch.abs(param1 - param2))}")
        else:
            print(f"No difference found in layer: {key}")


def plot_metrics(all_results, results_dir, close_plots=True):
    if len(all_results) > 1:
        res1, res2 = all_results[0], all_results[1]
    else:
        res1, res2 = all_results[0], all_results[0]

    if close_plots:
        results_dir = os.path.join(results_dir, "close_plots")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    title_dict = {
        "steps": r"($\times 10^3$)",
        "average_time_taken": "Average Solution Time (sec)",
        "average_solution_length": "Average Solution Length (moves)",
        "average_generated_nodes": r"Average Generated Nodes ($\times 10^6$)",
        "percentage_solved": "Problems Solved (%)"
    }

    steps1 = res1.get("steps", [])
    steps2 = res2.get("steps", [])

    metrics1 = {key: values for key, values in res1.items() if key != "steps"}
    metrics2 = {key: values for key, values in res2.items() if key != "steps"}

    common_keys = set(metrics1.keys()).intersection(set(metrics2.keys()))

    for key in common_keys:
        values1 = metrics1[key]
        values2 = metrics2[key]

        plt.figure()

        # smoothed_values1 = savgol_filter(values1, window_length=len(values1), polyorder=2)
        # smoothed_values2 = savgol_filter(values2, window_length=len(values2), polyorder=2)

        plt.plot(steps1, values1, marker='o', label='DeepCubeA')
        # plt.plot(steps1, smoothed_values1, label='Smoothed DCA')

        plt.plot(steps2, values2, marker='o', label='DoubleDeepCubeA')
        # plt.plot(steps2, smoothed_values2, label='Smoothed DoubleDCA')

        title = title_dict[key]

        plt.title(f"{title[:title.index('(')]}VS Steps")
        plt.xlabel(title_dict['steps'])
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)

        if not close_plots:
            max_step = max(max(steps1), max(steps2))
            max_value = max(max(values1), max(values2))

            plt.xlim(0, max_step * 1.5)
            plt.ylim(0, max_value * 1.5)

        save_file_location = os.path.join(results_dir, f'{key}_vs_steps_comparison.png')
        plt.savefig(save_file_location)
        print(f"Saved {title} vs Steps comparison plot in {save_file_location}")

        plt.close()


def plot_results_folder():
    par_dir = "results/cube3/2707"
    directories = [f"{par_dir}/res-old", f"{par_dir}/res-new"]
    res = []

    for directory in directories:
        plot_results = defaultdict(list)
        sorted_files = sorted([f for f in os.listdir(directory) if f.endswith('.pkl')],
                              key=lambda x: int(x.split('-')[-1].split('.')[0]))

        for filename in sorted_files:
            if filename.endswith('.pkl'):
                print(f"Processing {filename}")
                file_path = os.path.join(directory, filename)
                checkpoint_iter_n = filename.split('-')[-1].split('.')[0]

                # Open and load the pkl file
                with open(file_path, 'rb') as file:
                    results = pickle.load(file)

                solved = len(results["solutions"]) != 0

                if solved:
                    plot_results["steps"].append(int(checkpoint_iter_n))

                    plot_results["average_solution_length"].append(
                        sum([len(sol) for sol in results["solutions"]]) / len(results["solutions"])
                    )

                    plot_results["average_time_taken"].append(
                        sum(results["times"]) / len(results["times"])
                    )

                    plot_results["average_generated_nodes"].append(
                        sum(results["num_nodes_generated"]) / len(results["num_nodes_generated"])
                    )

                    plot_results["percentage_solved"].append(
                        100 * len(results["num_nodes_generated"]) / len(results["states"])
                    )

        res.append(plot_results)

    plot_metrics(res, directories[1])


if __name__ == "__main__":
    # print_results("data/cube3/test/data_0.pkl")
    plot_results_folder()
    # compare_state_dict()

