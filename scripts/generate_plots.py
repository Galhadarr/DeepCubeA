import os
import pickle
import subprocess
from argparse import ArgumentParser


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of nnet model")

    args = parser.parse_args()
    assert os.path.isdir(args.model_dir)

    results = {
        "solutions": [],
        "paths": [],
        "times": [],
        "num_nodes_generated": []
    }

    # Get list of model checkpoints
    checkpoints = [f for f in os.listdir(args.model_dir) if f.startswith('model_state_dict_') and f.endswith('.pt')]
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))  # Sort by iteration number

    for checkpoint in checkpoints:

        # Define the command to run the A* algorithm
        command = [
            'python', 'C:\\Users\\Or\\PycharmProjects\\DeepCubeA\\search_methods\\astar.py',
            '--states', 'C:\\Users\\Or\\PycharmProjects\\DeepCubeA\\data\\cube3\\test\\data_0.pkl',
            '--model_dir', args.model_dir,
            '--model_checkpoint', checkpoint,
            '--env', 'cube3',
            '--weight', '0.8',
            '--batch_size', '20000',
            '--results_dir', 'C:\\Users\\Or\\PycharmProjects\\DeepCubeA\\results\\cube3',
            '--nnet_batch_size', '10000',
        ]

        subprocess.run(command)

        # Read the results from the file
        results_file = 'C:\\Users\\Or\\PycharmProjects\\DeepCubeA\\results\\cube3\\results.pkl'
        with open(results_file, 'rb') as f:
            output_data = pickle.load(f)

        # Append the results
        results["solutions"].append(output_data.get("solutions", []))
        results["paths"].append(output_data.get("paths", []))
        results["times"].append(output_data.get("times", []))
        results["num_nodes_generated"].append(output_data.get("num_nodes_generated", []))

    return results



if __name__ == "__main__":
    main()
