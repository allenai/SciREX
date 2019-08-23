default = {"BERT_FINE_TUNE": "none", "USE_LSTM": "true", "BALANCING_STRATEGY" : "none"}

experiments = [
    {"BERT_FINE_TUNE": "none", "USE_LSTM": "true"},
    # {"BERT_FINE_TUNE": "11", "USE_LSTM": "false"},
    # {"BERT_FINE_TUNE": "all", "USE_LSTM": "false"},
    # {"BALANCING_STRATEGY": "class_weight"},
    # {"BALANCING_STRATEGY": "sample"},
    # {"BERT_FINE_TUNE": "11,10,9", "USE_LSTM": "false"},
]

import subprocess

from copy import deepcopy
import argparse


def run_experiment(experiment, dry_run):
    env_vars = deepcopy(default)
    env_vars.update(experiment)

    print(env_vars)

    config_file = 'dygie/commands/train_pwc_crf.sh'
    exp_name = 'crf--relation--' + "--".join([k + '-' + v.replace(',', '.') for k, v in env_vars.items()])

    cmd = " ".join(
        [
            "python",
            "runtime-scripts/run_beaker_experiment.py",
            f"{config_file}",
            "--gpu-count 1",
            f"--name {exp_name}",
            f"--spec_output_path {'spec_' + exp_name + '.yaml'}", 
        ] + ['--env ' + k + "=" + v for k, v in env_vars.items()] + (["--dry-run"] if dry_run else [])
    )
    completed = subprocess.run(cmd, shell=True)
    print(f"returncode: {completed.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dry-run', action='store_true', help='If specified, an experiment will not be created.')
    args = parser.parse_args()

    for e in experiments :
        run_experiment(e, args.dry_run)
