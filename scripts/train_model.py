"""Train a cost model with a dataset."""

import argparse
import logging
import pickle
import random
import os

import torch
import numpy as np

import tvm
from tvm.auto_scheduler.utils import to_str_round
from tvm.auto_scheduler.cost_model import RandomModelInternal

from common import load_and_register_tasks, str2bool, get_task_info_filename
from tune_network import get_network
    
from tvm.auto_scheduler import SketchPolicy, extract_tasks
from tvm.auto_scheduler.feature import get_per_store_features_from_states
from tvm.auto_scheduler.dataset import Dataset, LearningTask
from tvm.auto_scheduler.cost_model.xgb_model import XGBModelInternal
from tvm.auto_scheduler.cost_model.mlp_model import MLPModelInternal
from tvm.auto_scheduler.cost_model.lgbm_model import LGBModelInternal
from tvm.auto_scheduler.cost_model.tabnet_model import TabNetModelInternal
from tvm.auto_scheduler.cost_model.metric import (
    metric_rmse,
    metric_r_squared,
    metric_pairwise_comp_accuracy,
    metric_top_k_recall,
    metric_peak_score,
    metric_mape,
    random_mix,
)

def estimate_end2end(mod, params, target, number, model):
    # Extract search tasks
    target = tvm.target.Target(target)
    tasks, task_weights = extract_tasks(mod["main"], params, target)

    # import tvm.auto_scheduler as auto_scheduler
    # log_file = "tmp.txt"
    # print("Begin tuning...")
    # tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    # tune_option = auto_scheduler.TuningOptions(
    #     num_measure_trials=100,  # change this to 20000 to achieve the best performance
    #     runner=auto_scheduler.LocalRunner(repeat=1, enable_cpu_cache_flush=True),
    #     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    # )
    # tuner.tune(tune_option)
    # raise

    rst = 0
    for task_id, task in enumerate(tasks):
        policy = SketchPolicy(task, verbose=0)
        states = policy.sample_initial_population()[:number]

        learning_task = LearningTask(task.workload_key, str(task.target))
        features = get_per_store_features_from_states(states, task)
        eval_dataset = Dataset.create_one_task(learning_task, features, None)
        ret = model.predict(eval_dataset)[learning_task]
        # Predict 0 for invalid states that failed to be lowered.
        # _ret = []
        # for idx, feature in enumerate(features):
        #     if feature.min() == feature.max() == 0:
        #         ret[idx] = float('-inf')
        #     else:
        #         _ret.append(ret[idx])
        _ret = ret
        assert len(_ret) > 0
        # print(np.mean(_ret), task_weights[task_id])
        rst += np.mean(_ret) * task_weights[task_id]

        # sch, args = task.compute_dag.apply_steps_from_state(state, task.layout_rewrite_option)
    
    return rst

def evaluate_model(model, test_set):
    # make prediction
    prediction = model.predict(test_set)

    # compute weighted average of metrics over all tasks
    tasks = list(test_set.tasks())
    weights = [len(test_set.throughputs[t]) for t in tasks]
    print("Test set sizes:", weights)

    rmse_list = []
    r_sqaured_list = []
    pair_acc_list = []
    mape_list = []
    mape_avg_list = []
    peak_score1_list = []
    peak_score5_list = []


    for task in tasks:
        
        ### Calculate flop_cnt
        file_name = get_task_info_filename(task.workload_key, tvm.target.Target(task.target))
        file_name = file_name.replace("network_info", "to_measure_programs").replace("task.pkl", "json")
        inputs, _ = tvm.auto_scheduler.RecordReader(file_name).read_lines()
        search_task = tvm.auto_scheduler.measure.recover_measure_input(inputs[0]).task
        flop_ct = search_task.compute_dag.flop_ct

        preds = prediction[task]
        labels = test_set.throughputs[task]

        rmse_list.append(np.square(metric_rmse(preds, labels)))
        r_sqaured_list.append(metric_r_squared(preds, labels))
        pair_acc_list.append(metric_pairwise_comp_accuracy(preds, labels))
        mape_list.append(metric_mape(preds, labels))
        mape_avg_list.append(metric_mape(flop_ct/preds, flop_ct/labels))
        peak_score1_list.append(metric_peak_score(preds, labels, 1))
        peak_score5_list.append(metric_peak_score(preds, labels, 5))

    rmse = np.sqrt(np.average(rmse_list, weights=weights))
    r_sqaured = np.average(r_sqaured_list, weights=weights)
    pair_acc = np.average(pair_acc_list, weights=weights)
    mape = np.average(mape_list, weights=weights)
    mape_avg = np.average(mape_avg_list, weights=weights)
    peak_score1 = np.average(peak_score1_list, weights=weights)
    peak_score5 = np.average(peak_score5_list, weights=weights)

    eval_res = {
        "RMSE": rmse,
        "R^2": r_sqaured,
        "pairwise comparision accuracy": pair_acc,
        "mape": mape,
        "mape_avg": mape_avg,
        "average peak score@1": peak_score1,
        "average peak score@5": peak_score5,
    }

    networks = ["resnet_50", "densenet_121"]
    bs_s = [1, 32]
    target = "cuda"
    for network in networks:
        for bs in bs_s:
            network_args = {
                "network": network,
                "batch_size": bs,
            }
            mod, params, inputs = get_network(network_args)
            end2end = estimate_end2end(mod, params, target, 1, model)
            eval_res[f"{network}-bs_{bs}"] = end2end
    return eval_res


def make_model(name, use_gpu=False):
    """Make model according to a name"""
    if name == "xgb":
        return XGBModelInternal(use_gpu=use_gpu)
    elif name == "mlp":
        return MLPModelInternal()
    elif name == 'lgbm':
        return LGBModelInternal(use_gpu=use_gpu)
    elif name == 'tab':
        return TabNetModelInternal(use_gpu=use_gpu)
    elif name == "random":
        return RandomModelInternal()
    else:
        raise ValueError("Invalid model: " + name)
 

def train_zero_shot(
    dataset,
    train_ratio,
    model_names,
    split_scheme,
    use_gpu,
    out_dir=".workspace/cm"):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Split dataset
    if split_scheme == "within_task":
        train_set, test_set = dataset.random_split_within_task(train_ratio)
    elif split_scheme == "by_task":
        train_set, test_set = dataset.random_split_by_task(train_ratio)
    elif split_scheme == "by_target":
        train_set, test_set = dataset.random_split_by_target(train_ratio)
    else:
        raise ValueError("Invalid split scheme: " + split_scheme)

    print("Train set: %d. Task 0 = %s" % (len(train_set), train_set.tasks()[0]))
    if len(test_set) == 0:
        test_set = train_set
    print("Test set:  %d. Task 0 = %s" % (len(test_set), test_set.tasks()[0]))

    # Make models
    names = model_names.split("@")
    models = []
    for name in names:
        models.append(make_model(name, use_gpu))

    eval_results = []
    for name, model in zip(names, models):
        # Train the model
        filename = os.path.join(out_dir, name + ".pkl")
        model.fit_base(train_set, valid_set=test_set)
        print("Save model to %s" % filename)
        model.save(filename)

        # Evaluate the model
        eval_res = evaluate_model(model, test_set)
        print(name, to_str_round(eval_res))
        eval_results.append(eval_res)

    # Print evaluation results
    for i in range(len(models)):
        print("-" * 60)
        print("Model: %s" % names[i])
        for key, val in eval_results[i].items():
            print("%s: %.4f" % (key, val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", type=str, default=["dataset.pkl"])
    parser.add_argument("--models", type=str, default="xgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-scheme",
        type=str,
        choices=["by_task", "within_task", "by_target"],
        default="within_task",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--use-gpu", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to use GPU for xgb.")
    parser.add_argument("--out-dir", type=str, default='.workspace/cm')
    args = parser.parse_args()
    print("Arguments: %s" % str(args))

    # Setup random seed and logging
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    logging.basicConfig()
    logging.getLogger("auto_scheduler").setLevel(logging.DEBUG)

    print("Load all tasks...")
    load_and_register_tasks()

    print("Load dataset...")
    dataset = pickle.load(open(args.dataset[0], "rb"))
    for i in range(1, len(args.dataset)):
        tmp_dataset = pickle.load(open(args.dataset[i], "rb"))
        dataset.update_from_dataset(tmp_dataset)

    train_zero_shot(
        dataset,
        args.train_ratio,
        args.models,
        args.split_scheme,
        args.use_gpu,
        args.out_dir)

'''
python3 train_model.py --train-ratio 0.95
python3 train_model.py \
    --train-ratio 0.95 \
    --models xgb@mlp@tab@random \
    --use-gpu \
    --dataset .workspace/dataset.pkl
python3 train_model.py \
    --train-ratio 0.95 \
    --models xgb \
    --use-gpu \
    --dataset .workspace/dataset.pkl
'''


