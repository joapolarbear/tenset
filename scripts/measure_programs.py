"""Measure all programs

Usage:
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=e5-2666"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=e5-2673"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=epyc-7452"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=epyc-7r32"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=i7-8750h"
python3 measure_programs.py --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m"
python3 measure_programs.py --target "llvm -mcpu=skylake-avx512 -model=platinum-8272l"
python3 measure_programs.py --target "llvm -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod -model=graviton2"
python3 measure_programs.py --target "llvm -mtriple=aarch64-linux-gnu -mattr=+neon -model=a72" --other-args "--rpc-device-key rasp4b-64 --rpc-host kraken --rpc-port 9191 --rpc-n-parallel 4"

python3 measure_programs.py --target "cuda --model=t4"
"""
import argparse
import glob
import os
import pickle
import time
import json

from tqdm import tqdm
import numpy as np

import tvm
from tvm import auto_scheduler

from common import (
    load_and_register_tasks,
    get_measure_record_filename,
    get_test_measure_record_filename,
    get_to_measure_filename,
    get_compare_measure_record_filename
)

INVALID_TIME_UPPER = 1e10
PROGRESS_FILE = "progress.txt"

def make_measurer(run_timeout, repeat, number, enable_cpu_cache_flush,
                  verbose, log_filename):
    builder = auto_scheduler.measure.LocalBuilder()
    runner = auto_scheduler.measure.LocalRunner(
        timeout=run_timeout, repeat=repeat, number=number,
        enable_cpu_cache_flush=enable_cpu_cache_flush)
    measurer = auto_scheduler.measure.ProgramMeasurer(
        builder,
        runner,
        [auto_scheduler.RecordToFile(log_filename)],
        verbose=verbose,
    )
    return measurer


def remeasure_file(task_idx, task, target, target_host, batch_size, measurer_kwargs):
    raise ValueError("Do not allow to override existing measure_records")
    # Make folder and log filename
    target = tvm.target.Target(target)
    log_filename = get_measure_record_filename(task, target)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    # Make measure
    measurer_kwargs['log_filename'] = log_filename
    measurer = make_measurer(**measurer_kwargs)

    # Read reference measurement inputs
    to_measure_filename = get_to_measure_filename(task)
    inputs, _ = auto_scheduler.RecordReader(to_measure_filename).read_lines()
    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task
    task = auto_scheduler.SearchTask(
        workload_key=task.workload_key,
        target=target,
        target_host=target_host,
        hardware_params=task.hardware_params,
        layout_rewrite_option=task.layout_rewrite_option,
    )
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)

    # Do measurement
    for i in range(0, len(inputs), batch_size):
        print(f"===== task: {task_idx}\t programs: {i}/{len(inputs)} =====")
        inp_batch = []
        for inp in inputs[i:min(len(inputs), i + batch_size)]:
            inp_batch.append(auto_scheduler.MeasureInput(task, inp.state))
        res_batch = measurer.measure(task, empty_policy, inp_batch)

        timeout_ct = 0
        for res in res_batch:
            if res.error_no == auto_scheduler.measure.MeasureErrorNo.BUILD_TIMEOUT:
                timeout_ct += 1

def check_same_task(input1, input2):
    '''Check whether two inputs belong to the same states'''
    inp1 = auto_scheduler.measure.recover_measure_input(input1, True)
    inp2 = auto_scheduler.measure.recover_measure_input(input2, True)
    return inp1.state == inp2.state

def parse_cost(res, verbose=False):
    costs = list(res.costs)
    costs = [c.value for c in costs]
    if costs[0] == INVALID_TIME_UPPER:
        assert len(costs) == 1
        # print(costs, "Timeout during run")
        return None, None
    dur = np.mean(costs)
    std = np.std(costs)
    if verbose:
        print(f"Time cost (second): {res.costs}")
        # print("Program:")
        # print(inp.state)
        x = input()
    return dur, std

def remeasure_and_compare(task_idx, task, target, target_host, batch_size, measurer_kwargs):
    # Make folder and log filename
    target = tvm.target.Target(target)
    log_filename = get_test_measure_record_filename(task, target)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    ### Create a file to store the compare results
    compare_filename = get_compare_measure_record_filename(task, target)
    os.makedirs(os.path.dirname(compare_filename), exist_ok=True)

    ### Retrieve Tenset dataset corresponding to `task` and `target`
    tenset_log_filename = get_measure_record_filename(task, target)
    tenset_inputs, tenset_results = auto_scheduler.RecordReader(tenset_log_filename).read_lines()

    # Make measure
    measurer_kwargs['log_filename'] = log_filename
    measurer = make_measurer(**measurer_kwargs)

    # Read reference measurement inputs
    to_measure_filename = get_to_measure_filename(task)
    inputs, _ = auto_scheduler.RecordReader(to_measure_filename).read_lines()

    ### Check whether states in measure_records (tenset_inputs) and 
    # states in to_measure_programs (inputs) are in the same order
    if False:
        assert len(tenset_inputs) == len(inputs), (len(tenset_inputs), len(inputs))
        total_state_num = len(tenset_inputs)
        match_cnt = 0
        for i in range(total_state_num):
            if check_same_task(tenset_inputs[i], inputs[i]):
                match_cnt += 1
            if i > 1:
                assert not check_same_task(tenset_inputs[i-1], tenset_inputs[i]), f"Tenset task {task_idx}: states idx {i-1} and {i} are the same"
        print(f"Tenset task {task_idx}: There are {match_cnt}/{total_state_num} states are in the same order")
        raise

    ### Retrieve the task
    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task
    task = auto_scheduler.SearchTask(
        workload_key=task.workload_key,
        target=target,
        target_host=target_host,
        hardware_params=task.hardware_params,
        layout_rewrite_option=task.layout_rewrite_option,
    )
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)

    # Do measurement
    results = {"tenset": [], "measure": []}
    for i in range(0, len(inputs), batch_size):
        print(f"===== task: {task_idx}\t programs: {i}/{len(inputs)} =====")
        inp_batch = []
        for inp in inputs[i:min(len(inputs), i + batch_size)]:
            inp_batch.append(auto_scheduler.MeasureInput(task, inp.state))
        res_batch = measurer.measure(task, empty_policy, inp_batch)

        tenset_res_batch = tenset_results[i:min(len(inputs), i+batch_size)]

        for res_idx in range(len(res_batch)):
            tenset_dur_std = parse_cost(tenset_res_batch[res_idx])
            # print(f"Tenset cost: {tenset_dur_std[0]:.6f}\u00B1{tenset_dur_std[1]:.6f}s")
            results["tenset"].append(tenset_dur_std)
            if res_batch[res_idx].error_no == 0:
                dur_std = parse_cost(res_batch[res_idx])
                # print(f"Measured cost: {dur_std[0]:.6f}\u00B1{dur_std[1]:.6f}s")
                results["measure"].append(dur_std)
            else:
                # print(f"Measured cost: Error {res_batch[res_idx].error_no}")
                results["measure"].append((INVALID_TIME_UPPER, 0))
    with open(compare_filename, 'w') as fp:
        json.dump(results, fp, indent=4)

def comapre_analysis(task_idx, task, target):
    ### Create a file to store the compare results
    compare_filename = get_compare_measure_record_filename(task, target)
    os.makedirs(os.path.dirname(compare_filename), exist_ok=True)

    if not os.path.exists(compare_filename):
        return
    with open(compare_filename, 'r') as fp:
        results = json.load(fp)

    tenset_dur, tenset_std = zip(*results["tenset"])
    measure_dur, measure_std = zip(*results["measure"])

    tenset_dur, tenset_std = np.array(tenset_dur), np.array(tenset_std)
    measure_dur, measure_std = np.array(measure_dur), np.array(measure_std)
    assert len(measure_dur.shape) == len(tenset_dur.shape) \
        and len(measure_dur.shape) == 1 \
        and len(measure_dur) == len(tenset_dur), \
            (tenset_dur.shape, measure_dur.shape)

    ### Filter out invalid records
    indexes = np.where(np.logical_and(measure_dur!=INVALID_TIME_UPPER, tenset_dur != None))[0]
    tenset_dur, tenset_std = tenset_dur[indexes], tenset_std[indexes]
    measure_dur, measure_std = measure_dur[indexes], measure_std[indexes]

    ### Filter out unstalbe records
    std2avg_upper_bound = 0.01
    indexes = np.where(np.logical_and(tenset_std / tenset_dur < std2avg_upper_bound, 
                        measure_std / measure_dur < std2avg_upper_bound))[0]
    _tenset_dur = tenset_dur[indexes]
    _measure_dur = measure_dur[indexes]

    error = np.mean(np.abs(_measure_dur - _tenset_dur) / _measure_dur) * 100
    print(f"Task {task_idx}: Discrepancy between Tenset and Measure: {error:.3f}%")
    return error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, default="measure_compare",
        choices=["remeasure", "measure_compare", "analysis"])
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-idx", type=int, default=0, help="Start task index")
    parser.add_argument("--end-idx", type=int, default=1000000, help="End task index")
    parser.add_argument("--step-idx", type=int, default=1, help="Task interval to measure")
    args = parser.parse_args()

    # Load task registry
    print("Load all tasks...")
    tasks = load_and_register_tasks()

    end_idx = min(args.end_idx, len(tasks))

    error_list = []
    print(f"tasks: range(start={args.start_idx}, end={end_idx}, step={args.step_idx})")
    # Remeasure all tasks
    for i in range(args.start_idx, end_idx, args.step_idx):
        with open(PROGRESS_FILE, "a") as fout:
            fout.write(f"Begin {i}/{len(tasks)}: {time.time():.2f}\n")
        task = tasks[i]

        # Set measurement arguments
        measurer_kwargs = {
            "run_timeout": 15,
            "number": 2,
            "enable_cpu_cache_flush": (task.target.kind == "llvm"),
            "verbose": 1,
            "repeat": 2,
        }

        # if task.compute_dag.flop_ct >= 2416443392.0:
        #     measurer_kwargs['repeat'] = 4
        # elif task.compute_dag.flop_ct >= 834928640.0:
        #     measurer_kwargs['repeat'] = 6
        # elif task.compute_dag.flop_ct <= 2097152.0:
        #     measurer_kwargs['repeat'] = 10
        # else:
        #     measurer_kwargs['repeat'] = 8

        # Run measurement
        target = tvm.target.Target(args.target)
        if args.option == "remeasure":
            print(f"########## Task {i}, FLOPs = {task.compute_dag.flop_ct} ##########")
            # print(task.compute_dag)
            remeasure_file(i, task, target, args.target_host, args.batch_size, measurer_kwargs)
        elif args.option == "measure_compare":
            print(f"########## Task {i}, FLOPs = {task.compute_dag.flop_ct} ##########")
            # print(task.compute_dag)
            remeasure_and_compare(i, task, target, args.target_host, args.batch_size, measurer_kwargs)
            with open(PROGRESS_FILE, "a") as fout:
                fout.write(f"End {i}/{len(tasks)}: {time.time():.2f}\n")
        elif args.option == "analysis":
            error = comapre_analysis(i, task, target)
            if error:
                error_list.append(error)
        else:
            raise ValueError(f"Invalid option {args.option}")
    
    ### Summary
    if args.option == "analysis":
        error_list = np.array(error_list)
        print(f"Discrepancy between Tenset and Measure for {len(error_list)} tasks: "
            f"{np.mean(error_list):.3f}(\u00B1{np.std(error_list):.3f}) %")

        

