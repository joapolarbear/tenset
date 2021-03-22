"""Tune a network"""
import argparse
import logging
import random
import time

import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.contrib.graph_runtime as runtime
from tvm.auto_scheduler.utils import to_str_round

from dump_network_info import get_network_with_key
from common import str2bool, log_line, BenchmarkRecord


def get_network(network_args):
    name, batch_size = network_args['network'], network_args['batch_size']
    if name in ['resnet_18', 'resnet_50', 'mobilenet_v2', 'mobilenet_v3',
                'wide_resnet_50', 'resnext_50']:
        network_key = (name, [(batch_size, 3, 224, 224)])
    elif name in ['bert_tiny', 'bert_base', 'bert_medium', 'bert_large']:
        network_key = (name, [(batch_size, 128)])
    elif name == 'dcgan':
        network_key = (name, [(batch_size, 3, 64, 64)])
    else:
        raise ValueError("Invalid network: " + network)

    return get_network_with_key(network_key)


def get_tuning_option(tuning_args, target):
    n_trials, run_timeout, log_file = (
        tuning_args['n_trials'], tuning_args['run_timeout'], tuning_args['log_file'])

    if "cpu" in target.keys:
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=auto_scheduler.LocalRunner(
                timeout=run_timeout, repeat=10, number=1, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
    else:
        raise NotImplementedError

    return tuning_opt


def tune_and_evaluate(network_args, tuning_args, target, target_host, result_file):
    mod, params, inputs = get_network(network_args)

    # Do auto-tuning
    if not tuning_args['eval_only']:
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

        for idx, task in enumerate(tasks):
            print(
                "========== Task %d  (workload key: %s...) =========="
                % (idx, task.workload_key[:20])
            )
            print(task.compute_dag)

        tuning_opt = get_tuning_option(tuning_args, target)

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights,
            load_model_file=tuning_args['load_model'])
        policy = 'sketch.%s' % tuning_args['cost_model']
        tuner.tune(tuning_opt, search_policy=policy)

    # Build module
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(mod, target=target, params=params)
    ctx = tvm.context(str(target), 0)
    module = runtime.GraphModule(lib["default"](ctx))

    # Feed input data
    for name, shape, dtype in inputs:
        data_np = np.random.uniform(size=shape).astype(dtype)
        module.set_input(name, data_np)

    # Evaluate
    ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=3)
    prof_res = np.array(ftimer().results)
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res) * 1000, np.std(prof_res) * 1000))

    # Dump results
    log_line(BenchmarkRecord(str(target.kind), 'gpu' if 'gpu' in target.keys else 'cpu',
                             'network',
                             "%s.B%d" % (network_args['network'], network_args['batch_size']),
			     'ours', 'default',
                             {"costs": prof_res}, time.time()),
                             args.result_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Search task related arguments
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--eval-only", action='store_true')

    # Search strategy related arguments
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--cost-model", type=str, choices=['xgb', 'random', 'xgb-no-update'],
                        default='xgb', help="The type of program cost model")
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--load-model", type=str, help="Load pre trained cost model file")

    # Log file related arguments
    parser.add_argument("--log-file", type=str, help="Write measurement records to this log file")
    parser.add_argument("--result-file", type=str,
                        help="Save end-to-end latency to this file",
                        default="results.tsv")

    # Measurement related and other arguments
    parser.add_argument("--num-measure-per-iter", type=int, default=64,
                        help="The number of programs to be measured at each iteration")
    parser.add_argument("--build-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=25)
    parser.add_argument("--early-stopping", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    #logging.basicConfig()
    #logging.getLogger('auto_scheduler').setLevel(logging.DEBUG)

    target = tvm.target.Target(args.target)
    if target.model == "unknown":
        log_file = args.log_file or "%s-B%d-%s.json" % (args.network, args.batch_size,
                                                        target.kind)
    else:
        log_file = args.log_file or "%s-B%d-%s-%s.json" % (args.network, args.batch_size,
                                                        target.kind, target.model)

    network_args = {
        "network": args.network,
        "batch_size": args.batch_size,
    }

    tuning_args = {
        "eval_only": args.eval_only,
        "n_trials": args.n_trials,
        "log_file": log_file,
        "run_timeout": args.run_timeout,
        "cost_model": args.cost_model,
        "load_model": args.load_model,
    }

    tune_and_evaluate(network_args, tuning_args, target, args.target_host,
                      args.result_file)

