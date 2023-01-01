"""Make a dataset file for cost model training.

Usage:
python3 make_dataset.py dataset/measure_records/t4/*.json
python3 make_dataset.py dataset/measure_records/t4/*.json --sample-in-files 100
"""
import argparse
import glob
import random

from tenset_cost_model.dataset import make_dataset_from_log_file
from common import load_and_register_tasks, get_measure_record_filename
import tvm 
from utils.device_info import query_cc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("logs", nargs="+", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--data-size", type=int, default=None)
    parser.add_argument("--sample-in-files", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-file", type=str, default='dataset.pkl')
    parser.add_argument("--min-sample-size", type=int, default=48)
    args = parser.parse_args()

    load_and_register_tasks()

    # files = args.logs
    # fix, instead of read in samples, read files under the folder
    assert args.data_dir, 'data dir is required.'
    if args.data_dir is not None:
        files = glob.glob(f"{args.data_dir}/*.json")
        # files = next(walk(args.data_dir), (None, None, []))[2]
        if args.data_size:
            files = files[:args.data_size]

    if args.sample_in_files:
        random.seed(args.seed)
        files = random.sample(files, args.sample_in_files) # task -> multi records, chose records
    
    # files record is not consistent, mapping and record
    # TODO: may not be the optimal solution, we first solve the problem in a straight and naive way
    class Obj: pass
    task = Obj()
    device = files[0].split('/')[-2] # t4
    arch = query_cc(device.upper())
    target = tvm.target.cuda(arch=f'sm_{arch}', model=device) # create t4 tvm cuda target
    # print(target.kind, target.model)
    new_files = []
    for file in files:
        file_name = file.split('/')[-1]
        workload_key, kind = file_name.split('],')
        workload_key, kind = workload_key[1:] + ']', kind.split(')')[0]
        task.workload_key = workload_key
        new_file = get_measure_record_filename(task, target)
        new_files.append(new_file)
    files = new_files

    print("Number of Task: ", len(files))
    make_dataset_from_log_file(files, args.out_file, args.min_sample_size)
    print('DataSet Generated')
