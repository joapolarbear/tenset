"""Print programs in a measurement record file.

Usage:
# Print all programs
python3 print_programs.py --filename 'dataset/measure_records/e5-2673/([12b88bedece6984af589a28b43e0f3c4,1,56,56,64,3,3,64,128,1,1,1,128,1,28,28,128],llvm).json'
# Print a specific program
python3 print_programs.py --filename 'dataset/measure_records/e5-2673/([12b88bedece6984af589a28b43e0f3c4,1,56,56,64,3,3,64,128,1,1,1,128,1,28,28,128],llvm).json' --idx 31
"""

import argparse

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler import ComputeDAG, LayoutRewriteOption
from tvm.auto_scheduler.measure import recover_measure_input
from tvm.auto_scheduler.workload_registry import workload_key_to_tensors

from common import load_and_register_tasks

# import os
# tasks = load_and_register_tasks()
# dirname = "dataset/measure_records/t4"
# filename = os.path.join(dirname, os.listdir(dirname)[0])
# os.path.exists(filename)
# inputs, results = auto_scheduler.RecordReader(filename).read_lines()
# inp = inputs[0]
# res = results[0]
# inp = recover_measure_input(inp, True)

def print_program(index, inp, res):
    inp = recover_measure_input(inp, True)
    # import code
    # code.interact(local=locals())
    print("=" * 60)
    print(f"Index: {index}")
    print(f"Time cost (second): {res.costs}")
    print("Program:")
    print(inp.state)
    for idx, stage in enumerate(inp.state.stages):
        print(type(stage.op_type))
        raise
        print(f"\nStage {idx}: op_type={stage.op_type}")
        print(f"    op_name={stage.op.name}")
        if not isinstance(stage.op, tvm.te.tensor.PlaceholderOp):
            print(f"    op_body={stage.op.body}")
            print(f"    op_axis={stage.op.axis}")
            for iter in stage.iters:
                '''
                # Static trans table for thread bind and annotation
                # This is used to transform the annotation name to C++ enum
                ANNOTATION_TRANS_TABLE = {
                    "none": 0,
                    "unroll": 1,
                    "vectorize": 2,
                    "parallel": 3,
                    "vthread": 4,
                    "blockIdx.x": 5,
                    "threadIdx.x": 6,
                    "blockIdx.y": 7,
                    "threadIdx.y": 8,
                    "blockIdx.z": 9,
                    "threadIdx.z": 10,
                    "tensorize": 11,
                }

                iter.annotation IteratorAnnotation
                '''
                print(f"        name={iter.name}, kind={iter.iter_kind}, anot={iter.annotation}, range={iter.range}, extent={iter.range.extent}")

        # stage.op.input_tensors
        # stage.op.num_outputs
              
        # stage.op.tag
        # stage.op.attrs
        # stage.op.reduce_axis
        # stage.op.handle
        # stage.op.output()
        # stage.op.same_as()

    # print(AsText(inp.state, False, None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--idx", type=int)
    args = parser.parse_args()

    print("Load tasks...")
    tasks = load_and_register_tasks()

    inputs, results = auto_scheduler.RecordReader(args.filename).read_lines()
    if args.idx is None:
        for i in range(len(inputs)):
            print_program(i, inputs[i], results[i])
    else:
        print_program(args.idx, inputs[args.idx], results[args.idx])

