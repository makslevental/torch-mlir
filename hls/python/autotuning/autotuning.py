# noinspection PyUnresolvedReferences
import errno
import os
import re
import json
import shutil
import subprocess
import sys
from os import mkdir
from pathlib import Path
from subprocess import STDOUT, PIPE

import optuna

from hls.python.autotuning.parse_csynth_report import parse_report


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else:
            raise


'-affine-data-copy-generate=" fast-mem-capacity={FAST_MEM_CAPACITY}  fast-mem-space={FAST_MEM_SPACE}  generate-dma min-dma-transfer={MIN_DMA_TRANSFER}  slow-mem-space={SLOW_MEM_SPACE}  skip-non-unit-stride-loops tag-mem-space={TAG_MEM_SPACE} "'
'-affine-loop-unroll=" unroll-factor={UNROLL_FACTOR}  unroll-up-to-factor unroll-full unroll-num-reps={UNROLL_NUM_REPS}  unroll-full-threshold={UNROLL_FULL_THRESHOLD} "'
'-affine-loop-fusion=" fusion-compute-tolerance={FUSION_COMPUTE_TOLERANCE}  fusion-fast-mem-space={FUSION_FAST_MEM_SPACE}  fusion-local-buf-threshold={FUSION_LOCAL_BUF_THRESHOLD}  fusion-maximal mode={MODE} "'
'-affine-loop-tile=" cache-size={CACHE_SIZE}  separate tile-size={TILE_SIZE}  tile-sizes={TILE_SIZES} "'
'-affine-loop-unroll-jam=" unroll-jam-factor={UNROLL_JAM_FACTOR} "'
'-affine-parallelize=" max-nested={MAX_NESTED}  parallel-reductions"'
'-affine-super-vectorize=" virtual-vector-size={VIRTUAL_VECTOR_SIZE}  test-fastest-varying={TEST_FASTEST_VARYING}  vectorize-reductions"'

LOOP_UNROLL = (
    '-affine-loop-unroll=" unroll-factor={UNROLL_FACTOR}  unroll-up-to-factor=true"'
)
LOWER_PASSES = ' -lower-affine -convert-scf-to-cf -convert-memref-to-llvm -convert-arith-to-llvm -convert-std-to-llvm="use-bare-ptr-memref-call-conv=1" -reconcile-unrealized-casts'

MLIR_OPT_BIN_FP = (
    "/home/mlevental/dev_projects/torch-mlir/cmake-build-debug/bin/mlir-opt"
)
MLIR_TRANSLATE_BIN_FP = (
    "/home/mlevental/dev_projects/torch-mlir/cmake-build-debug/bin/mlir-translate"
)
TORCH_MLIR_OPT_BIN_FP = (
    "/home/mlevental/dev_projects/torch-mlir/cmake-build-debug/bin/torch-mlir-opt"
)
LLVM_OPT_BIN_FP = "/home/mlevental/dev_projects/torch-mlir/cmake-build-debug/bin/opt"
VHLS_SO = (
    "/home/mlevental/dev_projects/torch-mlir/cmake-build-debug/lib/VhlsLLVMRewriter.so"
)

STARTING_MLIR_FILE = (
    "/home/mlevental/dev_projects/torch-mlir/hls/scripts/braggnn.affine.baseline.mlir"
)
BRAGGNN_FP = "/home/mlevental/dev_projects/torch-mlir/hls/scripts/braggnn.affine.mlir"
LLVM_MLIR_FP = (
    "/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.llvm.mlir"
)
LL_FILE = "/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.ll"
OPT_LL_FILE = "/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.opt.vitis.ll"
WRAPPER_CPP_FILE = "/home/mlevental/dev_projects/torch-mlir/hls/scripts/wrapper.cpp.fmt"
WRAPPER_CPP_VITIS_FILE = (
    "/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/wrapper.cpp"
)
VITIS_SH_FILE = "/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/run_vitis.sh"


def call_bin_and_write(cmd, OUT_FP):
    print(cmd, ">", OUT_FP, file=sys.stderr)
    out = subprocess.run(cmd, capture_output=True, shell=True)
    out_decoded = out.stdout.decode()
    open(OUT_FP, "w").write(out_decoded)


def objective(trial):
    orig_flags = [
        # "-affine-loop-coalescing",
        '-affine-loop-fusion="fusion-compute-tolerance={FUSION-COMPUTE-TOLERANCE} fusion-maximal={FUSION-MAXIMAL} mode={MODE}"',
        "-affine-loop-invariant-code-motion",
        "-affine-loop-normalize",
        # "-affine-loop-tile",
        '-affine-loop-unroll="unroll-factor={UNROLL-FACTOR} unroll-up-to-factor={UNROLL-UP-TO-FACTOR} unroll-full={UNROLL-FULL}"',
        '-affine-parallelize="max-nested={MAX-NESTED} parallel-reductions={PARALLEL-REDUCTIONS}"',
        "-affine-scalrep",
        "-affine-simplify-structures",
    ]

    flags = set()
    for i in range(100):
        flag = trial.suggest_categorical(f"flag{i}", list(orig_flags))
        if flag in flags:
            continue
        if "loop-fusion" in flag:
            flag = flag.format(
                **{
                    "MODE": trial.suggest_categorical(
                        "mode",
                        ["sibling"]
                        # ["greedy", "producer", "sibling"]
                    ),
                    "FUSION-COMPUTE-TOLERANCE": round(
                        trial.suggest_float(
                            "fusion-compute-tolerance", 0.2, 0.8, step=0.1
                        ),
                        2,
                    ),
                    "FUSION-MAXIMAL": trial.suggest_categorical(
                        "fusion-maximal", ["true", "false"]
                    ),
                }
            )
        if "loop-unroll" in flag:
            flag = flag.format(
                **{
                    "UNROLL-FACTOR": trial.suggest_int("unroll-factor", 1, 10),
                    "UNROLL-UP-TO-FACTOR": trial.suggest_categorical(
                        "unroll-up-to-factor", ["true", "false"]
                    ),
                    "UNROLL-FULL": trial.suggest_categorical(
                        "unroll-full", ["true", "false"]
                    ),
                }
            )
        if "parallelize" in flag:
            flag = flag.format(
                **{
                    "MAX-NESTED": trial.suggest_int("max-nested", 1, 10),
                    "PARALLEL-REDUCTIONS": trial.suggest_categorical(
                        "parallel-reductions", ["true", "false"]
                    ),
                }
            )
        flags.add(flag)

        if len(flags) == 4:
            break

    try:

        cmd = " ".join([MLIR_OPT_BIN_FP, STARTING_MLIR_FILE] + list(flags))
        call_bin_and_write(cmd, BRAGGNN_FP)

        replace_me = open(BRAGGNN_FP).read().replace("alloc(", "alloca(")
        open(BRAGGNN_FP, "w").write(replace_me)


        in_batch, in_channel, in_height, in_width, out_batch, out_channel = re.search(
            r"func @forward\(%.*: memref<(\d+)x(\d+)x(\d+)x(\d+)x.*>, %.*: memref<(\d+)x(\d+)x.*>\)",
            replace_me,
        ).groups()
        replace_me = open(WRAPPER_CPP_FILE).read()
        replacements = {
            "IN_BATCH": in_batch,
            "IN_CHANNEL": in_channel,
            "IN_HEIGHT": in_height,
            "IN_WIDTH": in_width,
            "OUT_BATCH": out_batch,
            "OUT_CHANNEL": out_channel,
            "DTYPE": "int8_t",
        }
        for k, v in replacements.items():
            replace_me = replace_me.replace(f"XXX_{k}_XXX", v)
        open(WRAPPER_CPP_VITIS_FILE, "w").write(replace_me)

        # cmd = " ".join(
        #     [
        #         TORCH_MLIR_OPT_BIN_FP,
        #         BRAGGNN_FP,
        #         "-lower-affine",
        #         # "-torch-hls-promote-allocs",
        #     ]
        # )
        # call_bin_and_write(cmd, BRAGGNN_FP)

        cmd = " ".join([MLIR_OPT_BIN_FP, BRAGGNN_FP, LOWER_PASSES])
        call_bin_and_write(cmd, LLVM_MLIR_FP)

        cmd = " ".join([MLIR_TRANSLATE_BIN_FP, LLVM_MLIR_FP, "-mlir-to-llvmir"])
        call_bin_and_write(cmd, LL_FILE)

        cmd = " ".join(
            [
                LLVM_OPT_BIN_FP,
                LL_FILE,
                f"-S -enable-new-pm=0 -load {VHLS_SO} -mem2arr -strip-debug -instcombine -xlnmath -xlnname -strip-attr",
            ]
        )
        call_bin_and_write(cmd, OPT_LL_FILE)

        if os.path.exists(f"logs/{trial.number}"):
            shutil.rmtree(f"logs/{trial.number}")
        os.makedirs(f"logs/{trial.number}")
        open(f"logs/{trial.number}/flags.txt", "w").write(" ".join(list(flags)))
        shutil.copy(BRAGGNN_FP, f"logs/{trial.number}/braggnn.mlir")

        out = subprocess.run(
            "/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/run_vitis.sh",
            shell=True,
            executable="/bin/bash",
            stdout=PIPE,
            stderr=STDOUT,
        )
        open(f"logs/{trial.number}/vitis.log", "w").write(out.stdout.decode())
        copyanything(
            "/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/proj/solution1/syn/report",
            f"logs/{trial.number}/report",
        )

        path = Path("/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff")
        res = parse_report(path, "proj", "forward")
    except Exception as e:
        print(e)
        raise optuna.TrialPruned()

    error = res["avg_latency"]

    return error


#
# study = optuna.create_study()  # Create a new study.
# study.optimize(objective, n_trials=100)
# -affine-loop-fusion="fusion-maximal mode=sibling"


def parse_passes(affine_json, pass_name):
    passs = affine_json[pass_name]
    pass_info = {
        "pass_name": pass_name,
        "argument": passs["argument"],
        "summary": passs["summary"],
        "options": {},
    }
    pass_options = [affine_json[opt["def"]] for opt in passs["options"]]
    for pass_option in pass_options:
        opt_name = pass_option["cppName"]
        if pass_option["additionalOptFlags"] and "enum" in pass_option["type"]:
            pass_type = "enum"
            additional_opts = list(
                map(
                    lambda x: x.replace('"', "").strip(),
                    pass_option["additionalOptFlags"].split(",")[1::3],
                )
            )
        else:
            pass_type = pass_option["type"]
            additional_opts = pass_option["additionalOptFlags"]

        pass_info["options"][opt_name] = {
            "argument": pass_option["argument"],
            "defaultValue": pass_option["defaultValue"],
            "description": pass_option["description"],
            "type": pass_type,
            "additionalOptFlags": additional_opts,
        }

    return pass_info


def parse_affine_passes(affine_passes_json_fp):
    affine_passes = json.load(open(affine_passes_json_fp))
    passes = {}
    passes["AffineDataCopyGeneration"] = parse_passes(
        affine_passes, "AffineDataCopyGeneration"
    )
    passes["AffineLoopUnroll"] = parse_passes(affine_passes, "AffineLoopUnroll")
    passes["AffineLoopFusion"] = parse_passes(affine_passes, "AffineLoopFusion")
    passes["AffineLoopTiling"] = parse_passes(affine_passes, "AffineLoopTiling")
    passes["AffineLoopUnrollAndJam"] = parse_passes(
        affine_passes, "AffineLoopUnrollAndJam"
    )
    passes["AffineParallelize"] = parse_passes(affine_passes, "AffineParallelize")
    passes["AffineVectorize"] = parse_passes(affine_passes, "AffineVectorize")

    json.dump(passes, open("affine_pass_info.json", "w"), indent=2)


def make_pass_opts_cmd_line(pass_json, pass_name, option_names=None, values=None):
    casts = {
        "bool": lambda x: True if x == "true" else False,
        "unsigned": int,
        "uint64_t": int,
        "double": float,
        "enum": lambda x: x[0],
    }

    pass_info = pass_json[pass_name]
    if option_names is None:
        option_names = pass_info["options"].keys()

    cmd_line = f"-{pass_info['argument']}=\""
    for option_name in option_names:
        option = pass_info["options"][option_name]
        argument = option["argument"]
        cmd_line += f" {argument}"
        if option["type"] != "bool":
            cmd_line += f'={{{argument.replace("-", "_").upper()}}} '
    cmd_line += '"'

    print(cmd_line)


def make_affine_pass_opts_cmd_lines(affine_pass_info_fp):
    affine_pass_info_json = json.load(open(affine_pass_info_fp))
    make_pass_opts_cmd_line(affine_pass_info_json, "AffineDataCopyGeneration")
    make_pass_opts_cmd_line(affine_pass_info_json, "AffineLoopUnroll")
    make_pass_opts_cmd_line(affine_pass_info_json, "AffineLoopFusion")
    make_pass_opts_cmd_line(affine_pass_info_json, "AffineLoopTiling")
    make_pass_opts_cmd_line(affine_pass_info_json, "AffineLoopUnrollAndJam")
    make_pass_opts_cmd_line(affine_pass_info_json, "AffineParallelize")
    make_pass_opts_cmd_line(affine_pass_info_json, "AffineVectorize")


if __name__ == "__main__":
    #  llvm-tblgen --dump-json -I/home/mlevental/dev_projects/torch-mlir/external/llvm-project/mlir/include /home/mlevental/dev_projects/torch-mlir/external/llvm-project/mlir/include/mlir/Dialect/Affine/Passes.td > affine_passes.json
    # parse_affine_passes("affine_passes.json")
    # make_affine_pass_opts_cmd_lines("affine_pass_info.json")
    # replace = open("/home/mlevental/dev_projects/torch-mlir/hls/scripts/braggnn.affine.baseline.mlir").read()
    # in_batch, in_channels, in_height, in_width, out_batch, out_channels = re.search(r"func @forward\(%.*: memref<(\d+)x(\d+)x(\d+)x(\d+)xf32>, %.*: memref<(\d+)x(\d+)xf32>\)", replace).groups()
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///mydb.db",
        load_if_exists=True,  # to skip creating a new study if it already exists.
    )  # Create a new study.
    study.optimize(objective, n_trials=100, n_jobs=1)
