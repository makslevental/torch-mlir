import json
from pathlib import Path
from pprint import pprint


def to_int(s):
    if s == "-":
        return 0
    return int(s)


def to_float(s):
    if s == "-":
        return 0.0
    return float(s)


def parse_report(directory, proj_dir_name, kernel_prefix):
    directory = directory / proj_dir_name / "solution1"
    try:
        solution_data = json.load((directory / "solution1_data.json").open())
        metrics = solution_data["ModuleInfo"]["Metrics"][kernel_prefix]

        freq_estimate = to_float(metrics["Timing"]["Estimate"])  # ms

        return {
            "bram": to_int(metrics["Area"]["BRAM_18K"]),
            "lut": to_int(metrics["Area"]["LUT"]),
            "dsp": to_int(metrics["Area"]["DSP"]),
            "avg_latency": to_int(metrics["Latency"]["LatencyAvg"]) * freq_estimate,
            "best_latency": to_int(metrics["Latency"]["LatencyBest"]) * freq_estimate,
            "worst_latency": to_int(metrics["Latency"]["LatencyWorst"]) * freq_estimate,
        }
    except FileNotFoundError as e:
        raise Exception(e.filename)


if __name__ == "__main__":
    path = Path("/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff")
    res = parse_report(path, "proj", "forward")
    pprint(res)
