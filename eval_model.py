# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
from contextlib import ExitStack
import yaml
from our_utils.mpeg_parsing import *
from our_utils.parallel_process import parallel_process, Popen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def flatten(l):
    return [item for sublist in l for item in sublist]


def print_progress(from_path, to_path, comment=""):
    if not isinstance(from_path, list):
        from_path = [from_path]
    if not isinstance(to_path, list):
        to_path = [to_path]
    from_path_str = ", ".join(from_path)
    to_path_str = ", ".join(to_path)
    logger.info(f"[{from_path_str}] -> [{to_path_str}] {comment}")


def run_pcerror(decoded_pc, input_pc, input_normals, pcerror_cfg_params, pcerror_path, pcerror_result):
    f = open(pcerror_result, "w")
    
    command = [pcerror_path,
                  "-a", input_pc, "-b", decoded_pc]
    if input_normals is not None:
        command += ["-n", input_normals]
    
    print(command)
    return Popen(command + pcerror_cfg_params,
                 stdout=f, stderr=f)


def run_experiment(output_dir, model_name, arch_type,
                   N_levels, M, enh, attention,
                   num_scales, scale_min, scale_max,
                   color_space, squeeze_type, model_dir, pc_name, 
                   pcerror_path, pcerror_cfg_path, input_pc, input_normals,
                   num_parallel, no_stream_redirection=False):

    with open(pcerror_cfg_path, "r") as f:
        pcerror_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    enc_pc_filenames = [f"{pc_name}.ply.bin"]
    dec_pc_filenames = [f"{x}.ply" for x in enc_pc_filenames]
    pcerror_result_filenames = [f"{x}.pc_error" for x in dec_pc_filenames]
    enc_pcs = [os.path.join(output_dir, x) for x in enc_pc_filenames]
    dec_pcs = [os.path.join(output_dir, x) for x in dec_pc_filenames]
    pcerror_results = [os.path.join(output_dir, x) for x in pcerror_result_filenames]
    exp_reports = [os.path.join(output_dir, f"report.json")]

    compress_log = os.path.join(output_dir, "compress.log")
    decompress_log = os.path.join(output_dir, "decompress.log")

    # Create folder
    os.makedirs(output_dir, exist_ok=True)

    # Encoding or Encoding/Decoding
    if all(os.path.exists(x) for x in enc_pcs) and (all(os.path.exists(x) for x in dec_pcs)):
        print_progress(input_pc, enc_pcs, "(exists)")
    else:
        print_progress(input_pc, enc_pcs)
        with ExitStack() as stack:
            if no_stream_redirection:
                f = None
            else:
                f = open(compress_log, "w")
                stack.enter_context(f)
            
            model_path = model_dir+"/checkpoint.pth.tar"

            command = ["python", "main.py",
                            "--command","encode",
                            "--model_name", model_name,
                            "--arch_type", arch_type,
                            "--color_space", color_space,
                            "--squeeze_type",squeeze_type,
                            "--model_path", model_path,
                            "--N_levels", str(N_levels),
                            "--M", str(M),
                            "--enh_channels", str(enh),
                            "--attention_channels", str(attention),
                            "--num_scales", str(num_scales),
                            "--scale_min", str(scale_min),
                            "--scale_max", str(scale_max),
                            "--input_file", input_pc,
                            "--output_file", *enc_pcs
                            ]

            subprocess.run(command, stdout=f, stderr=f, check=True)

    # Decoding
    if all(os.path.exists(x) for x in dec_pcs):
        print_progress(enc_pcs, dec_pcs, "(exists)")
    else:
        print_progress(enc_pcs, dec_pcs)
        with ExitStack() as stack:
            if no_stream_redirection:
                f = None
            else:
                f = open(decompress_log, "w")
                stack.enter_context(f)

            model_path = model_dir+"/checkpoint.pth.tar"

            command = ["python", "main.py", 
                            "--command", "decode",
                            "--model_name", model_name,
                            "--arch_type", arch_type,
                            "--color_space", color_space,
                            "--squeeze_type",squeeze_type,
                            "--model_path", model_path,
                            "--N_levels", str(N_levels),
                            "--M", str(M),
                            "--enh_channels", str(enh),
                            "--attention_channels", str(attention),
                            "--num_scales", str(num_scales),
                            "--scale_min", str(scale_min),
                            "--scale_max", str(scale_max),
                            "--input_file", *enc_pcs,
                            "--output_file", *dec_pcs,
                            "--geo", input_pc]

            subprocess.run(command, stdout=f, stderr=f, check=True)

    pcerror_cfg_params = [[f"--{k}", str(v)] for k, v in pcerror_cfg.items()]
    pcerror_cfg_params_init = flatten(pcerror_cfg_params)
    pcerror_cfg_params = []
    for i in range(0,len(pcerror_cfg_params_init),2):
        pcerror_cfg_params.append(pcerror_cfg_params_init[i]+"="+pcerror_cfg_params_init[i+1])

    params = []
    for pcerror_result, decoded_pc in zip(pcerror_results, dec_pcs):
        if os.path.exists(pcerror_result):
            print_progress(decoded_pc, pcerror_result, "(exists)")
        else:
            print_progress(decoded_pc, pcerror_result)
            params.append((decoded_pc, input_pc, input_normals, pcerror_cfg_params, pcerror_path, pcerror_result))
    parallel_process(run_pcerror, params, num_parallel)

    for pcerror_result, enc_pc, decoded_pc, experiment_report in zip(pcerror_results, enc_pcs, dec_pcs, exp_reports):
        if os.path.exists(experiment_report):
            print_progress("all", experiment_report, "(exists)")
        else:
            print_progress("all", experiment_report)
            pcerror_data = parse_pcerror(pcerror_result)

            color_total_size_in_bytes = os.stat(enc_pc).st_size
            if "enh" in experiment_report:
                color_total_size_in_bytes+= os.stat(enc_pc+"res.bin").st_size
            input_point_count = pcerror_data["input_point_count"]
            output_point_count = pcerror_data["decoded_point_count"]

            color_bits_per_input_point = color_total_size_in_bytes * 8 / input_point_count
            color_bits_per_output_point = color_total_size_in_bytes * 8 / output_point_count

            data = {
                "color_bitstream_size_in_bytes": color_total_size_in_bytes,
                "color_bits_per_input_point": color_bits_per_input_point,
                "color_bits_per_output_point": color_bits_per_output_point,
                "input_point_count": input_point_count,
                "output_point_count": output_point_count,
            }
            data = {**data, **pcerror_data}
            with open(experiment_report, "w") as f:
                json.dump(data, f, sort_keys=True, indent=4)

    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="eval_model.py", description="Run experiment for a point cloud.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output_dir", help="Output directory", required=True)
    parser.add_argument("--model_name", help="Model name", required=True)
    parser.add_argument('--arch_type', help='Which architecture to use in the model',type=str, default="VAE")
    parser.add_argument("--color_space", help="Color Space YUV or RGB", required=True)
    parser.add_argument("--squeeze_type", help="Type of squeezing strategy for the voxel shuffling layer on the inverse archi",type=str, default="avg")
    parser.add_argument("--model_dir", help="Model directory", required=True)
    parser.add_argument("--N_levels",help="Number of levels in the invertible core.", type=int, default=3)
    parser.add_argument("--M",help="Number of filters in the output of the channel average block", type=int, default=128)
    parser.add_argument("--enh_channels",help="Number of filters per layer.", type=int, default=32)
    parser.add_argument("--attention_channels",help="Number of filters per layer.", type=int, default=192)
    parser.add_argument("--num_scales",help="Number of Gaussian scales to prepare range coding tables for.",type=int, default=64)
    parser.add_argument("--scale_min",help="Minimum value of standard deviation of Gaussians",type=float, default=.11)
    parser.add_argument("--scale_max",help="Maximum value of standard deviation of Gaussians",type=float, default=256.)
    parser.add_argument("--pc_name", help="Point cloud name", required=True)
    parser.add_argument("--input_pc", help="Path to input point cloud", required=True)
    parser.add_argument("--input_normals", help="Path to input point cloud", default=None)
    parser.add_argument("--pcerror_path", help="Path to pcerror executable", required=True)
    parser.add_argument("--debug", help="Path to the debug folder, will write each block as a separate PC in the folder")
    parser.add_argument("--pcerror_cfg_path", help="Path to pcerror configuration", required=True)
    parser.add_argument("--num_parallel", help="Number of parallel jobs", default=multiprocessing.cpu_count(), type=int)
    parser.add_argument("--no_stream_redirection", help="Disable stdout and stderr redirection.", default=False, action="store_true")
    args = parser.parse_args()

    run_experiment(args.output_dir, args.model_name, args.arch_type, 
                   args.N_levels, args.M, args.enh_channels, args.attention_channels,
                   args.num_scales, args.scale_min, args.scale_max,
                   args.color_space, args.squeeze_type, 
                   args.model_dir, args.pc_name, args.pcerror_path, args.pcerror_cfg_path, args.input_pc, args.input_normals,
                   args.num_parallel, args.no_stream_redirection)
