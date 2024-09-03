# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import argparse
import logging
import os
import yaml
from our_utils.parallel_process import parallel_process, Popen
from glob import glob
import json
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def read_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def get_dict_results(paths):
    x = []
    y = []
    for _, report in enumerate(paths):
            x.append(read_json(report)["color_bits_per_input_point"])
            y.append(read_json(report)["y_psnr"])

    order = np.argsort(np.array(x))

    return np.array(x)[order], np.array(y)[order]

def run_experiment(output_dir, model_name, arch_type,
                   N_levels, M, enh_channels,attention_channels,
                   num_scales,scale_min,scale_max,
                   color_space, squeeze_type, model_dir, pc_name, 
                   pcerror_path, pcerror_cfg_path, input_pc, input_normals, no_stream_redirection=False):

    os.makedirs(output_dir, exist_ok=True)
    additional_params = []
    if input_normals is not None:
        additional_params += ["--input_normals", input_normals]
    if no_stream_redirection:
        f = None
        additional_params += ["--no_stream_redirection"]
    else:
        f = open(os.path.join(output_dir, "experiment.log"), "w")
    return Popen(["python", "eval_model.py",
                  "--output_dir", output_dir,
                  "--model_name", model_name,       
                  "--arch_type", arch_type,
                  "--color_space", color_space,
                  "--squeeze_type", squeeze_type,
                  "--model_dir", model_dir,
                  "--N_levels", str(N_levels),
                  "--M", str(M),
                  "--enh_channels", str(enh_channels),
                  "--attention_channels", str(attention_channels),
                  "--num_scales", str(num_scales),
                  "--scale_min", str(scale_min),
                  "--scale_max", str(scale_max),
                  "--pc_name", pc_name,
                  "--input_pc", input_pc,
                  "--pcerror_path", pcerror_path,
                  "--pcerror_cfg_path", pcerror_cfg_path,
                  ] + additional_params, stdout=f, stderr=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="eval_all.py", description="Run experiments.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Experiments file path.", default="config_files/eval_config.yaml")
    parser.add_argument("--num_parallel", help="Number of parallel jobs. Adjust according to CPU memory.", default=2, type=int)
    parser.add_argument("--no_stream_redirection", help="Disable stdout and stderr redirection.", default=False, action="store_true")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        experiments = yaml.load(f.read(), Loader=yaml.FullLoader)
    keys = ["MODEL_PATH","MPEG_TMC13_DIR", "PCERROR", "MPEG_DATASET_DIR", "EXPERIMENT_DIR", "pcerror_mpeg_mode",
            "model_configs"]
    MODEL_PATH, MPEG_TMC13_DIR, PCERROR, MPEG_DATASET_DIR, EXPERIMENT_DIR, pcerror_mpeg_mode, model_configs = [experiments[x] for x in keys]

    logger.info("Starting our method\"s experiments")
    params = []
    for experiment in experiments["data"]:
        pc_name, cfg_name, input_pc, input_norm = [experiment[x] for x in ["pc_name", "cfg_name", "input_pc", "input_norm"]]
        opt_output_dir = os.path.join(EXPERIMENT_DIR, pc_name)
        for model_config in model_configs:
            model_id = model_config["id"]
            lambdas = model_config["lambda"]
            color_space = model_config["color_space"]
            arch_type = model_config["arch_type"]
            N_levels = model_config["N_levels"]
            M = model_config["num_filters_M"]
            enh_channels = model_config["enh_channels"]
            attention_channels = model_config["attention_channels"]
            num_scales = model_config["num_scales"]
            scale_min = model_config["scale_min"]
            scale_max = model_config["scale_max"]
            if "NF" in arch_type:
                squeeze_type = model_config["squeeze_type"]
            else:
                squeeze_type = ["naive" for i in lambdas]
            for i, lmbda in enumerate(lambdas):
                if lmbda == "max":
                    lmbda=0
                lmbda_str = f"{lmbda:.2e}"
                checkpoint_id = model_config.get("checkpoint_id", model_id)
                model_dir = os.path.join(MODEL_PATH, checkpoint_id, lmbda_str)
                current_output_dir = os.path.join(opt_output_dir, model_id, lmbda_str)

                pcerror_cfg_path = f"{MPEG_TMC13_DIR}/cfg/{pcerror_mpeg_mode}/{cfg_name}/r06/pcerror.cfg"
                input_pc_full = os.path.join(MPEG_DATASET_DIR, input_pc)
                if input_norm is not None:
                    input_norm_full = os.path.join(MPEG_DATASET_DIR, input_norm)
                else:
                    input_norm_full = None
                if not os.path.exists(os.path.join(model_dir, "done")):
                    logger.warning(f"Model training is not finished: skipping {model_dir} for {pc_name}")
                else:
                    if not os.path.exists(os.path.join(current_output_dir, f"report.json")):
                        params.append((current_output_dir, 
                                        model_id,
                                        arch_type,
                                        N_levels[i],
                                        M[i],
                                        enh_channels[i],
                                        attention_channels[i],
                                        num_scales[i],
                                        scale_min[i],
                                        scale_max[i],
                                        color_space[i],
                                        squeeze_type[i],
                                        model_dir, 
                                        pc_name,
                                        PCERROR, 
                                        pcerror_cfg_path,
                                        input_pc_full,
                                        input_norm_full,
                                        args.no_stream_redirection))
                                        
    parallel_process(run_experiment, params, args.num_parallel)
    for experiment in experiments["data"]:
        pc_name, cfg_name, input_pc, input_norm = [experiment[x] for x in ["pc_name", "cfg_name", "input_pc", "input_norm"]]
        opt_output_dir = os.path.join(EXPERIMENT_DIR, pc_name)
        for model in model_config:
            model_id = model_config["id"]
            current_output_dir = os.path.join(opt_output_dir, model_id)
            report_paths = glob(os.path.join(current_output_dir,"*/report.json"))
            bpp, psnr = get_dict_results(report_paths)
            df = pd.DataFrame({'bpp': bpp, 'y-psnr': psnr})
            df.to_csv(os.path.join(current_output_dir,"results.csv"))
    logger.info("Done")