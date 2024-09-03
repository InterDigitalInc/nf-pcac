# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import yaml
import argparse
import os
import subprocess
from pathlib import Path

def lmbda_to_str(lmbda):
    if lmbda =="lossless":
        return lmbda
    else:
        return f"{lmbda:.2e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_all.py", description="Train all models for an experimental setup.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_path", help="Train config file path.", default ="config_files/train_config.yaml")
    args = parser.parse_args()

    # Read the config file
    with open(args.config_path, "r") as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Create the directory where we will save the weights
    os.makedirs(config["WEIGHTS_PATH"], exist_ok=True)

    print("Starting training of all models...")
    # For each different model in the config file:
    for model in config["model_configs"]:
        # Get the model ID and create the sub-directory where we will save the weights
        id = model["id"]
        model_path = os.path.join(config["WEIGHTS_PATH"],id)
        os.makedirs(model_path,exist_ok=True)

        # Train for every lambda in the model - 1 lambda = 1 bitrate point
        for i, lmbda in enumerate(model["lambda"]):
            # Check if we are training for the max PSNR for the model (meaning we will ignore the bitrate constraint and get the better reconstruction possible)
            if lmbda=="max":
                lmbda=0
                psnr_max=True
            else:
                psnr_max=False

            model_lmbda_path=os.path.join(model_path,lmbda_to_str(lmbda))

            # Check if training was already performed before launching it
            if not os.path.exists(os.path.join(model_lmbda_path, 'done')):
                
                # Create the subdirectory for each lambda:
                os.makedirs(model_lmbda_path,exist_ok=True)

                # Start the log path
                log_path = model_lmbda_path+"/training.log"

                # Dump the config file to keep track of which training it is
                with open(model_lmbda_path+"/config.yaml", "w") as outfile:
                    yaml.dump(config, outfile, default_flow_style=False)
                
                # Start the model training
                print("Training model to: "+model_lmbda_path)
                with open(log_path,"w") as f:
                    command = ["python", "train_model.py",
                    "--model_save_path", model_lmbda_path+"/checkpoint.pth.tar",
                    "--model_name",str(model["id"]+"_"+lmbda_to_str(lmbda)),
                    "--arch_type",str(model["arch_type"]),
                    "--batch_size", str(model["batch_size"]),
                    "--learning_rate", str(model["learning_rate"]),
                    "--epochs", str(model["epochs"][i]),
                    "--lmbda", str(lmbda),
                    "--color_space",str(model["color_space"][i]),
                    "--N_levels", str(model["N_levels"][i]),
                    "--M", str(model["num_filters_M"][i]),
                    "--enh_channels", str(model["enh_channels"][i]),
                    "--attention_channels", str(model["attention_channels"][i]),
                    "--num_scales", str(model["num_scales"][i]),
                    "--scale_min", str(model["scale_min"][i]),
                    "--scale_max", str(model["scale_max"][i])]

                    command.append("--train_data_path")
                    for train_data in model["train_dataset"]:
                        command.append(train_data)

                    command.append("--validation_data_path")
                    for val_data in model["validation_dataset"]:
                        command.append(val_data)

                    ##############################################################
                    if psnr_max:
                        command.append("--psnr_max")
                        
                    if "squeeze_type" in model.keys():
                        command.append("--squeeze_type")
                        command.append(str(model["squeeze_type"][i]))

                    if "load_weights" in model.keys() and i<len(model["load_weights"]):
                        if not model["load_weights"][i]=="None":
                            command.append("--load_weights")
                            command.append(str(model["load_weights"][i]))
                    
                    if "data_augmentation" in model.keys():
                        command.append("--data_augmentation")
                        for da in model["data_augmentation"]:
                            command.append(str(da))

                    subprocess.run(command,stdout=f, stderr=f, check=True)
                
                # After the subprocess is finished (training is done) create a file to indicate it is over
                Path(os.path.join(model_lmbda_path, "done")).touch()