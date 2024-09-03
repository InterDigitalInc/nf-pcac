# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import torch
# torch.manual_seed(13)
# Choose the best type of convolution for the training
torch.backends.cudnn.benchmark = True

import numpy as np
np.random.seed(13)

import random
random.seed(13)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from models.model_selection import model_selection
from glob import glob
from custom_data import CustomDataset
from data_augmentation import ComposeDataAugmentation
from data_augmentation.transforms import ChannelSwap, RotationAugmentation, SparseRandomSolarize, SparseColorJitter, ColorShift
from torch.utils.data import DataLoader
from train_utils import configure_optimizers, human_format, save_checkpoint, train_one_epoch, validation_epoch, count_parameters
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import argparse
from models.compress_model_base import RateDistortionLoss
import MinkowskiEngine as ME

def main(args):

    writer = SummaryWriter(log_dir="runs/"+args.model_name)
    train = []
    validation = []

    for data_train in args.train_data_path:
        train+=glob(os.path.join(data_train, "*.ply"))

    for data_validation in args.validation_data_path:
        validation+=glob(os.path.join(data_validation, "*.ply"))

    # Define the device used for training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    AUX_LR = args.aux_lr
    N_EPOCHS = args.epochs

    print("Loading the requested model: "+args.model_name)
    print("Architecture type: "+ args.arch_type)
    
    # Model Selection based on model id
    model = model_selection(args)
    model = model.to(device)

    n_params = count_parameters(model)
    print("Number of parameters is: " + human_format(n_params))

    loss_layer = RateDistortionLoss(lmbda=args.lmbda)
    optimizer, aux_optimizer = configure_optimizers(model,LEARNING_RATE,AUX_LR)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=20, factor=0.5)
    
    data_augmentation_list = []
    if "channel_swap" in args.data_augmentation:
        data_augmentation_list.append(ChannelSwap())
    if "solarize" in args.data_augmentation:
        data_augmentation_list.append(SparseRandomSolarize(0.5))
    if "colot_jitter" in args.data_augmentation:
        data_augmentation_list.append(SparseColorJitter(brightness=1.5,contrast=1.5,saturation=1.5,hue=0.5))
    if "color_shift" in args.data_augmentation:
        data_augmentation_list.append(ColorShift(shift=20))
    if "rotation" in args.data_augmentation:
        data_augmentation_list.append(RotationAugmentation())

    train_dataset = CustomDataset(pcFileList=train,
                                  YUV=True if args.color_space=="YUV" else False,
                                  out255=False,
                                  data_augmentation=ComposeDataAugmentation(data_augmentation_list))

    print("Training with {} PC blocks".format(len(train_dataset)))
    print("Color Space: {}".format(args.color_space))

    train_loader=DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=4, # You can modify the num_workers depending on your PC memory
                            pin_memory=(device == "cuda"),
                            collate_fn=ME.utils.batch_sparse_collate)

    validation_dataset = CustomDataset(pcFileList=validation,
                                        YUV=True if args.color_space=="YUV" else False,
                                        out255=False,
                                        data_augmentation=ComposeDataAugmentation(data_augmentation_list))
    
    print("Validation with {} PC blocks".format(len(validation_dataset)))
    
    if len(validation_dataset)!=0:
        val_loader=DataLoader(validation_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=(device == "cuda"),
                                collate_fn=ME.utils.batch_sparse_collate)
    else:
        val_loader = []

    # To continue training from a checkpoint
    last_epoch=0
    if args.load_weights is not None:
        
        print("loading previous checkpoint: ", args.load_weights)
        checkpoint = torch.load(args.load_weights,map_location=device)
        last_epoch = checkpoint["epoch"]+1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # Set the best loss to infinite to update it
    best_loss = float("inf")

    print("Start training...")
    for epoch in range(last_epoch,last_epoch+N_EPOCHS):
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
        if optimizer.param_groups[0]['lr'] < 1e-5:
            break
        train_loss, train_mse_loss, train_bpp_loss, train_aux_loss = train_one_epoch(model,
                                                                                    loss_layer,
                                                                                    train_loader,
                                                                                    optimizer,
                                                                                    aux_optimizer,
                                                                                    epoch)
        # Write the train loss into the tensorboard graph
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("MSE_Loss/Train", train_mse_loss, epoch)
        writer.add_scalar("BPP_Loss/Train", train_bpp_loss, epoch)
        writer.add_scalar("Aux_Loss/Train", train_aux_loss, epoch)

        if len(val_loader):
            val_loss, val_mse_loss, val_bpp_loss, val_aux_loss = validation_epoch(model,
                                                                                epoch,
                                                                                loss_layer,
                                                                                val_loader)
            lr_scheduler.step(val_loss)
            # Write the validation loss into the tensorboard graph
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("MSE_Loss/Val", val_mse_loss, epoch)
            writer.add_scalar("BPP_Loss/Val", val_bpp_loss, epoch)
            writer.add_scalar("Aux_Loss/Val", val_aux_loss, epoch)
        else:
            val_loss = float("inf")
            lr_scheduler.step(train_loss)

        is_best = val_loss<best_loss
        best_loss = min(val_loss, best_loss)

        # Save the best validation loss
        save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": val_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict()
                },
                is_best,
                filename=args.model_save_path,
                inference_only=True
            )
        
        # Save the last trained checkpoint
        save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": val_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict()
                },
                True,
                filename=os.path.join(os.path.dirname(args.model_save_path),"last_ckp.pth.tar"),
                inference_only=False
            )
        
    writer.flush()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_model.py", description="Train one model with the specified configuration",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_data_path", help="Path to the folders containing the PCs to train on", nargs="+", default=[], required=True)
    parser.add_argument("--validation_data_path",help="Path to the folders containing the PCs to validate on", nargs="+", default=[], required=True)
    parser.add_argument("--data_augmentation", nargs="+", default=[])
    parser.add_argument("--load_weights", help="Path where to load the weights from previous training")
    parser.add_argument("--model_save_path", help="Path where to save the trained model",type=str, default="checkpoint.pth.tar")
    parser.add_argument("--model_name", help="Name of the trained model",type=str, default="Default")
    parser.add_argument("--arch_type", help="Which architecture to use in the model",type=str, default="VAE")
    parser.add_argument("--batch_size",help="Batch Size to train on", type=int, default=16)
    parser.add_argument("--learning_rate",help="Learning rate to train on", type=float, default=1e-4)
    parser.add_argument("--aux_lr",help="Learning rate to train on", type=float, default=1e-3)
    parser.add_argument("--epochs",help="Number of epochs to train on", type=int, default=1)
    parser.add_argument("--lmbda",help="Lambda for rate-distortion tradeoff.", type=float, default=2e3)
    parser.add_argument("--color_space", help="Color space to use for training and validation",type=str, default="RGB")
    parser.add_argument("--squeeze_type", help="Type of squeezing strategy for the voxel shuffling layer on the inverse archi",type=str, default="avg")
    parser.add_argument("--N_levels",help="Number of levels in the invertible core.", type=int, default=3)
    parser.add_argument("--M",help="Number of filters in the output of the channel average block", type=int, default=128)    
    parser.add_argument("--enh_channels",help="Number of filters per layer.", type=int, default=64)
    parser.add_argument("--attention_channels",help="Number of filters per layer.", type=int, default=128)
    parser.add_argument("--num_scales",help="Number of Gaussian scales to prepare range coding tables for.",type=int ,default=64)
    parser.add_argument("--scale_min",help="Minimum value of standard deviation of Gaussians", type=float, default=.11)
    parser.add_argument("--scale_max",help="Maximum value of standard deviation of Gaussians", type=float ,default=256.)

    args = parser.parse_args()

    import timing
    main(args)