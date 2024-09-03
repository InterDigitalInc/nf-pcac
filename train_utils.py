# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

from time import time
import torch
import torch.optim as optim

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def configure_optimizers(model, learning_rate, aux_learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers
    Inputs:
        model = model to optimize the parameters
        learning_rate = learning rate for the main model
        aux_learning_rate = learning rate for the auxiliary part
        
    Outputs:
        optimizer = optimizer for the model parameters
        aux_optimizer = optimizer for the quantiles part of the model"""

    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and "residual" not in n and p.requires_grad
    }

    aux_parameters = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and "residual" not in n and p.requires_grad
    }

    no_grad_params = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and not p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
    inter_params = parameters & aux_parameters & no_grad_params
    union_params = parameters | aux_parameters | no_grad_params

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_learning_rate,
    )

    return optimizer, aux_optimizer

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", inference_only=False):
    """ Function to save the state of the model if the loss is best
    Inputs:
        state = dictionary with the state of the model (weights)
        is_best = boolean, if true will save the file to 'filename'
        filename = string, path where to save the file
    """
    if is_best:
        if inference_only:
            torch.save(state["state_dict"],filename)
        else:
            torch.save(state, filename)

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def count_parameters(model):
    """ Function to count the number of trained parameters in a model
    Inputs:
        model = torch.Module, the model to count the number of coefs
    Outputs:
        n = total number of parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model,loss_layer,train_dataloader,optimizer,aux_optimizer,epoch, clip_max_norm=1.0):
    """ Function to train one epoch of the model and do the backward propagation (will be inside a loop in the main function)
    Inputs:
        model = torch.Module, the model to train
        loss_layer = torch.Module, the loss layer chosen to train the model
        train_dataloader = torch.DataLoader, the dataloader to feed the data to the model
        optimizer = the optimizer to do the backward propagation of the main model (Adam)
        aux_optimizer = the optimizer to do the backward propagation of the auxiliary loss (Adam)
        epoch = int, which epoch the training is currently on
    Outputs:
        avg_loss.avg = the total average loss of the current epoch over all batches (R+lmda*D)
        bpp_loss.avg = the average bpp loss over all batches (log likelihood)
        mse_loss.avg = the average MSE loss over all batches (MSE between 2 values of 0-1, good value is under 0.0002)
        aux_loss_avg.avg = the average of the auxiliary loss
    
    """
    # Put the model in training mode and its parameters into CUDA
    model.train()
    device = next(model.parameters()).device
    import timing
    # Create the average meter to keep track of the loss during the entire epoch
    avg_loss_log = AverageMeter()
    bpp_loss_log = AverageMeter()
    mse_loss_log = AverageMeter()
    aux_loss_log = AverageMeter()

    import MinkowskiEngine as ME
    start_time = time()

    # Loop through all the batches in the train dataloader
    for i, d in enumerate(train_dataloader):

        # Put the data in the device (GPU)
        d = ME.SparseTensor(d[1],d[0],device=device)

        # Reset the gradients of both optimizers
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # Pass the data through the model
        out_net = model(d)

        # Calculate the loss using the loss layer (RateDistortion)
        out_criterion = loss_layer(out_net,d)

        # Backward propagation of the loss
        out_criterion["loss"].backward()
        
        # Clip all the gradients to avoid vanishing/exploding gradiends and take a step
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # Do the same steps with the model auxiliary loss
        aux_loss = model.aux_loss()
        if aux_loss.requires_grad:
            aux_loss.backward()
        aux_optimizer.step()
            
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize(device)
        # Print the loss each 10% of the progresss
        if i%int(len(train_dataloader)/10)==0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*train_dataloader.batch_size}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.5f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.5f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.5f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
        
        avg_loss_log.update(out_criterion["loss"].item())
        mse_loss_log.update(out_criterion["mse_loss"].item())
        bpp_loss_log.update(out_criterion["bpp_loss"].item())
        aux_loss_log.update(aux_loss.item())
    
    end_time = time()-start_time
    print("Time Taken: " + str(end_time))

    print(
        f"Train epoch {epoch}: Average losses:"
        f"\tLoss: {avg_loss_log.avg:.5f} |"
        f"\tMSE loss: {mse_loss_log.avg:.5f} |"
        f"\tBpp loss: {bpp_loss_log.avg:.5f} |"
        f"\tAux loss: {aux_loss_log.avg:.2f}\n"
    )

    return avg_loss_log.avg, mse_loss_log.avg, bpp_loss_log.avg, aux_loss_log.avg
    

def validation_epoch(model,epoch,loss_layer,test_dataloader):
    """ Function to validate one epoch of the model
    Inputs:
        model = torch.Module, the model to train
        loss_layer = torch.Module, the loss layer chosen to train the model
        test_dataloader = torch.DataLoader, the dataloader to feed the data to the model
    Outputs:
        avg_loss.avg = the total average loss of the validation data (R+lmda*D)
        bpp_loss.avg = the average bpp loss over the validation data (log likelihood)
        mse_loss.avg = the average MSE loss over the validation data (MSE between 2 values of 0-1, good value is under 0.0002)
        aux_loss_avg.avg = the average of the auxiliary loss
    
    """
    # Set the model in evaluation mode
    model.eval()
    # Get the device (GPU)
    device = next(model.parameters()).device

    loss_log = AverageMeter()
    mse_loss_log = AverageMeter()
    bpp_loss_log = AverageMeter()
    aux_loss_log = AverageMeter()

    import MinkowskiEngine as ME
    index = None
    # Use torch no grad to not compute the gradient and backprop
    with torch.no_grad():
        for d in test_dataloader:

            # Put the data in the device (GPU)
            d = ME.SparseTensor(d[1],d[0],device=device)

            # Pass the data through the model
            out_net = model(d)

            # Compute the loss
            out_criterion = loss_layer(out_net,d)

            # Update the loss for the entire test dataset
            mse_loss_log.update(out_criterion["mse_loss"].item())
            bpp_loss_log.update(out_criterion["bpp_loss"].item())
            loss_log.update(out_criterion["loss"].item())
            aux_loss_log.update(model.aux_loss().item())
            

    # Print all the average losses
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss_log.avg:.5f} |"
        f"\tMSE loss: {mse_loss_log.avg:.5f} |"
        f"\tBpp loss: {bpp_loss_log.avg:.5f} |"
        f"\tAux loss: {aux_loss_log.avg:.2f}\n"
    )

    # Return all the losses to write in the summary
    return loss_log.avg, mse_loss_log.avg, bpp_loss_log.avg, aux_loss_log.avg
