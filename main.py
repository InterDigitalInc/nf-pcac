# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import gzip
import argparse
import torch
from binary_syntax import load_compressed_file, save_compressed_file
from our_utils.load_model_utils import load_state_dict
from models.model_selection import model_selection
from our_utils.transform_io_utils import read_PC, write_PC
import numpy as np

import MinkowskiEngine as ME

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def encode(args):
  """Compresses a Point Cloud.
  Inputs:
    args.input_file = the path to the .ply to compress
    args.model_path = the path of the trained model
    args.output_file = the path to write the binary output file
  Outputs:
    No outputs from the function = will write the binary file to args.output_file
  """
  # Look for a cuda device that is available for use in the compression
  # device = "cuda" if torch.cuda.is_available() else "cpu"
  # On the GPU the inference is unstable!!!
  device = "cpu"

  # Define the model
  logger.info("Defining model and loading into "+device)
  model = model_selection(args)

  # Load the trained weights into the model and put the model in the correct device
  state_dict = load_state_dict(torch.load(args.model_path))
  model.load_state_dict(state_dict["state_dict"])
  model.eval()
  model = model.to(device)
  model.update(force=True)
  logger.info("Model loaded into "+device)

  # Load the point cloud into memory
  logger.info("Reading Point Cloud...")

  # Read the input file, put it into the correct color space and in the 0-1 range
  points = read_PC(args.input_file,YUV=args.color_space=='YUV')

  logger.info("Encoding Point Cloud...")
  with torch.no_grad():
    # Make the coords and feats tensors to be put in the sparse tensor
    coords, feats = ME.utils.sparse_collate([points[:,:3]],[points[:,3:]])
    points_tensor = ME.SparseTensor(feats,coords,device=device)        

    # Pass the tensor through the model to obtain the bitstreams
    out_enc = model.compress(points_tensor)

  bitstr = save_compressed_file([out_enc["strings"][0],out_enc["strings"][1]])
  with gzip.open(args.output_file,"wb") as f:
    f.write(bitstr)
  

def decode(args):
  """Decompresses a PC"""
  
  # Look for a cuda device that is available for use in the decompression
  # device = "cuda" if torch.cuda.is_available() else "cpu"
  # On the GPU the inference is unstable!!!
  device = "cpu"

  # Define the model
  logger.info("Defining model and loading into "+device)
  model = model_selection(args)

  # Load the trained weights into the model and put the model in the correct device
  state_dict = load_state_dict(torch.load(args.model_path))
  model.load_state_dict(state_dict["state_dict"])
  model.eval()
  model = model.to(device)
  model.update(force=True)
  logger.info("Model loaded into "+device)

  # Read the shape information and compressed string from the binary file,
  # and decompress the image using the model.
  logger.info("Reading bitstreams...")
  with gzip.open(args.input_file,"rb") as f:
    bitstream = load_compressed_file(f)

  if args.geo is not None:
    geo_points = read_PC(args.geo)
  else:
    raise Exception("The geometry information is needed to decode the bitstream") 

  logger.info("Decoding the bitstream...")
  with torch.no_grad():
    # Get the latent space coordinates and hyper latent space coordinates
    coords, feats = ME.utils.sparse_collate([geo_points[:,:3]],[np.zeros_like(geo_points[:,3:])])
    geo_tensor = ME.SparseTensor(feats,coords,device=device)
    # Get the latent coordinates since we already know the geometry information
    latent_coords = model.analysis_transform(geo_tensor)
    # Make sure the latent coordinates do not contain any info from the original colors
    latent_coords._F=latent_coords._F*0
    hyper_latent_coords = model.hyper_analysis_transform(latent_coords)
    # Make sure the hyper latent coordinates do not contain any info from the original colors
    hyper_latent_coords._F=hyper_latent_coords._F*0

    out_dec = model.decompress(bitstream, latent_coords, hyper_latent_coords)

    x_hat_features = out_dec["x_hat"].F.cpu().detach().numpy()
    x_hat_features = np.clip(x_hat_features,0,1)

    x_hat_coord = out_dec["x_hat"].C.cpu().detach().numpy()

  decoded_points = np.concatenate((x_hat_coord[:,1:],x_hat_features),axis=1)
  write_PC(decoded_points,args.output_file,inYUV=args.color_space=="YUV")

def main(args):
  if args.command=="encode":
    encode(args)
  elif args.command=="decode":
    decode(args)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog="main.py", description="Encode or decode a point cloud depending on the command argument",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--command", help="What to do, encode or decode", required=True)
  parser.add_argument("--input_file",help="File to encode or decode, .ply or .bin", required=True)
  parser.add_argument("--output_file",help="Path to the output of the encoding or decoding, .bin or .ply", required=True)
  parser.add_argument("--model_name", help="Name of the trained model",type=str, default="Default")
  parser.add_argument('--arch_type', help='Which architecture to use in the model',type=str, default="NF")
  parser.add_argument("--color_space", help="Color space in which to encode/decode",type=str, default="RGB")
  parser.add_argument("--squeeze_type", help="Type of squeezing strategy for the voxel shuffling layer on the inverse archi",type=str, default="avg")
  parser.add_argument("--model_path",help="Path to the trained model",type=str, required=True)
  parser.add_argument("--geo",help="Path to the .ply to use as decoded geometry: .ply")
  parser.add_argument("--N_levels",help="Number of filters per layer.",type=int, default=64)
  parser.add_argument("--M",help="Number of filters per layer.",type=int, default=128)
  parser.add_argument("--enh_channels",help="Number of filters per layer.", type=int, default=64)
  parser.add_argument("--attention_channels",help="Number of filters per layer.", type=int, default=128)
  parser.add_argument("--num_scales",help="Number of Gaussian scales to prepare range coding tables for.",type=int, default=64)
  parser.add_argument("--scale_min",help="Minimum value of standard deviation of Gaussians",type=float, default=.11)
  parser.add_argument("--scale_max",help="Maximum value of standard deviation of Gaussians",type=float, default=256.)

  args = parser.parse_args()

  main(args)