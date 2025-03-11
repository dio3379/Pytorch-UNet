import argparse
import logging
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from unet.ShadowRemovalUnet import ShadowRemovalUnet
from utils.data_loading_for_ISTD_Dataset import ISTDDataset_ShadowRemovalUnet

def predict(
    model,
    device,
    test_dir_A: Path,
    test_dir_B: Path,
    test_dir_C: Path,
    output_dir: Path,
    img_scale: float = 0.5,
    amp: bool = False,
):
    # Create dataset for testing
    test_dataset = ISTDDataset_ShadowRemovalUnet(test_dir_A, test_dir_B, test_dir_C, img_scale)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Track metrics
    total_psnr = 0.0
    total_ssim = 0.0
    total_images = 0
    
    # Get a list of all filenames from directory A
    file_names_A = [f.stem for f in test_dir_A.glob('*') if f.is_file()]
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc='Predicting images', unit='img'):
            # Get sample from dataset
            sample = test_dataset[i]
            image = sample['image']
            true_mask = sample['mask']
            
            # Get original filename from directory A
            # If sample contains the name, use it; otherwise use the filename from directory A
            if 'name' in sample:
                base_name = sample['name']
            else:
                # Use the i-th filename from directory A if available, otherwise use a default name
                base_name = file_names_A[i] if i < len(file_names_A) else f'image_{i}'
            
            # Create output filename by adding '_OUT' suffix
            output_name = f"{base_name}_OUT"
            
            # Add batch dimension
            image = image.unsqueeze(0)
            true_mask = true_mask.unsqueeze(0)
            
            # Move to device and set correct format
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_mask = true_mask.to(device=device, dtype=torch.float32)
            
            # Predict
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                output = model(image)
            
            # Convert to numpy arrays for visualization and metrics
            output_np = output.cpu().numpy()[0]
            true_mask_np = true_mask.cpu().numpy()[0]
            
            # Calculate metrics
            # Transpose from (C, H, W) to (H, W, C) for metrics calculation
            output_img = np.transpose(output_np, (1, 2, 0))
            true_img = np.transpose(true_mask_np, (1, 2, 0))
            
            # Ensure values are in [0, 1] range
            output_img = np.clip(output_img, 0, 1)
            
            # Calculate metrics
            img_psnr = psnr(true_img, output_img, data_range=1.0)
            img_ssim = ssim(true_img, output_img, data_range=1.0, channel_axis=2)
            
            total_psnr += img_psnr
            total_ssim += img_ssim
            total_images += 1
            
            # Save the predicted image
            # Convert from (C, H, W) to (H, W, C) for OpenCV
            output_for_save = np.transpose(output_np, (1, 2, 0))
            
            # Convert to uint8 for saving
            output_for_save = (output_for_save * 255).astype(np.uint8)
            
            # Save using OpenCV (BGR format for OpenCV)
            output_for_save_bgr = cv2.cvtColor(output_for_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f'{output_name}.png'), output_for_save_bgr)
            
            # Log the file correspondence
            logging.debug(f'Processed: Input={base_name}, Output={output_name}.png')
    
    # Calculate average metrics
    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images
    
    logging.info(f'Average PSNR: {avg_psnr:.2f}')
    logging.info(f'Average SSIM: {avg_ssim:.4f}')
    
    return avg_psnr, avg_ssim

def get_args():
    parser = argparse.ArgumentParser(description='Predict using trained UNet model')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the model checkpoint file')
    parser.add_argument('--test-dir-A', type=str, required=True, 
                        help='Directory containing test A images')
    parser.add_argument('--test-dir-B', type=str, required=True, 
                        help='Directory containing test B images')
    parser.add_argument('--test-dir-C', type=str, required=True, 
                        help='Directory containing test C images (ground truth)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory to save the predicted images')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Image downscaling factor')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use mixed precision')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='Enable verbose logging')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    # Set logging level based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Check if CUDA is available
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Convert path strings to Path objects
    test_dir_A = Path(args.test_dir_A)
    test_dir_B = Path(args.test_dir_B)
    test_dir_C = Path(args.test_dir_C)
    output_dir = Path(args.output)
    
    # Load model
    logging.info(f'Loading model from {args.model}')
    model = ShadowRemovalUnet(4, 3)
    model = model.to(memory_format=torch.channels_last)
    
    state_dict = torch.load(args.model, map_location=device)
    # Remove mask_values if it exists (used during training)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    
    model.load_state_dict(state_dict)
    model.to(device=device)
    
    # Predict images
    avg_psnr, avg_ssim = predict(
        model=model,
        device=device,
        test_dir_A=test_dir_A,
        test_dir_B=test_dir_B,
        test_dir_C=test_dir_C,
        output_dir=output_dir,
        img_scale=args.scale,
        amp=args.amp
    )
    
    logging.info(f'Prediction completed. Results saved to {output_dir}')
    logging.info(f'Test set performance: PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}')

# python predict_ShadowRemovalUnet.py -m /home/Data_Pool/zjnu/qianlf/ImgEnhancement/Pytorch-UNet/ShadowRemove_model/best_model.pth --test-dir-A /home/Data_Pool/zjnu/qianlf/ImgEnhancement/data/ISTD_Dataset/test/test_A --test-dir-B /home/Data_Pool/zjnu/qianlf/ImgEnhancement/data/ISTD_Dataset/test/test_B --test-dir-C /home/Data_Pool/zjnu/qianlf/ImgEnhancement/data/ISTD_Dataset/test/test_C -o /home/Data_Pool/zjnu/qianlf/ImgEnhancement/data/outputdir
