import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from unet.ShadowRemovalUnet import ShadowRemovalUnet
from utils.data_loading_for_ISTD_Dataset import ISTDDataset_ShadowRemovalUnet

dir_img = Path('/home/Data_Pool/zjnu/qianlf/ImgEnhancement/data/ISTD_Dataset/train/train_A')
dir_mask = Path('/home/Data_Pool/zjnu/qianlf/ImgEnhancement/data/ISTD_Dataset/train/train_B')
dir_true_label = Path('/home/Data_Pool/zjnu/qianlf/ImgEnhancement/data/ISTD_Dataset/train/train_C')
dir_checkpoint = Path('./checkpoints/')

# 新的评估函数，适用于线性恢复任务
def evaluate_linear_restoration(model, dataloader, device, amp):
    model.eval()
    num_val_batches = len(dataloader)
    psnr_score = 0
    ssim_score = 0

    # 迭代验证数据集
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc='验证', unit='batch', leave=False):
            images, true_masks = batch['image'], batch['mask']
            
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            
            # 前向传播
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)
            
            # 转换为NumPy数组用于计算PSNR和SSIM
            pred_np = masks_pred.cpu().numpy()
            true_np = true_masks.cpu().numpy()
            
            # 计算每张图像的PSNR和SSIM
            for i in range(images.shape[0]):
                # 转换为可视化范围 [0, 1]
                pred_img = np.transpose(pred_np[i], (1, 2, 0))
                true_img = np.transpose(true_np[i], (1, 2, 0))
                
                # 确保值在[0, 1]范围内
                pred_img = np.clip(pred_img, 0, 1)
                true_img = np.clip(true_img, 0, 1)
                
                # 计算PSNR
                batch_psnr = psnr(true_img, pred_img, data_range=1.0)
                psnr_score += batch_psnr
                
                # 计算SSIM (多通道)
                batch_ssim = ssim(true_img, pred_img, data_range=1.0, channel_axis=2)
                ssim_score += batch_ssim
    
    # 计算平均分数
    num_images = num_val_batches * images.shape[0]
    avg_psnr = psnr_score / num_images
    avg_ssim = ssim_score / num_images
    
    model.train()
    return avg_psnr, avg_ssim

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. 创建数据集
    dataset = ISTDDataset_ShadowRemovalUnet(dir_img, dir_mask, dir_true_label, img_scale)

    # 2. 分割为训练/验证部分
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. 创建数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''开始训练:
        训练周期:          {epochs}
        批次大小:          {batch_size}
        学习率:           {learning_rate}
        训练集大小:        {n_train}
        验证集大小:        {n_val}
        保存检查点:        {save_checkpoint}
        设备:             {device.type}
        图像缩放:          {img_scale}
        混合精度:          {amp}
    ''')

    # 4. 设置优化器、损失函数、学习率调度器和AMP的损失缩放
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # 目标：最大化PSNR
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0
    best_psnr = 0.0  # 跟踪最佳PSNR值

    # 5. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'周期 {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.in_channels, \
                    f'网络定义了 {model.in_channels} 个输入通道, ' \
                    f'但加载的图像有 {images.shape[1]} 个通道。请检查 ' \
                    '图像是否正确加载。'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                # 将true_masks转换为float32类型，而不是long类型
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                    loss = loss.float()

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # 评估阶段
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # 使用适合线性恢复任务的评估函数
                        avg_psnr, avg_ssim = evaluate_linear_restoration(model, val_loader, device, amp)
                        
                        # 使用PSNR作为学习率调度器的指标
                        scheduler.step(avg_psnr)
                        
                        logging.info(f'全局步骤 {global_step}: 验证 PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}')
                        logging.info(f'学习率: {optimizer.param_groups[0]["lr"]}')
                        
                        # 保存最佳模型
                        if avg_psnr > best_psnr:
                            best_psnr = avg_psnr
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            state_dict = model.state_dict()
                            state_dict['mask_values'] = dataset.mask_values
                            torch.save(state_dict, str(dir_checkpoint / 'best_model.pth'))
                            logging.info(f'新的最佳模型已保存！PSNR: {best_psnr:.2f}')

        # 计算平均周期损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f'周期 {epoch} 完成。平均损失: {avg_epoch_loss:.4f}')
        
        # 在每个周期结束时评估和保存检查点
        if save_checkpoint:
            avg_psnr, avg_ssim = evaluate_linear_restoration(model, val_loader, device, amp)
            logging.info(f'周期 {epoch} 验证 PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}')
            
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'检查点 {epoch} 已保存！')


def get_args():
    parser = argparse.ArgumentParser(description='在图像和目标掩码上训练UNet')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=300, help='训练周期数')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='批次大小')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='学习率', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='从.pth文件加载模型')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='图像下采样因子')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='用作验证的数据百分比 (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='使用混合精度')
    parser.add_argument('--classes', '-c', type=int, default=3, help='类别数')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备 {device}')

    # 根据数据调整这里
    # RGB图像的in_channels=3
    # out_channels是每个像素要输出的概率数
    model = ShadowRemovalUnet(4, 3)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'网络:\n'
                 f'\t{model.in_channels} 输入通道\n'
                 f'\t{model.out_channels} 输出通道(类别)\n')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'模型从 {args.load} 加载')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('检测到内存不足错误! '
                      '启用检查点以减少内存使用，但这会减慢训练速度。 '
                      '考虑启用AMP (--amp) 以实现快速且内存高效的训练')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )