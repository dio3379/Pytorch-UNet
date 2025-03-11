import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    # 获取文件后缀名
    ext = splitext(filename)[1]
    # 如果文件后缀名为.npy，则使用numpy加载图像
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    # 如果文件后缀名为.pt或.pth，则使用torch加载图像，并将其转换为numpy数组
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    # 否则，使用PIL加载图像
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    # 根据给定的索引、掩码目录和掩码后缀，获取掩码文件
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    # 加载掩码文件
    mask = np.asarray(load_image(mask_file))
    # 如果掩码是二维的，返回掩码的唯一值
    if mask.ndim == 2:
        return np.unique(mask)
    # 如果掩码是三维的，将掩码展平，然后返回掩码的唯一值
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    # 如果掩码不是二维或三维的，抛出异常
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        # 初始化函数，传入图片目录、掩码目录、缩放比例和掩码后缀
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        # 断言缩放比例在0和1之间
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # 获取图片目录下的所有文件名，并去除后缀
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        # 如果没有找到文件，抛出异常
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        # 打印日志信息，创建数据集
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        # 打印日志信息，扫描掩码文件以确定唯一值
        logging.info('Scanning mask files to determine unique values')
        # 使用多进程池，遍历所有文件名，获取掩码文件中的唯一值
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        # 将所有唯一值合并，并排序
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # 打印日志信息，显示唯一值
        logging.info(f'Unique mask values: {self.mask_values}')

    # 定义一个方法，用于返回对象的长度
    def __len__(self):
        # 返回对象的ids属性的长度
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        # 获取图片的宽度和高度
        w, h = pil_img.size
        # 计算新的宽度和高度
        newW, newH = int(scale * w), int(scale * h)
        # 断言新的宽度和高度大于0，否则缩放后的图片没有像素
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # 根据是否是mask来选择不同的插值方式
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # 将图片转换为numpy数组
        img = np.asarray(pil_img)

        # 如果是mask
        if is_mask:
            # 创建一个全为0的mask
            mask = np.zeros((newH, newW), dtype=np.int64)
            # 遍历mask_values
            for i, v in enumerate(mask_values):
                # 如果图片是灰度图
                if img.ndim == 2:
                    # 将图片中等于v的像素值赋值为i
                    mask[img == v] = i
                # 如果图片是彩色图
                else:
                    # 将图片中所有通道都等于v的像素值赋值为i
                    mask[(img == v).all(-1)] = i

            # 返回mask
            return mask

        else:
            # 如果图片是灰度图
            if img.ndim == 2:
                # 在第0维添加一个维度
                img = img[np.newaxis, ...]
            # 如果图片是彩色图
            else:
                # 将图片的维度从(h, w, c)转换为(c, h, w)
                img = img.transpose((2, 0, 1))

            # 如果图片的像素值大于1
            if (img > 1).any():
                # 将图片的像素值除以255.0
                img = img / 255.0

            # 返回图片
            return img

    def __getitem__(self, idx):
        # 获取指定索引的ID
        name = self.ids[idx]
        # 获取对应ID的mask文件
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        # 获取对应ID的图片文件
        img_file = list(self.images_dir.glob(name + '.*'))

        # 断言只有一个图片文件
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # 断言只有一个mask文件
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        # 加载mask文件
        mask = load_image(mask_file[0])
        # 加载图片文件
        img = load_image(img_file[0])

        # 断言图片和mask大小相同
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # 对图片进行预处理
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        # 对mask进行预处理
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        # 返回预处理后的图片和mask
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

class ISTDDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, transform=None):
        """
        ISTD数据集加载器，用于阴影检测
        
        参数:
            images_dir (str): 包含图像的目录路径
            mask_dir (str): 包含阴影掩码的目录路径
            scale (float): 调整图像大小的比例因子（0到1之间）
            transform: 可选的图像变换
        """
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, '缩放比例必须在0和1之间'
        self.scale = scale
        self.transform = transform
        
        # 获取图像目录中所有图像文件的列表
        self.img_files = [file for file in listdir(images_dir) 
                         if isfile(join(images_dir, file)) and not file.startswith('.')]
        
        # 获取掩码目录中所有掩码文件的列表
        self.mask_files = [file for file in listdir(mask_dir) 
                          if isfile(join(mask_dir, file)) and not file.startswith('.')]
        
        # 对文件进行排序以确保对齐
        self.img_files.sort()
        self.mask_files.sort()
        
        # 检查图像和掩码数量是否相同
        if len(self.img_files) != len(self.mask_files):
            raise RuntimeError(f'图像数量（{len(self.img_files)}）与'
                               f'掩码数量（{len(self.mask_files)}）不匹配')
        
        # 如果没有找到文件，抛出错误
        if not self.img_files:
            raise RuntimeError(f'在{images_dir}中没有找到输入文件，请确保您放置了图像')
        
        logging.info(f'创建包含{len(self.img_files)}个样本的ISTD数据集')
        logging.info('ISTD数据集是二分类的：0表示非阴影，1表示阴影')
        
        # 为了兼容train_model函数中的保存检查点功能，添加mask_values属性
        # 对于二分类任务，mask值只有0和1
        self.mask_values = [0, 1]

    def __len__(self):
        """返回数据集中的图像数量"""
        return len(self.img_files)
    
    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        """
        预处理图像（调整大小和归一化/二值化）
        
        参数:
            pil_img: 要预处理的PIL图像
            scale: 调整大小的比例因子
            is_mask: 图像是否是掩码
        
        返回:
            预处理后的图像，作为numpy数组
        """
        # 获取图像尺寸
        w, h = pil_img.size
        # 计算新尺寸
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, '缩放比例太小，调整大小后的图像将没有像素'
        
        # 调整图像大小（对于掩码使用NEAREST以保留二进制值，对于常规图像使用BICUBIC）
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        
        if is_mask:
            # 对于ISTD，将掩码转换为二进制（0表示非阴影，1表示阴影）
            # 在ISTD数据集中，掩码值通常为0（非阴影）或255（阴影）
            # 我们将255转换为1用于二进制分割
            mask = np.zeros((newH, newW), dtype=np.int64)
            if img.ndim == 2:
                # 如果是灰度掩码
                mask[img > 0] = 1  # 将任何非零值转换为1（阴影）
            else:
                # 如果是RGB掩码，首先转换为灰度
                img_gray = np.mean(img, axis=2)
                mask[img_gray > 0] = 1
            
            return mask
        else:
            # 处理输入图像
            if img.ndim == 2:
                # 如果是灰度图，添加通道维度
                img = img[np.newaxis, ...]
            else:
                # 从HWC转换为CHW格式
                img = img.transpose((2, 0, 1))
            
            # 如果像素值在[0-255]范围内，则归一化
            if (img > 1).any():
                img = img / 255.0
                
            return img
    
    def __getitem__(self, idx):
        """从数据集中获取样本（图像和掩码对）"""
        # 获取图像和掩码文件名
        img_filename = self.img_files[idx]
        mask_filename = self.mask_files[idx]
        
        # 加载图像和掩码
        img_path = join(self.images_dir, img_filename)
        mask_path = join(self.mask_dir, mask_filename)
        
        img = load_image(img_path)
        mask = load_image(mask_path)
        
        # 检查图像和掩码是否具有相同的尺寸
        if img.size != mask.size:
            logging.warning(f"图像{img_filename}的尺寸{img.size}与掩码{mask_filename}的尺寸{mask.size}不匹配")
            # 如果需要，将掩码调整为与图像相同的尺寸
            mask = mask.resize(img.size, Image.NEAREST)
        
        # 预处理图像和掩码
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        
        # 如果提供了额外的变换，则应用
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'img_file': img_filename,
            'mask_file': mask_filename
        }

class CarvanaDataset(BasicDataset):
    # 初始化CarvanaDataset类，继承BasicDataset类
    def __init__(self, images_dir, mask_dir, scale=1):
        # 调用父类BasicDataset的初始化方法
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')



class ISTDDataset_ShadowRemovalUnet(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, dir_true_label: str, scale: float = 1.0, transform=None):
        """
        ISTD数据集加载器，用于阴影去除任务。
        
        参数:
            images_dir (str): 包含阴影图像的目录路径
            mask_dir (str): 包含阴影掩码的目录路径
            dir_true_label (str): 包含去除阴影后图片的目录路径
            scale (float): 调整图像大小的比例因子（0到1之间）
            transform: 可选的图像变换
        """
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.dir_true_label = Path(dir_true_label)
        assert 0 < scale <= 1, '缩放比例必须在0和1之间'
        self.scale = scale
        self.transform = transform
        
        # 获取图像目录中所有图像文件的列表
        self.img_files = [file for file in listdir(images_dir) 
                        if isfile(join(images_dir, file)) and not file.startswith('.')]
        
        # 获取掩码目录中所有掩码文件的列表
        self.mask_files = [file for file in listdir(mask_dir) 
                        if isfile(join(mask_dir, file)) and not file.startswith('.')]
        
        # 获取去除阴影后图片目录中所有图片文件的列表
        self.true_label_files = [file for file in listdir(dir_true_label)
                        if isfile(join(dir_true_label, file)) and not file.startswith('.')]
        
        # 对文件进行排序以确保对齐
        self.img_files.sort()
        self.mask_files.sort()
        self.true_label_files.sort()
        
        # 检查图像、掩码和真实标签数量是否相同
        if len(self.img_files) != len(self.mask_files) or len(self.img_files) != len(self.true_label_files):
            raise RuntimeError(f'图像数量（{len(self.img_files)}）与'
                            f'掩码数量（{len(self.mask_files)}）或'
                            f'真实标签数量（{len(self.true_label_files)}）不匹配')
        
        # 如果没有找到文件，抛出错误
        if not self.img_files:
            raise RuntimeError(f'在{images_dir}中没有找到输入文件，请确保您放置了图像')
        
        logging.info(f'创建包含{len(self.img_files)}个样本的ISTD数据集')
        logging.info('ISTD数据集做去除阴影的Unet线性恢复任务')
        
        # 为了兼容train_model函数中的保存检查点功能，添加mask_values属性
        # 这里不需要二值mask_values，因为我们不是做分割任务，而是做线性恢复任务
        # 我们将0表示非阴影区域，1表示阴影区域
        self.mask_values = [0, 1, 2]

    def __len__(self):
        """返回数据集中的图像数量"""
        return len(self.img_files)
    
    @staticmethod
    def preprocess(pil_img, scale, is_mask, is_true_label=False):
        """
        预处理图像（调整大小和归一化/二值化）
        
        参数:
            pil_img: 要预处理的PIL图像
            scale: 调整大小的比例因子
            is_mask: 图像是否是掩码
            is_true_label: 图像是否是真实标签（无阴影图像）
        
        返回:
            预处理后的图像，作为numpy数组
        """
        # 获取图像尺寸
        w, h = pil_img.size
        # 计算新尺寸
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, '缩放比例太小，调整大小后的图像将没有像素'
        
        # 调整图像大小
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        
        if is_mask:
            # 对于掩码，将其转换为二进制（0表示非阴影，1表示阴影）
            mask = np.zeros((newH, newW), dtype=np.float32)
            if img.ndim == 2:
                # 如果是灰度掩码
                mask[img > 0] = 1  # 将任何非零值转换为1（阴影）
            else:
                # 如果是RGB掩码，首先转换为灰度
                img_gray = np.mean(img, axis=2)
                mask[img_gray > 0] = 1
            
            return mask
        else:
            # 处理输入图像或真实标签
            if img.ndim == 2:
                # 如果是灰度图，添加通道维度
                img = img[np.newaxis, ...]
            else:
                # 从HWC转换为CHW格式
                img = img.transpose((2, 0, 1))
            
            # 如果像素值在[0-255]范围内，则归一化
            if (img > 1).any():
                img = img / 255.0
                
            return img
    
    def __getitem__(self, idx):
        """从数据集中获取样本（带阴影图像、阴影掩码、无阴影图像的三元组）"""
        # 获取图像、掩码和真实标签文件名
        img_filename = self.img_files[idx]
        mask_filename = self.mask_files[idx]
        true_label_filename = self.true_label_files[idx]
        
        # 加载图像、掩码和真实标签
        img_path = join(self.images_dir, img_filename)
        mask_path = join(self.mask_dir, mask_filename)
        true_label_path = join(self.dir_true_label, true_label_filename)
        
        img = load_image(img_path)
        mask = load_image(mask_path)
        true_label = load_image(true_label_path)
        
        # 检查图像、掩码和真实标签是否具有相同的尺寸
        if img.size != mask.size:
            logging.warning(f"图像{img_filename}的尺寸{img.size}与掩码{mask_filename}的尺寸{mask.size}不匹配")
            # 将掩码调整为与图像相同的尺寸
            mask = mask.resize(img.size, Image.NEAREST)
        
        if img.size != true_label.size:
            logging.warning(f"图像{img_filename}的尺寸{img.size}与真实标签{true_label_filename}的尺寸{true_label.size}不匹配")
            # 将真实标签调整为与图像相同的尺寸
            true_label = true_label.resize(img.size, Image.BICUBIC)
        
        # 预处理图像、掩码和真实标签
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        true_label = self.preprocess(true_label, self.scale, is_mask=False, is_true_label=True)
        
        # 如果提供了额外的变换，则应用（需要修改transform以支持多个图像）
        if self.transform:
            transformed = self.transform(image=img, mask=mask, true_label=true_label)
            img = transformed['image']
            mask = transformed['mask']
            true_label = transformed['true_label']
        
        # 创建模型输入格式：4通道（3通道RGB图像 + 1通道阴影掩码）
        # 将掩码扩展为[1, H, W]形状并连接到图像
        mask_expanded = mask[None, ...]  # 形状变为[1, H, W]
        model_input = np.concatenate([img, mask_expanded], axis=0)  # 形状变为[4, H, W]
        
        return {
            'image': torch.as_tensor(model_input.copy()).float().contiguous(),  # 4通道输入
            'mask': torch.as_tensor(true_label.copy()).float().contiguous(),  # 3通道真实标签
            'img_file': img_filename,
            'mask_file': mask_filename,
            'true_label_file': true_label_filename
        }