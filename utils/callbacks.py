import os

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
                 miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = image_ids
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.period = period
        self.writer = SummaryWriter(log_dir)  # 添加TensorBoard写入器

        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.mious = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

    def on_epoch_end(self, epoch, model_eval, **kwargs):
        """改进的on_epoch_end方法，支持多参数传递"""
        # 从kwargs获取指标
        current_miou = kwargs.get('current_miou', 0.0)
        val_loss = kwargs.get('val_loss', 0.0)
        val_f_score = kwargs.get('val_f_score', 0.0)
        
        # 记录到TensorBoard
        self.writer.add_scalar('Val_mIoU', current_miou, epoch)
        if val_loss > 0:
            self.writer.add_scalar('Val_Loss', val_loss, epoch)
        if val_f_score > 0:
            self.writer.add_scalar('Val_F_Score', val_f_score, epoch)
        
        # 只在评估周期执行完整的mIoU计算
        if epoch % self.period == 0 and self.eval_flag:
            # 关键修复：处理DataParallel多卡模型，取出原始model
            original_model = model_eval.module if isinstance(model_eval, torch.nn.DataParallel) else model_eval
            original_model = original_model.module if isinstance(original_model, torch.nn.DataParallel) else original_model
            self.net = original_model
            
            gt_dir = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                # -------------------------------#
                #   从文件中读取图像
                # -------------------------------#
                image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
                image = Image.open(image_path)
                # ------------------------------#
                #   获得预测txt
                # ------------------------------#
                image = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))

            print("Calculate miou.")
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='val miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
            
            # 如果传入了current_miou且比计算的高，使用高的值
            if current_miou > temp_miou:
                self.mious[-1] = current_miou
        else:
            # 非评估周期，使用传入的current_miou
            if current_miou > 0:
                self.mious.append(current_miou)
                self.epoches.append(epoch)
                
                with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                    f.write(str(current_miou))
                    f.write("\n")


# ======================== 添加ImprovedEvalCallback类 ========================
class ImprovedEvalCallback(EvalCallback):
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
                 miou_out_path=".temp_miou_out", eval_flag=True, period=5):
        super().__init__(net, input_shape, num_classes, image_ids, dataset_path, 
                        log_dir, cuda, miou_out_path, eval_flag, period)
        self.miou_history = []  # 记录每轮验证的mIoU
        self.best_miou = 0.0    # 记录最佳mIoU
        self.current_miou = 0.0  # 当前mIoU值
        self.loss_history = []   # 记录验证损失
        self.f_score_history = []  # 记录F-score
        self.best_epoch = 0
        
    def on_epoch_end(self, epoch, model_eval, **kwargs):
        """改进的回调函数，支持多指标"""
        # 从kwargs获取指标
        if 'current_miou' in kwargs:
            self.current_miou = kwargs['current_miou']
            self.miou_history.append(self.current_miou)
            if self.current_miou > self.best_miou:
                self.best_miou = self.current_miou
                self.best_epoch = epoch
                
        if 'val_loss' in kwargs:
            self.loss_history.append(kwargs['val_loss'])
            
        if 'val_f_score' in kwargs:
            self.f_score_history.append(kwargs['val_f_score'])
        
        # 调用父类的on_epoch_end方法
        super().on_epoch_end(epoch, model_eval, **kwargs)
        
        # 记录到tensorboard（父类已经记录，这里可以添加额外记录）
        if hasattr(self, 'writer'):
            # 记录最佳mIoU
            self.writer.add_scalar('Best_mIoU', self.best_miou, epoch)
            
            # 记录F-score趋势
            if len(self.f_score_history) > 0:
                self.writer.add_scalar('Val_F_Score_Trend', self.f_score_history[-1], epoch)


# 兼容旧版本的调用方式
class CustomEvalCallback(ImprovedEvalCallback):
    """为了兼容旧代码而创建的别名类"""
    pass