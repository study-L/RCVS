# 1. 导入必要的库
import torch #深度学习框架
import torch.nn as nn
from einops.einops import rearrange  # 张量操作库einops，实现高效的张量维度操作，比原生PyTorch方法更易读
import cv2  # 图像处理库，主要用于图像处理
from model.matchformer import Matchformer  # 特征匹配网络 Matchformer
from ATGAN.modules.generator import Generator  # 图像融合生成器模块
import matplotlib #绘图库
matplotlib.use('TkAgg')  # TkAgg后端确保matplotlib在GUI环境正常显示
import matplotlib.pyplot as plt  # 用于绘图和可视化
from config.defaultmf import default_cfg  # Matchformer 的配置文件
import numpy as np  # 提供矩阵计算支持
import matplotlib.cm as cm  # 提供颜色映射的支持

#2. 定义伽马变换函数
# 先将 RGB 图像转换为 HSV（色相、饱和度、亮度）空间，对亮度通道进行非线性变换。
#采用HSV色彩空间分离亮度通道（V），避免直接处理RGB通道的颜色失真
def gamma_transform(img, gamma):
    """
    对输入图像进行伽马变换，用于增强亮度或对比度。
    :param img: 输入图像（RGB 或灰度图像）
    :param gamma: 伽马值控制变换的程度（>1 提高亮度；<1 降低亮度）
    :return: 经过伽马变换后的图像
    """
    is_gray = img.ndim == 2 or img.shape[1] == 1
    """
    判断是否为灰度图像,通过检查图像的维度（ndim）或通道数（shape[1]）来判断图像是否为灰度图像。
    如果图像的维度为 2（表示二维数组，即灰度图像）或者通道数为 1（也表示灰度图像），则is_gray为True。
     """
    if is_gray:
# OpenCV 加载彩色图像时默认使用 BGR（蓝、绿、红）顺序，而大多数深度学习模型使用 RGB（红、绿、蓝）顺序。
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 使用cv2.cvtColor函数将灰度图像转换为 BGR 格式的彩色图像。
                                                    # 这是因为后续的 HSV 转换操作在 BGR 格式上进行。

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #使用cv2.cvtColor函数将 BGR 图像转换为 HSV颜色空间。
                                                  #在 HSV 空间中，对亮度的调整更方便。

    illum = hsv[..., 2] / 255.  # 提取亮度通道（V 通道）
    illum = np.power(illum, gamma) #对归一化后的亮度通道应用伽马变换。np.power是 NumPy 中的幂运算函数，
                                  # 根据gamma值对亮度进行非线性调整。
    v = illum * 255.  #将经过伽马变换后的亮度值乘以 255，使其回到 0 - 255 的原始像素值范围。
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 2] = v.astype(np.uint8)  # 将调整并处理后的亮度值重新赋值给 HSV 图像的亮度通道
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  #使用cv2.cvtColor函数将 HSV 图像转换回 BGR 颜色空间
    if is_gray:  #再次检查图像是否原本是灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #如果是，将BGR图像转换回灰度图，以保持输出图像与输入图像的类型一致性
    return img

#3.定义绘制匹配图像的函数，该函数用于绘制匹配图像，将两幅输入图像和融合后的图像显示在一个画布上，并绘制匹配点和匹配线
def make_matching_figure(pred,
                         img0, img1, mkpts0, mkpts1, color,
                         kpts0=None, kpts1=None, text=[], dpi=140, path=None):
    #kpts0 和 kpts1：可选参数，分别是 img0 和 img1 中的关键点坐标，默认值为 None
    #text：可选参数，是一个字符串列表，用于在图中添加文本信息，默认值为空列表。dpi 指定图表的分辨率，默认值为 140
    #可选参数，指定保存图表的文件路径，如果为 None，则返回图表对象，默认值为 Non
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    #使用 assert 语句确保 mkpts0 和 mkpts1 中的匹配点数量一致，如果不一致则抛出 AssertionError 异常
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=dpi)#使用 plt.subplots 函数创建一个包含 1 行 3 列子图的图表
    axes[0].imshow(img0, cmap='gray')#分别在三个子图中显示 img0、img1 和 pred 图像，并使用灰度颜色映射
    axes[1].imshow(img1, cmap='gray')
    axes[2].imshow(pred, cmap='gray')
    for i in range(3):
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)
    #如果 kpts0 不为 None，则确保 kpts1 也不为 None，然后在 img0 和 img1 对应的子图中分别绘制关键点，
    # 关键点用白色（c='w'）的小点（s=2）表示。

    # 绘制匹配点和连接线
    if mkpts0.shape[0]!= 0 and mkpts1.shape[0]!= 0:
        fig.canvas.draw()#绘制图表，以便获取转换信息
        transFigure = fig.transFigure.inverted()#获取图表的逆变换对象，用于将数据坐标转换为图表坐标
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))#将mkpts0中的匹配点坐标从数据坐标转换为图表坐标。
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=(124 / 255, 252 / 255, 0), linewidth=1)
                     #创建一个包含所有匹配点连接线的列表，连接线的颜色为黄绿色（c=(124 / 255, 252 / 255, 0)）取值范围[0-1]
                     for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=(124 / 255, 252 / 255, 0), s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=(124 / 255, 252 / 255, 0), s=4)

    # put txts
    # txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    # fig.text(
    #     0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
    #     fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()#关闭当前的图表，释放相关的资源。这一步很重要，特别是在生成大量图表的情况下，及时关闭图表可以避免内存占用过高。
    else:
        return fig
    #如果 path 不为 None，则将图表保存为文件，文件路径为 path，并关闭图表；否则返回图表对象。

#4 定义颜色空间转换函数，RGB 图像转换为YCrCb颜色空间，转换过程中，对颜色分量进行了数值范围的限制，确保其在 [0,1]之间。
def RGB2YCrCb(rgb_image):
    """
    主要目的： YCrCb 颜色空间在一些图像处理应用中具有优势，例如在视频压缩中，因为人眼对亮度（Y）的敏感度高于对色差（Cr、Cb）的敏感度，
    所以可以对色差分量进行更粗的量化，从而实现数据压缩而不显著影响视觉质量。
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B#Y是亮度分量，它是通过对红、绿、蓝三个通道进行加权求和得到的。人眼对绿色的敏感度最高
    Cr = (R - Y) * 0.713 + 0.5 #Cr表示红色色差，它反映了红色与亮度的差异
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0, 1.0)#clamp 用于将张量中的元素限制在指定的范围内。将三个分量的值限制在[0,1]之间，确保它们符合颜色分量的取值范围
    Cr = Cr.clamp(0.0, 1.0).detach()#detach函数用于分离张量，使其不再参与梯度计算。如果后续不需要对这些分量进行反向传播计算梯度，使用detach可以节省计算资源
    Cb = Cb.clamp(0.0, 1.0).detach()
    return Y, Cb, Cr

# YCrCb 颜色空间转换回 RGB 图像，函数的返回值是转换后的 RGB 图像数据
def YCbCr2RGB(Y, Cb, Cr):
    """
        将YcrCb格式转换为RGB格式
        :param Y: 亮度分量，通常是一个形状为 [B, 1, H, W] 的 PyTorch 张量
        :param Cb: 蓝色色差分量，形状为 [B, 1, H, W] 的 PyTorch 张量
        :param Cr: 红色色差分量，形状为 [B, 1, H, W] 的 PyTorch 张量
        :return: 转换后的 RGB 图像，形状为 [B, 3, H, W] 的 PyTorch 张量
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)#使用torch.cat函数将Y、Cr和Cb三个张量在通道维度dim=1上拼接起来，形成一个形状为 [B,3,H,W]的张量 ycrcb
    B, C, W, H = ycrcb.shape#获取ycrcb张量的形状信息，分别赋值给变量 B、C、W 和 H
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)#形状为 [B * H * W, 3]，即将所有图像的所有像素点展开成一个二维矩阵
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
                       ).to(Y.device)#将矩阵 mat 移动到与张量 Y 相同的设备上
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)

    temp = (im_flat + bias).mm(mat)#将加上偏置后的 im_flat 与转换矩阵 mat 相乘，得到转换后的 RGB 数据 temp，其形状为 [B * H * W, 3]。

    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0, 1.0)
    return out #函数最后返回经过颜色空间转换并处理后的 RGB 图像数据 out，其形状为 [B, C, H, W]，可以用于后续的图像处理或显示等操作。

# 主函数核心流程四步骤
# 1读取两张输入图像。
# 2使用 Matchformer 检测并匹配两幅图像的特征点。
# 3使用生成器 Generator 对输入图像进行融合。
# 4最后可视化结果。
if __name__ == '__main__':  # 确保这个代码块只在脚本作为主程序运行时执行
    # from main import make_matching_figure  # 引入用于显示匹配图像的函数，如果在其他地方定义


    # 加载生成器网络（ATGAN）
    fusion_net = Generator()  # 实例化生成器网络
    # 加载预训练权重，严格匹配权重
    fusion_net.load_state_dict(torch.load(r'./ATGAN/checkpoint/modelc.ckpt', weights_only=True), strict=True)
    fusion_net = fusion_net.eval().cuda()  # 切换到评估模式并移动到GPU

    # 加载匹配网络（Matchformer）
    matcher = Matchformer(default_cfg)  # 实例化匹配网络
    # 加载匹配网络的预训练权重
    matcher.load_state_dict(torch.load(r'modelcc15epoch.ckpt', weights_only=True), strict=False)
    matcher = matcher.eval().cuda()  # 切换到评估模式并移动到GPU

    # 图像路径
    img0_pth = r'H:\data\dates\vr\1 (1).png'  # 可见光图像
    img1_pth = r'H:\data\dates\ir\1 (1).png'  # 红外图像（灰度）

    # 为了后续展示，重新指定图像路径
    img0_pth = r"DN5a.png"
    img1_pth = r"DN5b.png"

    # 读取图像
    img0_raw = cv2.imread(img0_pth)  # 读取可见光图像
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)  # 读取红外图像，注意它是灰度图像
    # 调整图像大小为 (640, 480)，确保输入大小一致且满足模型要求
    img0_raw = cv2.resize(img0_raw, (640, 480))  # input size should be divisible by 8
    img1_raw = cv2.resize(img1_raw, (640, 480))  # 同样调整红外图像大小

    # 将图像 0 的颜色空间从 BGR 转换为 RGB
    img0_raw = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2RGB)
    # 将 NumPy 图像转换为 PyTorch 张量，并进行归一化（像素值从 [0, 255] 转为 [0, 1]）
    img0 = torch.from_numpy(img0_raw)[None].cuda() / 255.  # 添加 batch 维度并移动到GPU
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.  # 同上

    # 调整图像的维度顺序，转换为 [n, c, h, w] 格式，符合 PyTorch 的输入要求
    img0 = rearrange(img0, 'n h w c ->  n c h w')  # 把 img0 的维度调整为适合网络输入的形状

    # 将图像 0 从 RGB 转换为 YCrCb 颜色空间（色度、亮度分量分离）
    vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img0)

    # 构建输入数据的字典
    batch = {'image0': vi_Y, 'image1': img1}  # 使用可见光图像的 Y 分量和红外图像作为输入

    # 前向传播计算匹配点
    with torch.no_grad():  # 在评估模式下禁用梯度计算，提高效率
        matcher(batch)  # 执行匹配网络的前向计算
        # 提取匹配点（通过网络预测的匹配位置）
        mkpts0 = batch['mkpts0_c'].cpu().numpy()  # 图像 0 的匹配点（坐标）
        mkpts1 = batch['mkpts1_c'].cpu().numpy()  # 图像 1 的匹配点（坐标）
        mconf = batch['mconf'].cpu().numpy()  # 匹配置信度

    conf = batch['conf_matrix'].cpu().numpy()  # 匹配的置信度矩阵

    # 使用 RANSAC 算法计算单应性矩阵（计算两个图像之间的几何变换）
    h, prediction = cv2.findHomography(mkpts1, mkpts0, cv2.USAC_MAGSAC, 5, confidence=0.99999, maxIters=100000)
    # 基于预测的匹配点进行筛选，去除错误的匹配点
    prediction = np.array(prediction, dtype=bool).reshape([-1])
    mkpts00 = mkpts0  # 保存原始匹配点
    mkpts11 = mkpts1
    mkpts0 = mkpts0[prediction]  # 筛选后的匹配点
    mkpts1 = mkpts1[prediction]

    # 对图像 1（红外图像）进行透视变换，使其与图像 0 对齐
    img11 = cv2.warpPerspective(img1_raw, h, (640, 480))  # 使用单应性矩阵进行图像变换

    # 将变换后的图像转换为 PyTorch 张量
    img11 = torch.from_numpy(img11)[None][None].cuda() / 255.

    # 读取图像 0 并进行预处理，保持与图像 1 一致
    im0 = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    im0 = cv2.resize(im0, (640, 480))  # 确保尺寸一致
    im0 = torch.from_numpy(im0)[None][None].cuda() / 255.

    # 使用生成器网络对变换后的图像和图像 0 进行融合
    fuse = fusion_net(img11, vi_Y)
    # 也可以尝试简单的图像叠加： fuse = (img11 + im0) * 0.5

    # 将融合后的结果从 YCbCr 转换回 RGB 颜色空间
    fuse = YCbCr2RGB(fuse, vi_Cb, vi_Cr)
    # 获取融合后的图像，并将其从 GPU 转移到 CPU，进行后续处理
    fuse = fuse.detach().cpu()[0]
    # 调整维度顺序，从 [c, h, w] 转为 [h, w, c]，便于显示
    fuse = rearrange(fuse, ' c h w ->  h w c').detach().cpu().numpy()

    print(mkpts0.shape)  # 打印匹配点的形状

    # 使用匹配点绘制匹配图像
    fig = make_matching_figure(fuse, img0_raw, img1_raw, mkpts0, mkpts1, color=None)
    plt.show()  # 显示图像

    # 你也可以直接保存图像
    cv2.imwrite(r'D:\RCVS\fuse.png', fuse * 255)
