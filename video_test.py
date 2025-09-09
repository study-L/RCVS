import os
import re
import cv2
import numpy as np
import torch
from einops import rearrange

# 导入自定义模型（需确保模型文件存在于指定路径）
from model.matchformer import Matchformer  # 特征匹配模型（用于计算图像配准矩阵）
from config.defaultmf import default_cfg   # Matchformer模型的配置参数
from ATGAN.modules.generator import Generator  # 图像融合生成器（融合红外与可见光信息）


# -------------------------- 颜色空间转换函数 --------------------------
def RGB2YCrCb(rgb_image):
    """将RGB图像转换为YCrCb颜色空间
    Y通道代表亮度，Cr和Cb代表色度信息
    输入：
        rgb_image: RGB格式的图像张量，形状为[B, C, H, W]
    输出：
        Y, Cb, Cr: 分离的亮度和色度通道，每个都是[B, 1, H, W]形状
    """
    R = rgb_image[:, 0:1]  # 提取红色通道，保持4D张量形状
    G = rgb_image[:, 1:2]  # 提取绿色通道
    B = rgb_image[:, 2:3]  # 提取蓝色通道

    # 计算亮度通道Y（人眼对绿色最敏感，所以权重最高）
    Y = 0.299 * R + 0.587 * G + 0.114 * B #反映图像明暗信息
    # 计算色度通道Cr（红色差异）和Cb（蓝色差异）
    Cr = (R - Y) * 0.713 + 0.5  # 标准化到0.5附近  #反映蓝色分量与亮度的差异
    Cb = (B - Y) * 0.564 + 0.5  # 标准化到0.5附近

    # 确保所有像素值都在0到1之间，图像处理当中常将像素值归一化到0-1范围避免溢出
    Y = Y.clamp(0.0, 1.0) #只有Y通道（亮度）需要参与后续的融合网络计算
    Cr = Cr.clamp(0.0, 1.0).detach()  # detach()断开梯度计算
    Cb = Cb.clamp(0.0, 1.0).detach() #Cr和Cb通道（色度）只需要保持原始信息，不需要通过网络反向传播梯度，不用跟踪他们的变化

    return Y, Cb, Cr


def YCbCr2RGB(Y, Cb, Cr):
    """将YCrCb颜色空间转换回RGB格式，核心目的是融合完成后，将亮度通道与原始色度通道重组，恢复彩色图像
    输入：
        Y: 亮度通道，形状[B, 1, H, W]
        Cb: 蓝色色度通道，形状[B, 1, H, W]
        Cr: 红色色度通道，形状[B, 1, H, W]
    输出：
        RGB格式的图像张量，形状[B, 3, H, W]，数值范围[0, 1]
    """
    #1 将三个通道合并为YCrCb格式
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape  # 获取批量大小、通道数、宽度和高度

    #2重塑张量以便进行矩阵运算：从[B, C, H, W]变为[B*H*W, 3]
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    # 将YCrCb颜色空间的图像张量转换成适合进行矩阵运算的扁平格式

    #3. 定义YCrCb→RGB的转换矩阵（ITU-R BT.601标准）
    # 矩阵作用：将亮度和色度差异还原为RGB三通道的数值
    mat = torch.tensor([[1.0, 1.0, 1.0],        # Y通道对RGB的贡献（均为1，亮度直接保留）
                        [1.403, -0.714, 0.0],   # Cr通道对RGB的贡献（主要影响红色）
                        [0.0, -0.344, 1.773]]).to(Y.device)  # Cb通道对RGB的贡献（主要影响蓝色）
    # 偏移量：还原Cr/Cb的归一化（之前+0.5，此处需减去）
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)  # 0.0/255是Y通道无偏移

    # 4. 执行转换计算（矩阵乘法+偏移）
    temp = (im_flat + bias).mm(mat)  # 先加偏移，再乘转换矩阵

    # 重塑回原始图像形状：[B*H*W, 3] -> [B, H, W, 3] -> [B, 3, H, W]
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)

    #6 限制数值范围在0到1之间，确保有效的RGB值 避免负数或超1导致图像失真
    return out.clamp(0, 1.0)


# -------------------------- 核心处理函数 --------------------------
def process_frame_pair(vis_frame, ir_frame, fusion_net, matcher):
    """
    单帧红外-可见光图像对处理流程： 尺寸统一→特征匹配→图像配准→亮度融合→色彩恢复

    参数：
        vis_frame: numpy.ndarray，原始可见光图像，RGB格式（HWC）
        ir_frame: numpy.ndarray，原始红外图像，灰度格式（HW，无色彩信息）
        fusion_net: Generator实例，预训练的图像融合网络（输入：红外+可见光亮度，输出：融合亮度）
        matcher: Matchformer实例，预训练的特征匹配网络（用于计算红外→可见光的配准矩阵）
    返回：
        fused_frame: numpy.ndarray，融合后的彩色图像，BGR格式（OpenCV显示需BGR）
        aligned_ir: numpy.ndarray，配准后的红外图像（灰度），与可见光尺寸/视角一致
        h: numpy.ndarray或None，配准用的单应性矩阵（3x3），None表示配准失败
        match_count: int，有效特征匹配点数量（反映配准可靠性，越多越可靠）
        mkpts0: numpy.ndarray，可见光图像中的特征匹配点坐标（Nx2）
        mkpts1: numpy.ndarray，红外图像中的特征匹配点坐标（Nx2，与mkpts0一一对应）
    """
    try:
        # 1. 图像预处理：统一尺寸（模型输入要求固定尺寸）
        vis_frame_rgb = cv2.resize(vis_frame, (640, 480))
        ir_frame = cv2.resize(ir_frame, (640, 480))

        # 2. 格式转换：numpy数组→PyTorch张量（模型输入要求）
        # 可见光：[H, W, 3] → [1, H, W, 3]（加批量维度）→ 归一化到[0,1] → 移至GPU
        img0 = torch.from_numpy(vis_frame_rgb)[None].cuda() / 255.  # 可见光图像，[None]是添加批次维度
        img1 = torch.from_numpy(ir_frame)[None][None].cuda() / 255.  # 红外图像，添加批次和通道维度

        # 调整维度顺序：从[1, H, W, 3]numpy默认   变为[1, 3, H, W]（PyTorch标准格式）
        img0 = rearrange(img0, 'n h w c -> n c h w')

        # 3. 颜色空间转换：RGB→YCrCb（分离亮度和色度）
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img0)

        # 4. 特征匹配：计算红外与可见光的对应点（用于配准）
        # 准备匹配模型 Matchformer输入（仅需亮度通道，色彩不影响特征匹配）
        batch = {'image0': vi_Y, 'image1': img1}   # image0：可见光亮度，image1：红外灰度

        # 在推理模式下运行模型（不计算梯度，节省内存和计算资源）
        with torch.no_grad():
            # 使用匹配模型找出两幅图像之间的特征对应点
            matcher(batch)

            # 获取匹配点坐标并转换为NumPy数组（从GPU移动到CPU）
            # 提取匹配点：GPU张量→CPU numpy数组（便于后续OpenCV计算）
            mkpts0 = batch['mkpts0_c'].cpu().numpy()  # 可见光图像中的特征点坐标
            mkpts1 = batch['mkpts1_c'].cpu().numpy()  # 红外图像中的对应特征点坐标

         # 5. 图像配准：计算单应性矩阵（透视变换矩阵）
            h = None  # 单应性矩阵初始化（描述红外→可见光的透视变换关系

            # 单应性矩阵计算需至少4个匹配点（3点确定平面，4点才能解透视变换）
            if len(mkpts0) >= 4:
                try:
                    # 调用OpenCV的鲁棒单应性矩阵估计（MAGSAC算法：抗噪能力强于RANSAC）
                    # 单应性矩阵描述了从红外图像到可见光图像的透视变换
                    h, prediction = cv2.findHomography(
                        mkpts1, mkpts0,  # 输入：源点（红外），目标点（可见光）
                        cv2.USAC_MAGSAC,
                        5,  # 重投影误差阈值（像素），小于此值的点被认为是内点
                        confidence=0.99,  # 置信度：99%概率得到正确矩阵
                        maxIters=10000   #最大迭代次数：确保找到最优解
                    )
          # 过滤外点：保留有效匹配点（prediction为布尔数组，True表示内点）
                    if h is not None:
                        prediction = np.array(prediction, dtype=bool).reshape([-1])
                        mkpts0 = mkpts0[prediction]  # 保留可见光内点
                        mkpts1 = mkpts1[prediction]  # 保留红外内点
                except Exception as e:
                    # 异常处理：如匹配点分布退化（共线）导致无法计算矩阵
                    print(f"单应性计算错误: {e}")
                    h = None  # 如果计算失败，保持h为None

            #6图像融合：配准后红外 + 可见光亮度
            # 如果成功计算出配准变换矩阵
            if h is not None:
                # 6.1使用单应性矩阵将红外图像变换到可见光图像的坐标系
                img11 = cv2.warpPerspective(
                    ir_frame,  # 输入：原始红外图像
                    h,  # 变换矩阵：红外→可见光
                    (640, 480)
                )# 输出尺寸：与可见光一致

                # 6.2将配准后的红外图像转换为PyTorch张量，转换为模型输入格式（批量+通道维度，归一化）
                img11_tensor = torch.from_numpy(img11)[None][None].cuda() / 255.


                # 6.3使用预训练网络融合 配准后的红外图像和可见光图像的亮度通道
                # 输入：配准后的红外（含热目标）+ 可见光亮度（含细节）
                # 输出：融合后的亮度通道（兼具两者信息
                fuse = fusion_net(img11_tensor, vi_Y)

                # 6.4 颜色恢复：将融合后的亮度与原始色度通道结合，转换回RGB
                # 原理：保留可见光的色度（色彩），替换为融合后的亮度（细节+热目标）
                fuse = YCbCr2RGB(fuse, vi_Cb, vi_Cr)

                # 6.5 处理融合结果以便显示：PyTorch张量→numpy数组（用于OpenCV显示），调整维度顺序，转换为numpy数组
                fuse = fuse.detach().cpu()[0]  #detach() 从PyTorch的计算图中分离出来，不再跟踪梯度（因为只是要显示结果，不需要再训练或计算梯度
                fuse = rearrange(fuse, 'c h w -> h w c').numpy()  # [C, H, W] -> [H, W, C] opencv需要高度，宽度，通道格式

                # 转换为8位无符号整数格式（0-255）和BGR颜色顺序（OpenCV显示格式）
                fused_frame = (fuse * 255).astype(np.uint8)#fuse 是模型输出的融合图像数据
                fused_frame = cv2.cvtColor(fused_frame, cv2.COLOR_RGB2BGR) #模型处理的时候用的是RGB，OpenCV用BGR,
                #将颜色通道顺序从RGB转换为BGR，确保图像显示时颜色正确

                # 返回融合结果和配准信息
                # fused_frame: 融合后的彩色图像（BGR格式，适合OpenCV显示）
                # img11: 配准后的红外图像（灰度图，与可见光视角对齐）
                # h: 单应性矩阵（3×3），描述红外→可见光的透视变换关系
                # len(mkpts0): 有效特征匹配点数量（衡量配准质量）
                # mkpts0: 可见光图像中的特征点坐标数组 [N,2],N表特征点数量，2表示每个特征点由两个数值组成
                # mkpts1: 红外图像中对应的特征点坐标数组 [N,2]
                return fused_frame, img11, h, len(mkpts0), mkpts0, mkpts1
            else:
                # 如果无法配准（匹配点不足或单应性计算失败），使用简单的加权融合
                # 将红外图像从灰度转换为BGR，以便与可见光图像融合
                ir_bgr = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
                # 加权融合：当无法精确配准时的备选方案
                # cv2.addWeighted函数实现图像加权融合，公式为：dst = src1*alpha + src2*beta + gamma
                # - vis_frame: 第一个输入图像（可见光图像）
                # - 0.7: 第一个图像的权重（alpha），表示可见光占70%
                # - ir_bgr: 第二个输入图像（转换为BGR的红外图像）
                # - 0.3: 第二个图像的权重（beta），表示红外占30%
                # - 0: 亮度调整值（gamma），此处不额外调整亮度
                simple_fused = cv2.addWeighted(vis_frame, 0.7, ir_bgr, 0.3, 0)

                return simple_fused, ir_frame, None, 0, [], []

    except Exception as e:
        # 如果处理过程中出现任何错误，打印错误信息并返回原始图像
        print(f"处理帧时出错: {e}")
        return vis_frame, ir_frame, None, 0, [], []


def init_models():
    """初始化融合网络和特征匹配模型
     功能：加载预训练权重，设置为评估模式（禁用训练特有的层），并移至GPU加速
     输出：
         fusion_net: 初始化好的图像融合网络（Generator）
         matcher: 初始化好的特征匹配模型（Matchformer）
     """
    print("正在加载模型...")

    # 初始化融合网络（ATGAN生成器）
    fusion_net = Generator()  # 实例化生成器（网络结构定义在ATGAN/modules/generator.py）
    # 加载预训练权重：strict=True表示权重文件的键必须与网络参数完全匹配
    # 若训练时网络结构有修改，需设为strict=False（允许部分权重不匹配）
    fusion_net.load_state_dict(torch.load(r'./ATGAN/checkpoint/modelc.ckpt'), strict=True)
    # 设置为评估模式：禁用Dropout层、冻结BatchNorm统计量（确保推理结果稳定）
    #禁用Dropout层的原因（训练时防过拟合，推理时要用全部神经元）
    # .cuda()将模型移至GPU（若无GPU，替换为.cpu()）
    fusion_net = fusion_net.eval().cuda()

    # 初始化特征匹配模型（Matchformer）
    matcher = Matchformer(config=default_cfg)  # 实例化匹配器（传入默认配置参数）
    # 加载预训练权重：strict=False允许权重文件与网络参数部分匹配（适配不同训练版本）
    matcher.load_state_dict(torch.load(r'modelcc15epoch.ckpt'), strict=False)
    matcher = matcher.eval().cuda()  # 评估模式+移至GPU

    return fusion_net, matcher


class DualModalLoader:
    """双模态图像加载器，用于加载和管理红外和可见光视频帧
     核心功能：按视频ID匹配两模态的对应帧，按顺序读取帧，统计帧数量
     假设图像命名格式："视频ID (帧号).扩展名"（如"1 (1).jpg"表示视频1的第1帧）
     """
    def __init__(self, ir_dir, visible_dir):
        """初始化加载器
        输入：
            ir_dir: 红外图像目录路径
            visible_dir: 可见光图像目录路径
        """
        self.ir_dir = ir_dir#初始化属性
        self.visible_dir = visible_dir
        self.current_video = None   # 当前处理的视频ID（如"1"）
        self.frame_list = []  # 共同帧编号列表
        self.current_index = 0   # 当前读取的帧索引（从0开始）
        self.ir_map, self.visible_map = {}, {}  # 帧号→文件名的映射（便于快速查找）

        # 检查目录是否存在
        if not os.path.exists(ir_dir):
            raise FileNotFoundError(f"红外目录不存在: {ir_dir}")
        if not os.path.exists(visible_dir):
            raise FileNotFoundError(f"可见光目录不存在: {visible_dir}")

    def load_video(self, video_id):
        """找出红外和可见光目录中属于当前视频的共同帧号（内部辅助方法）
           逻辑：通过正则表达式解析文件名，提取帧号，取两者的交集
           输出：
               sorted list: 按升序排列的共同帧编号列表（如[1,2,3,...,N]）
           """
        # 1. 统一视频ID格式为字符串（避免传入int型ID导致文件名匹配失败，如1 vs "1"）
        self.current_video = str(video_id)

        # 2. 调用内部辅助方法，找到当前视频在双模态目录中的共同帧号列表
        #    （_find_common_frames会同时更新self.ir_map和self.visible_map映射字典）
        self.frame_list = self._find_common_frames()

        # 3. 重置帧索引为0（新视频从第1帧开始读取，避免继承上一个视频的索引）
        self.current_index = 0

        # 4. 检查是否找到共同帧：无共同帧则加载失败，返回False并提示
        if not self.frame_list:
            print(f"警告: 视频ID={video_id} 在红外/可见光目录中无匹配帧（可能帧号不对应或文件缺失）")
            return False

        # 5. 加载成功：打印视频ID和总帧数，返回True
        print(f"视频加载成功 | 视频ID: {video_id} | 总帧数: {len(self.frame_list)}（每帧含红外+可见光数据）")
        return True

    def _find_common_frames(self):
        """找出红外和可见光目录中共同的帧（基于文件名匹配）
        输出：
            sorted list: 排序后的共同帧编号列表
        """
        ir_frames, visible_frames = {}, {}

        # 扫描红外目录，找出符合命名规则的文件
        for filename in os.listdir(self.ir_dir):
            # 使用正则表达式匹配文件名格式：视频ID(帧号).扩展名
            match = re.match(r'^' + re.escape(self.current_video) + r'\s*\((\d+)\)\.(jpg|png)$',
                             filename, re.IGNORECASE)
            if match:
                frame_num = int(match.group(1))  # 提取帧号
                ir_frames[frame_num] = filename  # 存储帧号到文件名的映射

        # 扫描可见光目录，同样找出符合命名规则的文件
        for filename in os.listdir(self.visible_dir):
            match = re.match(r'^' + re.escape(self.current_video) + r'\s*\((\d+)\)\.(jpg|png)$',
                             filename, re.IGNORECASE)
            if match:
                frame_num = int(match.group(1))
                visible_frames[frame_num] = filename

        # 找出两个目录中都存在的帧号（交集）
        common_frames = sorted(set(ir_frames.keys()) & set(visible_frames.keys()))

        # 建立帧编号到文件名的映射，便于后续读取
        self.ir_map = {frame: ir_frames[frame] for frame in common_frames}
        self.visible_map = {frame: visible_frames[frame] for frame in common_frames}

        return common_frames

    def read_frame(self):
        """读取下一对红外和可见光图像
        输出：
            ir_img: 红外图像（灰度），numpy数组
            visible_img: 可见光图像（RGB），numpy数组
            如果没有更多帧或读取失败，返回None, None
        """
        # 检查是否还有帧可读
        if not self.frame_list or self.current_index >= len(self.frame_list):
            return None, None

        # 获取当前帧编号并更新索引
        frame_num = self.frame_list[self.current_index]
        self.current_index += 1

        # 获取对应的文件名
        ir_file = self.ir_map.get(frame_num)
        visible_file = self.visible_map.get(frame_num)
        if not ir_file or not visible_file:
            return None, None

        # 构建完整文件路径
        ir_path = os.path.join(self.ir_dir, ir_file)
        visible_path = os.path.join(self.visible_dir, visible_file)

        # -------------------------- 5. 读取红外图像（灰度模式） --------------------------
        # cv2.IMREAD_GRAYSCALE：强制以灰度模式读取（红外图像无色彩信息，单通道更高效）
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        # 读取失败（文件损坏/路径错误）：打印错误，返回双None
        if ir_img is None:
            print(f"红外图像读取失败: {ir_path}（可能文件损坏或路径错误）")
            return None, None

        # -------------------------- 6. 读取可见光图像（彩色模式） --------------------------
        # cv2.IMREAD_COLOR：强制以彩色模式读取（默认返回BGR格式，需后续转RGB）
        visible_img = cv2.imread(visible_path, cv2.IMREAD_COLOR)
        # 读取失败：打印错误，返回双None
        if visible_img is None:
            print(f"可见光图像读取失败: {visible_path}（可能文件损坏或路径错误）")
            return None, None

        # -------------------------- 7. 颜色空间转换（BGR→RGB） --------------------------
        # 关键细节：OpenCV默认读取彩色图像为BGR格式，而后续处理（如YCrCb转换）需RGB格式
        # cv2.COLOR_BGR2RGB：将BGR通道顺序转为RGB，确保色彩一致性
        visible_img = cv2.cvtColor(visible_img, cv2.COLOR_BGR2RGB)

        # -------------------------- 8. 返回有效图像对 --------------------------
        return ir_img, visible_img

    def reset(self):
        """ 重置帧读取指针到序列开头
            核心用途：重新读取当前视频的帧（无需重新调用load_video，提升效率）
            示例：读完10帧后调用reset()，下次read_frame()会从第1帧重新开始读取"""
        self.current_index = 0

    def get_frame_count(self):
        """获取当前加载视频的双模态共同帧总数
        核心用途：用于主程序控制循环次数（避免超界读取）、显示进度（如“第1/150帧”）

        输出：
            int: 共同帧总数（frame_list的长度），0表示未加载视频或无共同帧"""
        return len(self.frame_list)  # 直接返回有序帧列表的长度（无需额外计算）


def create_composite_image(ir_frame, visible_frame, aligned_ir, fused_frame,
                           frame_info=None, mkpts0=None, mkpts1=None, show_matches=False):
    """ 创建2x2布局的合成图像，集中展示「原始+处理后」的双模态数据
    参数：
        ir_frame: numpy.ndarray，原始红外图像（灰度，shape=(H,W)）
        visible_frame: numpy.ndarray，原始可见光图像（RGB，shape=(H,W,3)）
        aligned_ir: numpy.ndarray 或 None，配准后的红外图像（灰度），None表示配准失败
        fused_frame: numpy.ndarray 或 None，融合后的图像（BGR，OpenCV显示格式），None表示融合失败
        frame_info: dict 或 None，帧信息字典，需包含键：
                    - 'current'：当前帧号（int）
                    - 'total'：总帧数（int）
                    - 'matches'：有效匹配点数（int）
                    None表示不显示帧信息
        mkpts0: numpy.ndarray 或 None，可见光图像中的匹配点坐标（Nx2，每行[x,y]）
        mkpts1: numpy.ndarray 或 None，红外图像中的匹配点坐标（Nx2，与mkpts0一一对应）
        show_matches: bool，是否绘制匹配线（连接可见光与红外的对应点，辅助判断配准质量）

    输出：
        composite: numpy.ndarray，合成图像（BGR格式，shape=(2H,2W,3)），可直接用cv2.imshow显示"""
    # -------------------------- 1. 统一图像尺寸（避免布局错位） --------------------------
    h, w = 480, 640  # 目标尺寸（与process_frame_pair中resize尺寸一致，保证比例）

    # -------------------------- 2. 处理原始红外图像（灰度→BGR） --------------------------
    # 检查红外图像是否为灰度图：
    # - ir_frame.shape 获取图像的形状信息（维度）
    # - len(ir_frame.shape) 获取维度数量
    # - == 2 表示图像是二维的（只有高度和宽度），说明是灰度图（单通道）
    # - 如果是灰度图需要转为BGR格式（3通道），这样才能和彩色图像一起合成显示
    if len(ir_frame.shape) == 2:  # 若为灰度图（单通道），转BGR（3通道，便于合成）
        ir_display = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    else:  # 若已为3通道（理论上不会发生），直接复制
        ir_display = ir_frame.copy()
    ir_display = cv2.resize(ir_display, (w, h))  # 缩放到统一尺寸

    # -------------------------- 3. 处理原始可见光图像（RGB→BGR） --------------------------
    # 可见光原始为RGB格式，转BGR（与合成图像整体格式一致，避免色彩失真）
    visible_display = cv2.cvtColor(visible_frame, cv2.COLOR_RGB2BGR)
    visible_display = cv2.resize(visible_display, (w, h))  # 统一尺寸

    # -------------------------- 4. 处理配准后红外图像（兼容失败场景） --------------------------
    if aligned_ir is not None:  # 配准成功：按原始红外逻辑处理
        if len(aligned_ir.shape) == 2:
            aligned_display = cv2.cvtColor(aligned_ir, cv2.COLOR_GRAY2BGR)
        else:
            aligned_display = aligned_ir.copy()
        aligned_display = cv2.resize(aligned_display, (w, h))
    else:  # 配准失败：创建空白图+提示文字（提升可读性）
        aligned_display = np.zeros((h, w, 3), dtype=np.uint8)  # 黑色空白图（0=黑色）
        # 绘制提示文字：居中显示"No Alignment"，白色字体（255,255,255），线宽2
        cv2.putText(aligned_display, "No Alignment", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # -------------------------- 5. 处理融合结果图像（兼容失败场景） --------------------------
    if fused_frame is not None:  # 融合成功：直接缩放（已为BGR格式）
        fused_display = cv2.resize(fused_frame, (w, h))
    else:  # 融合失败：创建空白图+提示文字
        fused_display = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(fused_display, "No Fusion", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # -------------------------- 6. 创建合成图像画布（2H x 2W） --------------------------
    # 2行2列布局：总高度=2*h，总宽度=2*w，3通道BGR，像素值初始为0（黑色）
    composite = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    # -------------------------- 7. 放置图像到画布（2x2布局） --------------------------
    composite[0:h, 0:w] = ir_display  # 左上：原始红外
    composite[0:h, w:w * 2] = visible_display  # 右上：原始可见光
    composite[h:h * 2, 0:w] = aligned_display  # 左下：配准后红外
    composite[h:h * 2, w:w * 2] = fused_display  # 右下：融合结果

    # -------------------------- 8. 绘制象限标题（绿色文字，便于区分） --------------------------
    # cv2.putText函数中，org参数(10, 30)表示文本左下角的坐标位置：
    # - 第一个值10：x坐标（距离图像左边缘的像素数）
    # - 第二个值30：y坐标（距离图像上边缘的像素数）
    # 由于图像左上角是原点(0,0)，所以(10,30)表示在左上角区域内添加标题文字
    cv2.putText(composite, "Infrared (Raw)", (10, 30),  # 左上标题
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 绿色（0,255,0），线宽2
    cv2.putText(composite, "Visible (Raw)", (w + 10, 30),  # 右上标题
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(composite, "Infrared (Aligned)", (10, h + 30),  # 左下标题
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(composite, "Fused Result", (w + 10, h + 30),  # 右下标题
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # -------------------------- 9. 绘制帧信息（底部状态栏） --------------------------
    if frame_info:
        # 拼接信息文本：如“Frame: 1/150 | Matches: 339”
        info_text = f"Frame: {frame_info['current']}/{frame_info['total']} | Matches: {frame_info['matches']}"
        # 绘制在画布底部（距下边缘10像素），绿色小字体（0.7倍）
        cv2.putText(composite, info_text, (10, h * 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # -------------------------- 10. 绘制匹配线（可选，辅助配准质量判断） --------------------------
    if show_matches and mkpts0 is not None and mkpts1 is not None and len(mkpts0) > 0:
        for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
            # 随机生成匹配线颜色（区分不同点对，提升可视化效果）
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # 计算匹配点在合成图像中的坐标（基于2x2布局）：
            # - 可见光点（mkpts0）在右上象限：x需加w（右移w像素），y不变
            pt_vis = (int(x0) + w, int(y0))
            # - 红外点（mkpts1）在左下象限：y需加h（下移h像素），x不变
            pt_ir = (int(x1), int(y1) + h)
            # 绘制匹配线（线宽1，连接两点）
            cv2.line(composite, pt_vis, pt_ir, color, 1)

    return composite


if __name__ == "__main__":
    """完整流程：初始化数据加载→加载视频→初始化模型→逐帧处理→可视化展示→交互控制
    交互按键：
    - 'q'：退出程序
    - 'm'：切换匹配线显示/隐藏
    - 'p'：暂停播放（按任意键继续）
    - 's'：保存当前合成图像（命名格式：composite_frame_帧号.jpg）  """
    # -------------------------- 1. 初始化双模态加载器 --------------------------
    # 指定红外/可见光图像目录（需根据实际文件路径修改）
    loader = DualModalLoader(
        ir_dir=r"D:\HDO_Raw_Data\ir",        # 红外图像存储目录
        visible_dir=r"D:\HDO_Raw_Data\vi"    # 可见光图像存储目录
    )

    # -------------------------- 2. 加载指定视频ID的帧序列 --------------------------
    video_id = "1"  # 待处理的视频ID（需与文件名中的ID一致，如"1"对应"1 (1).jpg"）
    # 若加载失败（无共同帧/目录错误），打印提示并退出
    if not loader.load_video(video_id):
        print("无法加载视频，请检查：1.目录路径是否正确 2.文件名是否符合「视频ID (帧号).扩展名」格式")
        exit()  # 终止程序

    # -------------------------- 3. 获取总帧数并打印进度 --------------------------
    total_frames = loader.get_frame_count()
    print(f"\n开始处理视频ID={video_id}，共 {total_frames} 帧...")

    # -------------------------- 4. 初始化深度学习模型（配准+融合） --------------------------
    # init_models()：加载预训练的Matchformer（配准）和Generator（融合）模型
    fusion_net, matcher = init_models()

    # -------------------------- 5. 创建可视化窗口 --------------------------
    # 窗口名称：Multi-Modal Fusion Results
    cv2.namedWindow("Multi-Modal Fusion Results", cv2.WINDOW_NORMAL)
    # 设置窗口初始大小（1280x960，与合成图像2x2布局匹配：480*2=960，640*2=1280）
    cv2.resizeWindow("Multi-Modal Fusion Results", 1280, 960)

    # -------------------------- 6. 初始化交互控制变量 --------------------------
    show_matches = False  # 初始不显示匹配线（避免画面杂乱）

    # -------------------------- 7. 主循环：逐帧处理 --------------------------
    for i in range(total_frames):
        # 7.1 读取当前帧的双模态图像
        ir_frame, visible_frame = loader.read_frame()
        # 若读取失败（文件损坏），打印提示并跳过当前帧
        if ir_frame is None or visible_frame is None:
            print(f"第 {i + 1}/{total_frames} 帧读取失败，跳过...")
            continue

        # 7.2 核心处理：配准+融合（调用process_frame_pair）
        fused_frame, aligned_ir, h, match_count, mkpts0, mkpts1 = process_frame_pair(
            visible_frame, ir_frame, fusion_net, matcher
        )

        # 7.3 打印当前帧处理状态（配准成功/失败+匹配点数）
        if h is not None:
            status = f"帧 {i + 1}/{total_frames}: 配准成功 | 有效匹配点数: {match_count}"
        else:
            status = f"帧 {i + 1}/{total_frames}: 配准失败（使用加权融合） | 有效匹配点数: {match_count}"
        print(status)

        # 7.4 准备帧信息（用于合成图像显示）
        frame_info = {
            'current': i + 1,    # 当前帧号（从1开始，符合用户习惯）
            'total': total_frames,  # 总帧数
            'matches': match_count  # 有效匹配点数
        }

        # 7.5 创建2x2合成图像
        composite = create_composite_image(
            ir_frame, visible_frame, aligned_ir, fused_frame,
            frame_info, mkpts0, mkpts1, show_matches
        )

        # 7.6 显示合成图像
        cv2.imshow("Multi-Modal Fusion Results", composite)

        # 7.7 处理键盘交互（等待100ms，控制播放速度）
        # waitKey(100)：每帧停留100ms（约10fps），同时捕获按键
        key = cv2.waitKey(100) & 0xFF  # 0xFF确保跨平台按键值一致
        if key == ord('q'):  # 按'q'键：退出循环，终止程序
            print("用户按下'q'键，退出处理...")
            break
        elif key == ord('m'):  # 按'm'键：切换匹配线显示/隐藏
            show_matches = not show_matches
            print(f"匹配线显示状态：{'已开启' if show_matches else '已关闭'}")
        elif key == ord('p'):  # 按'p'键：暂停（waitKey(0)表示等待任意按键）
            print("已暂停，按任意键继续播放...")
            cv2.waitKey(0)
        elif key == ord('s'):  # 按's'键：保存当前合成图像
            # 保存路径：当前工作目录，文件名含帧号（避免覆盖）
            save_path = f"composite_frame_{i + 1}.jpg"
            cv2.imwrite(save_path, composite)  # cv2.imwrite默认保存BGR格式（正确）
            print(f"第 {i + 1} 帧合成图像已保存至：{os.path.abspath(save_path)}")

    # -------------------------- 8. 程序结束：清理资源 --------------------------
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口（释放内存）
    print("\n程序正常结束！")