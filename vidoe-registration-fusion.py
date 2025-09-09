import torch
import torch.nn as nn
from einops.einops import rearrange
import cv2
import numpy as np
from urllib.parse import urlparse, urlunparse, quote
from model.matchformer import Matchformer
from config.defaultmf import default_cfg
from ATGAN.modules.generator import Generator
import time
import threading
from queue import Queue
from onvif import ONVIFCamera


def get_stream_uri(camera, username, password, index):
    """获取带认证的视频流地址"""
    try:
        media_service = camera.create_media_service()
        profiles = media_service.GetProfiles()
        profile = profiles[index]

        stream_uri = media_service.GetStreamUri({
            'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
            'ProfileToken': profile.token
        })

        original_uri = stream_uri.Uri
        parsed = urlparse(original_uri)
        credentials = f"{quote(username)}:{quote(password)}"
        authed_uri = urlunparse((
            parsed.scheme,
            f"{credentials}@{parsed.hostname}:{parsed.port}",
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
        return authed_uri
    except Exception as e:
        print(f"获取流地址失败: {e}")
        return None


def RGB2YCrCb(rgb_image):
    """将RGB格式转换为YCrCb格式"""
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0).detach()
    Cb = Cb.clamp(0.0, 1.0).detach()
    return Y, Cb, Cr


def YCbCr2RGB(Y, Cb, Cr):
    """将YcrCb格式转换为RGB格式"""
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)

    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0, 1.0)
    return out


def init_models():
    """初始化融合网络和匹配网络"""
    print("正在加载模型...")

    # 加载融合网络
    fusion_net = Generator()
    fusion_net.load_state_dict(torch.load(r'./ATGAN/checkpoint/modelc.ckpt'), strict=True)
    fusion_net = fusion_net.eval().cuda()

    # 加载匹配网络
    matcher = Matchformer(config=default_cfg)
    matcher.load_state_dict(torch.load(r'modelcc15epoch.ckpt'), strict=False)
    matcher = matcher.eval().cuda()

    return fusion_net, matcher


def process_frame_pair(vis_frame, ir_frame, fusion_net, matcher, last_homography=None):
    """处理单帧图像对"""
    try:
        # 调整尺寸
        vis_frame = cv2.resize(vis_frame, (640, 480))
        ir_frame = cv2.resize(ir_frame, (640, 480))

        # 确保红外图像是灰度图
        if len(ir_frame.shape) == 3:
            ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)

        # 转换颜色空间
        vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

        # 转换为tensor
        img0 = torch.from_numpy(vis_frame_rgb)[None].cuda() / 255.
        img1 = torch.from_numpy(ir_frame)[None][None].cuda() / 255.

        img0 = rearrange(img0, 'n h w c -> n c h w')
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img0)

        # 特征匹配
        batch = {'image0': vi_Y, 'image1': img1}

        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_c'].cpu().numpy()
            mkpts1 = batch['mkpts1_c'].cpu().numpy()

            h = None
            if len(mkpts0) >= 4:  # 需要至少4个点计算单应性
                try:
                    h, prediction = cv2.findHomography(
                        mkpts1, mkpts0,
                        cv2.USAC_MAGSAC, 5,
                        confidence=0.99,
                        maxIters=10000  # 降低迭代次数提高速度
                    )
                    if h is not None:
                        # 过滤内点
                        prediction = np.array(prediction, dtype=bool).reshape([-1])
                        mkpts0 = mkpts0[prediction]
                        mkpts1 = mkpts1[prediction]
                except:
                    h = None

            # 如果当前帧匹配失败，使用上一帧的变换矩阵
            if h is None and last_homography is not None:
                h = last_homography
                print("使用上一帧的变换矩阵")

            if h is not None:
                # 配准红外图像
                img11 = cv2.warpPerspective(ir_frame, h, (640, 480))
                img11_tensor = torch.from_numpy(img11)[None][None].cuda() / 255.

                # 融合
                fuse = fusion_net(img11_tensor, vi_Y)
                fuse = YCbCr2RGB(fuse, vi_Cb, vi_Cr)
                fuse = fuse.detach().cpu()[0]
                fuse = rearrange(fuse, 'c h w -> h w c').numpy()

                # 转换为显示格式
                fused_frame = (fuse * 255).astype(np.uint8)
                fused_frame = cv2.cvtColor(fused_frame, cv2.COLOR_RGB2BGR)

                return fused_frame, img11, h, len(mkpts0)
            else:
                # 匹配失败，返回简单融合
                simple_fused = cv2.addWeighted(vis_frame, 0.7,
                                               cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR), 0.3, 0)
                return simple_fused, ir_frame, None, 0

    except Exception as e:
        print(f"处理帧时出错: {e}")
        return vis_frame, ir_frame, last_homography, 0


class VideoStreamReader:
    """视频流读取器"""

    def __init__(self, stream_uri, name):
        self.stream_uri = stream_uri
        self.name = name
        self.cap = None
        self.frame_queue = Queue(maxsize=5)
        self.running = False

    def start(self):
        """启动视频流读取"""
        self.cap = cv2.VideoCapture(self.stream_uri)
        if not self.cap.isOpened():
            print(f"无法打开{self.name}视频流: {self.stream_uri}")
            return False

        # 设置缓冲区大小
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True

        # 启动读取线程
        self.thread = threading.Thread(target=self._read_frames)
        self.thread.daemon = True
        self.thread.start()

        print(f"{self.name}视频流启动成功")
        return True

    def _read_frames(self):
        """读取帧的线程函数"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # 如果队列满了，丢弃旧帧
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)

    def get_frame(self):
        """获取最新帧"""
        try:
            return self.frame_queue.get(timeout=0.1)
        except:
            return None

    def stop(self):
        """停止视频流"""
        self.running = False
        if self.cap:
            self.cap.release()


def process_multimodal_video_stream(camera1, camera2):
    """处理多模态视频流的主函数"""

    # 获取视频流地址
    visible_stream_uri = get_stream_uri(camera1, "admin", "system123", 0)
    infrared_stream_uri = get_stream_uri(camera2, "admin", "system123", 1)

    if not visible_stream_uri or not infrared_stream_uri:
        print("获取视频流地址失败!")
        return

    print(f"可见光流地址: {visible_stream_uri}")
    print(f"红外流地址: {infrared_stream_uri}")

    # 初始化模型
    fusion_net, matcher = init_models()

    # 创建视频流读取器
    vis_reader = VideoStreamReader(visible_stream_uri, "可见光")
    ir_reader = VideoStreamReader(infrared_stream_uri, "红外")

    # 启动视频流
    if not vis_reader.start() or not ir_reader.start():
        print("视频流启动失败!")
        return

    print("开始处理视频流...")
    print("按 'q' 退出, 按 's' 保存当前帧")

    last_homography = None
    frame_count = 0
    fps_start_time = time.time()

    try:
        while True:
            # 获取帧
            vis_frame = vis_reader.get_frame()
            ir_frame = ir_reader.get_frame()

            if vis_frame is None or ir_frame is None:
                continue

            frame_count += 1

            # 处理帧对
            start_time = time.time()
            fused_frame, aligned_ir, last_homography, match_count = process_frame_pair(
                vis_frame, ir_frame, fusion_net, matcher, last_homography
            )
            process_time = time.time() - start_time

            # 调整显示尺寸
            display_vis = cv2.resize(vis_frame, (320, 240))
            display_ir = cv2.resize(ir_frame, (320, 240))
            display_aligned_ir = cv2.resize(aligned_ir, (320, 240))
            display_fused = cv2.resize(fused_frame, (320, 240))

            # 转换红外图像为3通道用于显示
            if len(display_ir.shape) == 2:
                display_ir = cv2.cvtColor(display_ir, cv2.COLOR_GRAY2BGR)
            if len(display_aligned_ir.shape) == 2:
                display_aligned_ir = cv2.cvtColor(display_aligned_ir, cv2.COLOR_GRAY2BGR)

            # 创建组合显示
            top_row = np.hstack([display_vis, display_ir])
            bottom_row = np.hstack([display_aligned_ir, display_fused])
            combined = np.vstack([top_row, bottom_row])

            # 添加文字标注
            cv2.putText(combined, 'Visible', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, 'Infrared', (330, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, 'Aligned IR', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, 'Fused', (330, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示处理信息
            info_text = f"Matches: {match_count}, Time: {process_time:.3f}s"
            cv2.putText(combined, info_text, (10, combined.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 计算FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                print(f"FPS: {fps:.2f}, 匹配点数: {match_count}, 处理时间: {process_time:.3f}s")
                fps_start_time = time.time()

            cv2.imshow('Multimodal Video Fusion', combined)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"fused_frame_{timestamp}.png", fused_frame)
                print(f"保存帧: fused_frame_{timestamp}.png")

            # 定期清理GPU内存
            if frame_count % 100 == 0:
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"处理过程中出错: {e}")
    finally:
        # 清理资源
        vis_reader.stop()
        ir_reader.stop()
        cv2.destroyAllWindows()
        print("视频流处理结束")


# 主程序入口
if __name__ == '__main__':
    # camera1和camera2对象
    # camera1   # 可见光相机
    camera1 = ONVIFCamera(
        '192.168.1.68',  # 修改为你的实际IP
        80,
        'admin',
        'system123',
        no_cache=True
    )
    # camera2   # 红外相机
    camera2 = ONVIFCamera(
        '192.168.1.2',
        8000,
        'admin',
        'system123',
        no_cache=True
    )
    process_multimodal_video_stream(camera1, camera2)

    # print("请在调用process_multimodal_video_stream函数前先初始化camera1和camera2对象")