from urllib.parse import urlparse, quote, urlunparse
import cv2
from onvif import ONVIFCamera
import numpy as np


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


def resize_with_aspect_ratio(image, target_width, target_height, keep_original_size=False):
    """
    等比例缩放图像到指定尺寸

    Args:
        image: 输入图像
        target_width: 目标宽度
        target_height: 目标高度
        keep_original_size: 是否保持原始尺寸输出（添加黑边）

    Returns:
        resized_image: 缩放后的图像
        scale: 缩放比例
        actual_size: 实际缩放后的尺寸 (width, height)
    """
    original_height, original_width = image.shape[:2]

    # 计算缩放比例
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)  # 选择较小的比例保持宽高比

    # 计算缩放后的实际尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 缩放图像
    if len(image.shape) == 3:
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)  # 转换为3通道

    if keep_original_size:
        # 创建目标尺寸的黑色画布
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # 计算居中位置
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        # 将缩放后的图像放在画布中央
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        return canvas, scale, (new_width, new_height)
    else:
        return resized, scale, (new_width, new_height)


def create_comparison_layout(visible_frame, infrared_frame, layout_mode="horizontal"):
    """
    创建比较布局

    Args:
        visible_frame: 可见光图像
        infrared_frame: 红外图像
        layout_mode: 布局模式 ("horizontal", "vertical", "grid")

    Returns:
        combined_image: 组合图像
        layout_info: 布局信息字典
    """

    # 获取原始尺寸
    vis_h, vis_w = visible_frame.shape[:2]
    ir_h, ir_w = infrared_frame.shape[:2]

    print(f"📐 原始尺寸 - 可见光: {vis_w}x{vis_h}, 红外: {ir_w}x{ir_h}")

    if layout_mode == "horizontal":
        # 水平排列：统一高度，宽度按比例
        target_height = 480  # 统一显示高度

        # 等比例缩放两个图像到相同高度
        vis_resized, vis_scale, vis_actual = resize_with_aspect_ratio(visible_frame, 9999, target_height)
        ir_resized, ir_scale, ir_actual = resize_with_aspect_ratio(infrared_frame, 9999, target_height)

        # 水平拼接
        combined = np.hstack([vis_resized, ir_resized])

        layout_info = {
            "mode": "horizontal",
            "target_height": target_height,
            "visible_scale": vis_scale,
            "infrared_scale": ir_scale,
            "visible_actual_size": vis_actual,
            "infrared_actual_size": ir_actual,
            "combined_size": (vis_actual[0] + ir_actual[0], target_height)
        }

    elif layout_mode == "vertical":
        # 垂直排列：统一宽度，高度按比例
        target_width = 640  # 统一显示宽度

        # 等比例缩放两个图像到相同宽度
        vis_resized, vis_scale, vis_actual = resize_with_aspect_ratio(visible_frame, target_width, 9999)
        ir_resized, ir_scale, ir_actual = resize_with_aspect_ratio(infrared_frame, target_width, 9999)

        # 垂直拼接
        combined = np.vstack([vis_resized, ir_resized])

        layout_info = {
            "mode": "vertical",
            "target_width": target_width,
            "visible_scale": vis_scale,
            "infrared_scale": ir_scale,
            "visible_actual_size": vis_actual,
            "infrared_actual_size": ir_actual,
            "combined_size": (target_width, vis_actual[1] + ir_actual[1])
        }

    elif layout_mode == "grid":
        # 网格排列：统一尺寸，保持宽高比，添加黑边
        target_width = 480
        target_height = 360

        # 等比例缩放到统一尺寸（带黑边）
        vis_resized, vis_scale, vis_actual = resize_with_aspect_ratio(
            visible_frame, target_width, target_height, keep_original_size=True)
        ir_resized, ir_scale, ir_actual = resize_with_aspect_ratio(
            infrared_frame, target_width, target_height, keep_original_size=True)

        # 2x1网格排列
        combined = np.hstack([vis_resized, ir_resized])

        layout_info = {
            "mode": "grid",
            "target_size": (target_width, target_height),
            "visible_scale": vis_scale,
            "infrared_scale": ir_scale,
            "visible_actual_size": vis_actual,
            "infrared_actual_size": ir_actual,
            "combined_size": (target_width * 2, target_height)
        }

    return combined, layout_info


def add_info_overlay(image, layout_info, frame_count):
    """在图像上添加信息覆盖层"""
    overlay = image.copy()

    # 添加半透明背景
    cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # 添加文本信息
    texts = [
        f"Frame: {frame_count}",
        f"Layout: {layout_info['mode']}",
        f"Visible Scale: {layout_info['visible_scale']:.3f}",
        f"Infrared Scale: {layout_info['infrared_scale']:.3f}",
        f"Visible Size: {layout_info['visible_actual_size'][0]}x{layout_info['visible_actual_size'][1]}",
        f"Infrared Size: {layout_info['infrared_actual_size'][0]}x{layout_info['infrared_actual_size'][1]}"
    ]

    for i, text in enumerate(texts):
        cv2.putText(image, text, (15, 35 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return image

def main():
    print("=== 等比例缩放双摄像头查看器 ===")

    # 连接摄像头1 - 可见光
    print("正在连接摄像头1 (可见光)...")
    try:
        camera1 = ONVIFCamera(
            '192.168.1.68',  # 修改为你的实际IP
            80,
            'admin',
            'system123',
            no_cache=True
        )
        print("✓ 摄像头1连接成功")
    except Exception as e:
        print(f"✗ 摄像头1连接失败: {e}")
        return

    # 连接摄像头2 - 红外
    print("正在连接摄像头2 (红外)...")
    try:
        camera2 = ONVIFCamera(
            '192.168.1.2',  # 修改为你的实际IP
            8000,
            'admin',
            'system123',
            no_cache=True
        )
        print("✓ 摄像头2连接成功")
    except Exception as e:
        print(f"✗ 摄像头2连接失败: {e}")
        return

    # 获取视频流地址
    print("\n正在获取视频流地址...")
    visible_stream_uri = get_stream_uri(camera1, "admin", "system123", 0)
    infrared_stream_uri = get_stream_uri(camera2, "admin", "system123", 0)

    if not visible_stream_uri or not infrared_stream_uri:
        print("✗ 无法获取视频流地址")
        return

    # 初始化视频捕获
    print("\n正在初始化视频捕获...")
    cap_visible = cv2.VideoCapture(visible_stream_uri)
    cap_infrared = cv2.VideoCapture(infrared_stream_uri)

    # 设置缓冲区大小
    cap_visible.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_infrared.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap_visible.isOpened() or not cap_infrared.isOpened():
        print("✗ 视频流打开失败")
        return

    print("✓ 视频流打开成功")

    # 获取并显示原始视频信息
    print("\n📹 原始视频流信息:")
    vis_width = int(cap_visible.get(cv2.CAP_PROP_FRAME_WIDTH))
    vis_height = int(cap_visible.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vis_fps = cap_visible.get(cv2.CAP_PROP_FPS)

    ir_width = int(cap_infrared.get(cv2.CAP_PROP_FRAME_WIDTH))
    ir_height = int(cap_infrared.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ir_fps = cap_infrared.get(cv2.CAP_PROP_FPS)

    print(f"🔍 可见光: {vis_width}x{vis_height} @ {vis_fps:.1f}fps")
    print(f"🔍 红外: {ir_width}x{ir_height} @ {ir_fps:.1f}fps")

    # 计算原始宽高比
    vis_ratio = vis_width / vis_height
    ir_ratio = ir_width / ir_height
    print(f"📐 宽高比 - 可见光: {vis_ratio:.3f}, 红外: {ir_ratio:.3f}")

    # 创建窗口
    cv2.namedWindow('Scaled Dual Camera View', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Individual Views', cv2.WINDOW_NORMAL)

    # 当前布局模式
    layout_modes = ["horizontal", "vertical", "grid"]
    current_mode_index = 0
    current_mode = layout_modes[current_mode_index]

    print(f"\n🎬 开始显示等比例缩放视频流...")
    print("🎮 控制说明:")
    print("  'q' - 退出程序")
    print("  's' - 保存当前帧")
    print("  'm' - 切换布局模式 (horizontal/vertical/grid)")
    print("  'i' - 显示详细信息")
    print("  'r' - 重置窗口")
    print("  空格 - 暂停/继续")

    frame_count = 0
    paused = False

    try:
        while True:
            if not paused:
                # 读取帧
                ret1, visible_frame = cap_visible.read()
                ret2, infrared_frame = cap_infrared.read()

                if ret1 and ret2:
                    frame_count += 1

                    # 检查实际读取的帧尺寸
                    if frame_count == 1:
                        actual_vis_h, actual_vis_w = visible_frame.shape[:2]
                        actual_ir_h, actual_ir_w = infrared_frame.shape[:2]
                        print(
                            f"✓ 实际帧尺寸 - 可见光: {actual_vis_w}x{actual_vis_h}, 红外: {actual_ir_w}x{actual_ir_h}")

                    # 创建比较布局
                    combined_image, layout_info = create_comparison_layout(
                        visible_frame, infrared_frame, current_mode)

                    # 添加标签和分割线
                    if current_mode == "horizontal":
                        # 添加垂直分割线
                        split_x = layout_info["visible_actual_size"][0]
                        cv2.line(combined_image, (split_x, 0), (split_x, combined_image.shape[0]), (100, 100, 100), 2)
                        # 添加标签
                        cv2.putText(combined_image, 'Visible Light', (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(combined_image, 'Infrared', (split_x + 20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    elif current_mode == "vertical":
                        # 添加水平分割线
                        split_y = layout_info["visible_actual_size"][1]
                        cv2.line(combined_image, (0, split_y), (combined_image.shape[1], split_y), (100, 100, 100), 2)
                        # 添加标签
                        cv2.putText(combined_image, 'Visible Light', (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(combined_image, 'Infrared', (20, split_y + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    elif current_mode == "grid":
                        # 添加垂直分割线
                        split_x = layout_info["target_size"][0]
                        cv2.line(combined_image, (split_x, 0), (split_x, combined_image.shape[0]), (100, 100, 100), 2)
                        # 添加标签
                        cv2.putText(combined_image, 'Visible Light', (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(combined_image, 'Infrared', (split_x + 20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # 添加信息覆盖层
                    combined_with_info = add_info_overlay(combined_image, layout_info, frame_count)

                    # 显示组合图像
                    cv2.imshow('Scaled Dual Camera View', combined_with_info)

                    # 创建独立视图（小窗口）
                    vis_small, _, _ = resize_with_aspect_ratio(visible_frame, 320, 240, True)
                    ir_small, _, _ = resize_with_aspect_ratio(infrared_frame, 320, 240, True)
                    individual_view = np.hstack([vis_small, ir_small])

                    cv2.putText(individual_view, f'Original Sizes', (10, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow('Individual Views', individual_view)

                else:
                    if not ret1:
                        print("✗ 读取可见光帧失败")
                    if not ret2:
                        print("✗ 读取红外帧失败")

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('m'):
                # 切换布局模式
                current_mode_index = (current_mode_index + 1) % len(layout_modes)
                current_mode = layout_modes[current_mode_index]
                print(f"🔄 切换到 {current_mode} 布局模式")
            elif key == ord('s'):
                # 保存当前帧
                if ret1 and ret2:
                    timestamp = int(cv2.getTickCount())
                    combined_filename = f'combined_{current_mode}_{timestamp}.jpg'
                    cv2.imwrite(combined_filename, combined_with_info)
                    print(f"✓ 已保存: {combined_filename}")
            elif key == ord('i'):
                # 显示详细信息
                print(f"\n=== 详细信息 ===")
                print(f"帧数: {frame_count}")
                print(f"布局模式: {current_mode}")
                print(f"可见光缩放比例: {layout_info['visible_scale']:.3f}")
                print(f"红外缩放比例: {layout_info['infrared_scale']:.3f}")
                print(f"组合图像尺寸: {layout_info['combined_size']}")
            elif key == ord('r'):
                # 重置窗口
                cv2.destroyAllWindows()
                cv2.namedWindow('Scaled Dual Camera View', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Individual Views', cv2.WINDOW_NORMAL)
                print("✓ 窗口已重置")
            elif key == ord(' '):
                # 暂停/继续
                paused = not paused
                print(f"⏸️ {'暂停' if paused else '继续'}")

    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    finally:
        # 清理资源
        print("正在清理资源...")
        cap_visible.release()
        cap_infrared.release()
        cv2.destroyAllWindows()
        print("程序结束")


if __name__ == '__main__':
    main()