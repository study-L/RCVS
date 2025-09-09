import os
import cv2  ##加载OpenCV模块
'''
    功能：视频转图片序列。参数含义如下所示：
    pathIn：视频的路径，比如：F:\python_tutorials\test.mp4(自己需要设置！！)
    pathOut：设定提取的图片保存在哪个文件夹下，比如：F:\python_tutorials\frames1  如果该文件夹不存在，函数将自动创建它
    only_output_video_info：如果为True，只输出视频信息（长度、帧数和帧率），不提取图片
    extract_time_points：提取的时间点，单位为秒，为元组数据，比如，(2, 3, 5)表示只提取视频第2秒， 第3秒，第5秒图片
    initial_extract_time：提取的起始时刻，单位为秒，默认为0（即从视频最开始提取）
    end_extract_time：提取的终止时刻，单位为秒，默认为None（即视频终点）
    extract_time_interval：提取的时间间隔，单位为秒，默认为-1（即输出时间范围内的所有帧）
    output_prefix：图片的前缀名，默认为，图片的名称将为frame_000001.jpg、frame_000002.jpg、frame_000003.jpg......
    jpg_quality：设置图片质量，范围为0到100，默认为100（质量最佳）
    isColor：如果为False，输出的将是黑白图片
'''

#单纯播放视频
def pure_show_video(pathIn=''):
    cap = cv2.VideoCapture(pathIn)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  ##视频的帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  ##视频的帧率
    dur = n_frames / fps  ##视频的时间
    while True:
        _, frame = cap.read()
        cv2.imshow(f'{pathIn}', frame)
        cv2.waitKey(round((1/fps)*1000))

#对视频处理
def video2frames(pathIn='',
                 pathOut='',
                 only_output_video_info=False,
                 extract_time_points=None,
                 initial_extract_time=0,
                 end_extract_time=None,
                 extract_time_interval=-1,
                 output_prefix='',  #图片文件前缀
                 jpg_quality=100,
                 isColor=True):


    cap = cv2.VideoCapture(pathIn)  ##打开视频文件
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  ##视频的帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  ##视频的帧率
    dur = n_frames / fps  ##视频的时间

    ##如果only_output_video_info=True, 只输出视频信息，不提取图片
    if only_output_video_info:
        print('只输出视频信息，不提取图片')
        print(f"视频一共：{dur:.4f} 秒")
        print(f"视频一共帧数：{n_frames}")
        print(f"FPS:{fps:.4f}")

    ##提取特定时间点图片
    elif extract_time_points is not None:
        print(f'提取特定时间点{extract_time_points}的图片')
        if max(extract_time_points) > dur:  ##判断时间点是否符合要求
            raise NameError('最大时间点超出视频时长')

        try:
            os.mkdir(pathOut)
        except OSError:
            pass
        success = True
        count = 0
        while success and count < len(extract_time_points):
            cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * extract_time_points[count]))
            success, image = cap.read()
            if success:
                if not isColor:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  ##转化为黑白图片
                #print('Write a new frame: {}, {}th'.format(success, count + 1))
                cv2.imwrite(os.path.join(pathOut, "{}{:0>10d}.jpg".format(output_prefix, count + 1)), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                count = count + 1

    else:
        ##判断起始时间、终止时间参数是否符合要求
        if initial_extract_time > dur:
            raise NameError('initial extract time is larger than the video duration....')
        if end_extract_time is not None:
            if end_extract_time > dur:
                raise NameError('end extract time is larger than the video duration....')
            if initial_extract_time > end_extract_time:
                raise NameError('end extract time is less than the initial extract time....')

        ##时间范围内的每帧图片都输出
        if extract_time_interval == -1:
            if initial_extract_time > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time))
            try:
                os.mkdir(pathOut)
            except OSError:
                pass
            print(f'时间范围内的每帧图片都输出')
            print('Converting a video into frames......')
            #到视频结尾
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) * fps + 1
                success = True
                count = 0
                while success and count < N:
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        #print('Write a new frame: {}, {}/{}'.format(success, count + 1, n_frames))
                        cv2.imwrite(os.path.join(pathOut, "{:0>10d}.jpg".format(count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1
            #截取视频的结尾有特定设置
            else:
                success = True
                count = 0
                while success:
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        #print('Write a new frame: {}, {}/{}'.format(success, count + 1, n_frames))
                        cv2.imwrite(os.path.join(pathOut, "{:0>10d}.jpg".format(count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1

        ##判断提取时间间隔设置是否符合要求
        elif extract_time_interval > 0 and extract_time_interval < 1 / fps:
            raise NameError('extract_time_interval is less than the frame time interval....')
        elif extract_time_interval > (n_frames / fps):
            raise NameError('extract_time_interval is larger than the duration of the video....')

        ##时间范围内每隔一段时间输出一张图片
        else:
            try:
                os.mkdir(pathOut)
            except OSError:
                pass
            print('有时间间隔的输出')
            print(f'Converting a video into frames....')
            #截取视频的结尾有特定设置
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) / extract_time_interval + 1
                success = True
                count = 0
                while success and count < N:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        #print('Write a new frame2: {}, {}th'.format(success, count + 1))
                        cv2.imwrite(os.path.join(pathOut, "{}{:0>10d}.jpg".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1
                        print(f'\r ({count}/{int(N)})',end='')
            else:
                # 截取视频到结束
                success = True
                count = 0
                while success:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        #print('Write a new frame: {}, {}th'.format(success, count + 1))
                        cv2.imwrite(os.path.join(pathOut, "{}{:0>10d}.jpg".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1

    if not only_output_video_info:
        print(f'一共产生{count}张图片序列')





if __name__ == '__main__':

    #视频输入路径和输出路径
    # pathIn = r'H:\fusionvi\2022_07_02_09_42_IMG_2262.MOV'
    # pathOut = r'D:\lunwen\matchtu\vi1'
    pathIn = r'D:\edge_download\RCVS1.gif'
    pathOut = r'D:\fusiondata_seg'
    #使用哪一个功能取消注释。
    #功能1：单纯显示视频
    # pure_show_video(pathIn)

    #功能2：输出读取到的视频信息（长度、帧数、帧率、不提取图片）
    # video2frames(pathIn, only_output_video_info=True)

    #功能3：视频转为图片序列(提取特定时间点图片，单位为s)
    # video2frames(pathIn, pathOut,extract_time_points=(1,2,3,4,5,6))

    #功能4：视频转为图片序列(提取时间范围内的每帧图片，单位为s)
    # video2frames(pathIn, pathOut,initial_extract_time=0,end_extract_time=15)

    # 功能5：视频转为图片序列(提取时间范围内的图片，中间有间隔，单位为s)
    video2frames(pathIn, pathOut,initial_extract_time=0,extract_time_interval=0.05)
    # video2frames(pathIn, pathOut,initial_extract_time=40,end_extract_time=45,extract_time_interval=1)
