from onvif import ONVIFCamera
from urllib.parse import urlparse, urlunparse, quote
import cv2

camera1 = ONVIFCamera(
        '192.168.1.68',  # 可见光  frame.shape是(1080,1920,3)
        80,
        'admin',
        'system123',
        no_cache=True
    )

camera2 = ONVIFCamera(
    '192.168.1.2',
    8000,             #红外 红外的index只有0可以用               frame.shape是()
    'admin',
    'system123',
    no_cache=True
)

def get_stream_uri(camera,index):
    """作用就是获取带认证的视频流地址"""
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
        credentials = f"{quote(camera.user)}:{quote(camera.passwd)}"
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


uri = get_stream_uri(camera1,0)
cap = cv2.VideoCapture(uri)
print("RTSP URI:", uri)

while True:
    ret, frame = cap.read()
    if ret:
        pass
    else:
        print("没有读取")
        break
    print(frame.shape)
#   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #cv2.VideoCapture 是基于 FFmpeg 的，它解码时往往会自动把视频流转成 BGR 格式
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()



