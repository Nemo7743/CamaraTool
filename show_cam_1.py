import cv2
import time
import struct

def decode_fourcc(v):
    """
    將 OpenCV 的 FOURCC 數值解碼為字串 (例如 1196444237 -> 'MJPG')
    """
    try:
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])
    except:
        return "UNKNOWN"

def main(desired_format, cam_index):
    # 開啟攝影機 (通常 0 是預設鏡頭)
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    # -----------------------------------------------------------
    # 1. 設定解析度 640x480
    # -----------------------------------------------------------
    target_width = 640
    target_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    # -----------------------------------------------------------
    # 2. 設定影像格式 (FOURCC)
    # 請在此處修改字串以測試不同格式： 'MJPG', 'YUYV', 'YUY2', 'UYVY'
    # 注意：如果硬體不支援該格式，OpenCV 通常會自動退回到預設格式
    # -----------------------------------------------------------
    #desired_format = 'MJPG'  #在此修改： 'MJPG', 'YUYV', 'UYVY'
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*desired_format))

    # 用於計算 FPS
    prev_time = 0
    curr_time = 0

    print(f"嘗試設定格式為: {desired_format}")
    print(f"嘗試設定解析度為: {target_width}x{target_height}")
    print("按下 'q' 鍵退出程式")

    while True:
        # 讀取影格
        ret, frame = cap.read()
        
        if not ret:
            print("無法接收影格 (stream end?). Exiting ...")
            break

        # -------------------------------------------------------
        # 計算 FPS
        # -------------------------------------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # -------------------------------------------------------
        # 獲取當前實際的格式與解析度資訊
        # -------------------------------------------------------
        # 讀取鏡頭實際輸出的格式 (有時候設定失敗會跳回預設值)
        actual_fourcc_val = cap.get(cv2.CAP_PROP_FOURCC)
        actual_format = decode_fourcc(actual_fourcc_val)
        
        # 讀取實際解析度
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # -------------------------------------------------------
        # 在畫面上顯示資訊 (OSD)
        # -------------------------------------------------------
        info_text_fps = f"FPS: {fps:.2f}"
        info_text_fmt = f"Format: {actual_format} ({w}x{h})"

        # 繪製綠色文字
        cv2.putText(frame, info_text_fps, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, info_text_fmt, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 顯示影像
        cv2.imshow('Camera Preview', frame)

        # 按下 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_index = 0

    #desired_format = 'MJPG'
    #desired_format = 'YUYV'
    #desired_format = 'UYVY'
    desired_format = 'YUY2'
    main(desired_format, cam_index)