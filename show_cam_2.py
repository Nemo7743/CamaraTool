import cv2
import time

def decode_fourcc(v):
    """解碼 FOURCC 數值為字串"""
    try:
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])
    except:
        return "UNKNOWN"

def main(desired_format, cam_index):
    # 強制使用 DirectShow (cv2.CAP_DSHOW)，這在 Windows 上對高幀率支援較好且延遲較低
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    # -----------------------------------------------------------
    # 設定參數
    # -----------------------------------------------------------
    target_width = 640
    target_height = 480
    target_fps = 120.0  # 設定請求 120 FPS
    
    # 在這裡修改你要測試的格式： 'MJPG', 'YUYV', 'UYVY'
    # 注意：要跑滿 120fps，通常必須使用 'MJPG'
    # desired_format = 'MJPG' #'MJPG', 'YUYV', 'UYVY', 'YUY2'

    # 套用設定
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*desired_format))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    # 取得實際生效的參數 (僅讀取一次以節省迴圈內的資源)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_format = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    
    print(f"系統回報格式: {real_format} | 解析度: {real_w}x{real_h}")
    print("資訊將顯示於視窗標題欄以節省 CPU 資源。")
    print("按下 'q' 鍵退出。")

    # 變數初始化
    prev_time = time.time()
    frame_count = 0
    fps_update_interval = 0.5  # 每 0.5 秒更新一次標題文字，避免過度刷新 UI
    last_fps_update = time.time()
    
    window_name = "High FPS Preview (Battery Saver Mode)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        # 讀取影像
        ret, frame = cap.read()
        
        if not ret:
            break

        # -------------------------------------------------------
        # 省電核心：不對 frame 做任何繪圖操作 (putText 是很耗能的)
        # 直接顯示原始影像矩陣
        # -------------------------------------------------------
        cv2.imshow(window_name, frame)

        # -------------------------------------------------------
        # FPS 計算邏輯 (輕量化)
        # -------------------------------------------------------
        frame_count += 1
        curr_time = time.time()
        
        # 定時更新視窗標題 (取代畫面壓字)
        if (curr_time - last_fps_update) > fps_update_interval:
            fps = frame_count / (curr_time - last_fps_update)
            
            # 更新視窗標題
            title_text = f"FPS: {fps:.1f} | Fmt: {real_format} | Res: {real_w}x{real_h}"
            cv2.setWindowTitle(window_name, title_text)
            
            # 重置計數器
            frame_count = 0
            last_fps_update = curr_time

        # -------------------------------------------------------
        # 等待按鍵 (1ms 是最小值，能讓視窗響應，同時不限制 FPS 上限)
        # -------------------------------------------------------
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    desired_format = 'MJPG' #'MJPG', 'YUYV', 'UYVY', 'YUY2'
    cam_index = 0
    main(desired_format, cam_index)