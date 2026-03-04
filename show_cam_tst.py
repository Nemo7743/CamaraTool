import cv2
import time

def decode_fourcc(v):
    """將數值解碼為 FOURCC 字串"""
    try:
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])
    except:
        return "UNKNOWN"

def main():
    # 嘗試 1: 維持使用 DirectShow
    backend = cv2.CAP_DSHOW
    print(f"正在開啟攝影機 (Backend: CAP_DSHOW)...")
    
    cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    # -----------------------------------------------------------
    # 強制設定流程 (順序很重要！)
    # -----------------------------------------------------------
    target_width = 640
    target_height = 480
    target_fps = 120.0
    fourcc_str = 'MJPG' 

    print("\n--- 開始設定參數 ---")

    # 1. 先設定 FOURCC (嘗試強迫切換編碼器)
    fourcc_val = cv2.VideoWriter_fourcc(*fourcc_str)
    ret_fourcc = cap.set(cv2.CAP_PROP_FOURCC, fourcc_val)
    print(f"設定 FOURCC ({fourcc_str}): {'成功' if ret_fourcc else '失敗'}")

    # 2. **關鍵技巧**：嘗試先設定 FPS
    # 如果硬體在 YUY2 640x480 下跑不到 120fps，先設定 FPS 可能會強迫驅動程式切換格式
    ret_fps = cap.set(cv2.CAP_PROP_FPS, target_fps)
    print(f"設定 FPS ({target_fps}): {'成功' if ret_fps else '失敗'}")

    # 3. 最後設定解析度
    ret_w = cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    ret_h = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    print(f"設定解析度 ({target_width}x{target_height}): {'成功' if (ret_w and ret_h) else '失敗'}")

    # -----------------------------------------------------------
    # 檢查實際生效的參數
    # -----------------------------------------------------------
    time.sleep(1) # 給鏡頭一點時間暖機與切換
    
    real_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)

    print("\n--- 最終生效參數 ---")
    print(f"格式: {real_fourcc} (目標: {fourcc_str})")
    print(f"解析度: {real_w}x{real_h}")
    print(f"FPS: {real_fps}")
    
    if real_fourcc.upper() != fourcc_str:
        print("\n⚠️ 警告: 格式鎖定失敗！鏡頭仍在使用舊格式。")
        print("可能原因：")
        print("1. 鏡頭不支援此解析度的 MJPG (有些鏡頭只在 720p/1080p 開啟 MJPG)。")
        print("2. Windows 驅動程式強制覆蓋了設定。")
    else:
        print("\n✅ 成功鎖定 MJPG 格式！")

    print("\n啟動預覽 (按 'q' 離開)...")

    # 視窗設定
    window_name = "Debug Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    prev_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(window_name, frame)

        # 簡單的 FPS 監控
        frame_count += 1
        curr_time = time.time()
        if curr_time - prev_time >= 1.0:
            print(f"實際運行 FPS: {frame_count}")
            frame_count = 0
            prev_time = curr_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()