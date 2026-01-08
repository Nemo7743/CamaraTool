import cv2
import time

def test_camera_specs(camera_index=0):
    # 1. 開啟攝影機
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"無法開啟攝影機 (Index: {camera_index})")
        return

    # 2. 強制設定編碼格式為 YUYV (即 YUY2)
    # FourCC 代碼: 'Y', 'U', 'Y', 'V'
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    
    # 驗證是否成功設定為 YUYV
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_name = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"--------------------------------------------------")
    print(f"當前攝影機編碼格式: {codec_name}")
    
    if codec_name.upper() not in ['YUYV', 'YUY2']:
        print("警告: 攝影機可能不支援強制設定為 YUYV，或者驅動程式覆蓋了設定。")
        print("測試結果可能混入了 MJPG，請留意。")
    print(f"--------------------------------------------------")

    # 3. 定義要測試的解析度清單 (包含 8MP, 5MP, 1080p, 720p, VGA)
    resolutions_to_test = [
        (3264, 2448), # 8MP (商家宣稱)
        (2592, 1944), # 5MP
        (1920, 1080), # FHD
        (1280, 720),  # HD
        (640, 480)    # VGA
    ]

    for width, height in resolutions_to_test:
        print(f"\n正在測試請求解析度: {width}x{height} ...")
        
        # 設定解析度
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 讀取實際生效的解析度 (硬體可能會拒絕不支援的解析度並回退到最近的支援值)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width == 0 or actual_height == 0:
            print(" -> 無法讀取此解析度")
            continue
            
        print(f" -> 實際生效解析度: {actual_width}x{actual_height}")
        
        # 4. FPS 測試迴圈
        num_frames = 60  # 測試總幀數
        start_time = time.time()
        frame_count = 0
        
        print(" -> 開始擷取影像流以計算 FPS...")
        
        try:
            while frame_count < num_frames:
                ret, frame = cap.read()
                if not ret:
                    print(" -> 讀取幀失敗 (丟包或頻寬不足)")
                    break
                frame_count += 1
                
                # 若超過 5 秒還沒跑完，強制停止 (避免低 FPS 時卡太久)
                if (time.time() - start_time) > 5:
                    break
        except KeyboardInterrupt:
            print(" -> 使用者中斷測試")
            break
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if elapsed_time > 0:
            actual_fps = frame_count / elapsed_time
            print(f" -> 測試結果: {actual_fps:.2f} FPS (在 YUYV 模式下)")
        else:
            print(" -> 時間過短無法計算")

    cap.release()
    print("\n測試結束")


def test_low_res_performance(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    
    # 1. 強制設定 YUY2
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    
    # 2. 嘗試關閉自動曝光 (為了測試極限 FPS，不一定每台相機都支援此指令)
    # 0.25 是某些驅動的關閉值，或是 0，視相機而定，無效會被忽略
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
    cap.set(cv2.CAP_PROP_EXPOSURE, -5) # 設定一個較短的曝光時間 (負值通常代表 2的冪次方分之一秒)

    # 3. 專攻低解析度測試
    # USB 攝影機常見的低解析度標準：QQVGA (160x120), QCIF (176x144), QVGA (320x240)
    low_res_targets = [
        (320, 240),
        (176, 144),
        (160, 120) 
    ]

    print(f"--- 開始低解析度極限測試 (YUYV) ---")
    print("提示：請確保環境光線非常充足，以免因曝光時間過長導致掉幀\n")

    for w, h in low_res_targets:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        # 讀取實際生效值
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"請求: {w}x{h} -> 實際硬體輸出: {real_w}x{real_h}")
        
        if real_w == 640 and w < 640:
             print("  [悲報] 攝影機不支援此低解析度，強制回退到了 640x480。")
             print("  這意味著不管你要多小的圖，USB 傳輸的都是 VGA 大小，FPS 無法提升。")
             continue

        # 測試 FPS
        start = time.time()
        count = 0
        while True:
            ret, _ = cap.read()
            if not ret: break
            count += 1
            if count >= 100: # 測 100 幀
                break
        
        duration = time.time() - start
        fps = count / duration
        print(f"  -> 實測 FPS: {fps:.2f}\n")

    cap.release()




#======== 使用函式 ========

if __name__ == "__main__":
    # 如果筆電內建鏡頭，外接 USB 攝影機通常是 index 1，否則試試 0
    camera_index = 0

    test_camera_specs(camera_index)
    #test_low_res_performance(camera_index)
