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

if __name__ == "__main__":
    # 如果你有筆電內建鏡頭，外接 USB 攝影機通常是 index 1，否則試試 0
    test_camera_specs(camera_index=0)