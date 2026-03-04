import cv2
import time

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def test_camera_specs_dshow(camera_index=0):
    print(f"準備測試相機 Index: {camera_index} (使用 DirectShow 後端)...")

    # 測試項目
    tests = [
        {'fourcc': 'MJPG', 'w': 640, 'h': 480},
        {'fourcc': 'YUYV', 'w': 640, 'h': 480},
        {'fourcc': 'UYVY', 'w': 640, 'h': 480},
        {'fourcc': 'MJPG', 'w': 1920, 'h': 1080}
    ]

    for t in tests:
        target_fmt = t['fourcc']
        w, h = t['w'], t['h']
        
        print(f"\n--------------------------------------------------")
        print(f"測試請求: {target_fmt} @ {w}x{h}")

        # === 修改重點：將開啟相機移到迴圈內 ===
        # 每次測試都重新開啟，強迫驅動程式重新初始化
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("無法開啟相機")
            continue

        # 設定 (在讀取任何 frame 之前設定)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*target_fmt))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        # --- 額外建議：強制關閉自動曝光 (若需測試極限 FPS) ---
        # 0.25 是大約的曝光值，負數在某些驅動代表手動模式
        # cap.set(cv2.CAP_PROP_EXPOSURE, -4) 
        
        # 讀取實際值
        actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_codec = decode_fourcc(actual_fourcc_int).upper()
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f" -> 相機回應: {actual_codec} @ {actual_w}x{actual_h}")
        
        # 判斷是否設定成功
        # YUYV, YUY2, YUNV 常常是同一個東西
        is_match = (target_fmt == actual_codec) or \
                   (target_fmt == 'YUYV' and actual_codec == 'YUY2') or \
                   (target_fmt == 'MJPG' and actual_codec == 'MJPG')
        
        if not is_match:
            print(f" -> [失敗] 相機不支援 {target_fmt}，它回傳了 {actual_codec}")
            cap.release() # 記得釋放
            continue
            
        # 測試 FPS
        print(" -> 開始 FPS 測試 (抓取 60 幀)...")
        
        # 暖機
        for _ in range(5):
            cap.read()
            
        start = time.time()
        count = 0
        while count < 60:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            
        elapsed = time.time() - start
        fps = count / elapsed if elapsed > 0 else 0
        print(f" -> 實測 FPS: {fps:.2f}")

        # === 修改重點：每次測完釋放相機 ===
        cap.release()

    print("\n測試結束")

if __name__ == "__main__":
    # 請確認你的相機 Index
    test_camera_specs_dshow(camera_index=1)