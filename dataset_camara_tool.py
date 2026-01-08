import cv2
import os

# ================= 設定區域 (依需求修改) =================
# 1. 儲存圖片的資料夾路徑
SAVE_FOLDER = "dataset_train/tst01" 

# 2. 設定計數器紀錄檔的名稱
COUNTER_FILE = "counter.txt"

# 3. 攝影機編號
CAM1_INDEX = 0
CAM2_INDEX = 1

# 4. 解析度設定
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# =======================================================

def get_start_index(folder_path, counter_filename):
    """
    讀取或建立計數器檔案，回傳當前的編號。
    """
    counter_path = os.path.join(folder_path, counter_filename)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"[系統] 已建立資料夾: {folder_path}")
    
    if not os.path.exists(counter_path):
        with open(counter_path, "w") as f:
            f.write("0")
        print(f"[系統] 已建立計數檔，從 0 開始")
        return 0
    else:
        with open(counter_path, "r") as f:
            try:
                content = f.read().strip()
                return int(content) if content else 0
            except ValueError:
                print("[警告] 計數檔內容錯誤，重置為 0")
                return 0

def update_counter(folder_path, counter_filename, new_index):
    """
    更新計數器檔案到下一個號碼。
    """
    counter_path = os.path.join(folder_path, counter_filename)
    with open(counter_path, "w") as f:
        f.write(str(new_index))

def main():
    # 初始化攝影機
    cap1 = cv2.VideoCapture(CAM1_INDEX)
    cap2 = cv2.VideoCapture(CAM2_INDEX)

    # 設定解析度
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap1.isOpened() or not cap2.isOpened():
        print("[錯誤] 無法開啟攝影機，請檢查 USB 連接或索引編號。")
        return

    # 獲取當前起始編號
    current_idx = get_start_index(SAVE_FOLDER, COUNTER_FILE)
    print(f"[就緒] 按下 '空白鍵 (Space)' 拍照，按下 'q' 離開。")
    print(f"[資訊] 下一組照片將命名為: {current_idx:04d} 和 {current_idx+1:04d}")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("[錯誤] 無法讀取影像幀")
            break

        cv2.imshow("Camera 1", frame1)
        cv2.imshow("Camera 2", frame2)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print(f"[系統] 程式結束")
            break
        
        elif key == 32: # Space 鍵
            # 第一顆鏡頭使用 current_idx (例如 0000)
            filename1 = os.path.join(SAVE_FOLDER, f"{current_idx:04d}.jpg")
            
            # 第二顆鏡頭使用 current_idx + 1 (例如 0001)
            filename2 = os.path.join(SAVE_FOLDER, f"{current_idx + 1:04d}.jpg")

            # 存檔
            cv2.imwrite(filename1, frame1)
            cv2.imwrite(filename2, frame2)

            print(f"[儲存] 已儲存: {current_idx:04d}.jpg 與 {current_idx+1:04d}.jpg")

            current_idx += 2
            
            update_counter(SAVE_FOLDER, COUNTER_FILE, current_idx)

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()