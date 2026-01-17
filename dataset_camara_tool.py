import cv2
import os

# ================= 設定區域 (依需求修改) =================
# 1. 儲存圖片的資料夾路徑
SAVE_FOLDER = "dataset_train/Hand/Hand3"

# 其他路徑備用
#SAVE_FOLDER = "dataset_val/Hand/Hand1"
#SAVE_FOLDER = "dataset_test/Hand/Hand1"
#SAVE_FOLDER = "dataset_train/Tool"
#SAVE_FOLDER = "dataset_train/Blok"
#SAVE_FOLDER = "dataset_train/SafeItem"
#SAVE_FOLDER = "dataset_train/HandandTool"

# (已移除 COUNTER_FILE 設定)

# 2. 攝影機編號(在筆電運作時因為 0 代表筆電視訊鏡頭，所以這邊要修改成 1 和 2)
CAM1_INDEX = 0
CAM2_INDEX = 1

# 3. 解析度設定
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# =======================================================

def find_next_index(folder_path):
    """
    掃描資料夾內的圖片，找出目前最大的數字編號，
    並回傳 (最大編號 + 1) 作為本次的起始點。
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"[系統] 已建立資料夾: {folder_path}")
        return 1

    files = os.listdir(folder_path)
    max_idx = 0
    
    # 篩選出 jpg 檔且檔名為純數字的檔案
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            name_part, _ = os.path.splitext(f)
            if name_part.isdigit():
                idx = int(name_part)
                if idx > max_idx:
                    max_idx = idx
    
    # 如果找到檔案，回傳 最大值+1；如果是空資料夾，回傳 1
    return max_idx + 1

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

    # 自動獲取當前起始編號 (取代原本讀取 txt 的功能)
    current_idx = find_next_index(SAVE_FOLDER)
    
    print(f"[就緒] 按下 '空白鍵 (Space)' 拍照，按下 'q' 離開。")
    print(f"[資訊] 偵測到資料夾進度，下一組照片起始編號為: {current_idx:04d}")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("[錯誤] 無法讀取影像幀")
            break

        # 顯示畫面 (依需求開啟或關閉視窗)
        #cv2.imshow("Camera 1", frame1)
        cv2.imshow("Camera 2", frame2)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print(f"[系統] 程式結束")
            break
        
        elif key == 32: # Space 鍵
            # 依據您原本的邏輯：
            # 第一顆鏡頭使用 current_idx
            filename1 = os.path.join(SAVE_FOLDER, f"{current_idx + 1:04d}.jpg")
            
            # 第二顆鏡頭使用 current_idx + 1 
            # (注意：這會導致每按一次快門，數字其實跳了兩號，或者您是有意錯開命名)
            filename2 = os.path.join(SAVE_FOLDER, f"{current_idx:04d}.jpg")

            # 存檔
            #cv2.imwrite(filename1, frame1)
            cv2.imwrite(filename2, frame2)

            print(f"[儲存] 已儲存至: {current_idx:04d}")

            # 更新記憶體中的編號即可，不需要寫入文字檔
            current_idx += 1 

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()