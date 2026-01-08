import cv2

def check_camera_indices():
    # 測試 0 到 4 號的攝影機
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"[檢測] 發現攝影機 Index: {index}")
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera Index {index}", frame)
                print(f"       -> 按任意鍵關閉此視窗繼續檢測...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cap.release()
        else:
            print(f"[檢測] Index {index} 無裝置或無法開啟")

if __name__ == "__main__":
    print("正在掃描攝影機索引，請稍候...")
    check_camera_indices()
    print("檢測結束")