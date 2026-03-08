import cv2
import time

def check_camera_fps():
    # 打開預設視訊鏡頭 (0 代表第一部攝影機，若有外接鏡頭可嘗試 1, 2...)
    cap = cv2.VideoCapture(1)

    # 檢查鏡頭是否成功開啟
    if not cap.isOpened():
        print("錯誤：無法開啟鏡頭，請確認鏡頭是否連接或被其他程式佔用。")
        return

    # 讀取並顯示硬體宣告的預設 FPS
    hardware_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"硬體宣告的預設 FPS: {hardware_fps}")

    # 初始化計算實際 FPS 所需的時間變數
    prev_time = 0

    print("正在啟動鏡頭畫面... (在視窗上按下 'q' 鍵即可退出)")

    while True:
        # 讀取每一幀畫面
        ret, frame = cap.read()
        
        # 若無法讀取畫面則跳出迴圈
        if not ret:
            print("無法接收影像串流。")
            break

        # 取得當前時間並計算實際 FPS
        current_time = time.time()
        actual_fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # 在畫面的左上角加上 FPS 數值文字
        fps_text = f"FPS: {int(actual_fps)}"
        cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 255, 0), 2, cv2.LINE_AA)

        # 顯示影像視窗
        cv2.imshow('Camera FPS Check', frame)

        # 監聽鍵盤輸入，按下小寫 'q' 鍵即可離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 結束後釋放鏡頭資源並關閉所有 OpenCV 視窗
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_camera_fps()