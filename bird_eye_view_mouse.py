import cv2
import numpy as np

win_name = "scanning"
img = cv2.imread("img.jpg")
print(img.shape)
rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4,2), dtype=np.float32)

def onMouse(event, x, y, flags, param):  # 마우스 이벤트 콜백 함수 구현
    global  pts_cnt                     # 마우스로 찍은 좌표의 갯수 저장
    if event == cv2.EVENT_LBUTTONDOWN:  
        cv2.circle(draw, (x,y), 3, (0,255,0), -1) # 좌표에 초록색 동그라미 표시
        cv2.imshow(win_name, draw)

        pts[pts_cnt] = [x,y]            
        pts_cnt+=1
        if pts_cnt == 4:                      
            width, height = 500,500
            pts1 = pts
            print(pts1)
            pts2 = np.float32([[0,0],[width,0],[width,height],[0,height]])

            matrix = cv2.getPerspectiveTransform(pts1,pts2)
            result = cv2.warpPerspective(img, matrix,(width,height))
            cv2.imshow("Bird_Eye_View", result)
            cv2.waitKey(1000)

            result2 = cv2.warpPerspective(result, np.linalg.inv(matrix), (cols,rows))
            # cv2.imshow("original", result2)
            # cv2.waitKey(1000)
            # print(result2.shape)

            img3 = cv2.bitwise_or(img, result2)
            # img3 = cv2.add(img, result2)
            # cv2.imshow("output", img3)
            # cv2.waitKey(1000)
            

cv2.imshow(win_name, img)
cv2.setMouseCallback(win_name, onMouse)  
cv2.waitKey(0)
cv2.destroyAllWindows()