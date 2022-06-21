import cv2
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture("hard_road.mp4")
if video.isOpened():
    while True:
        vret, img = video.read()
        prev_left_fit = np.zeros(3,)
        prev_right_fit = np.zeros(3,)
        if vret:
            # ===== filtering =====
            # 흰색, 노란색 색상의 범위를 정해 해당하는 차선을 필터링한다.
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            # 흰색 필터
            white_lower = np.array([20, 150, 20])
            white_upper = np.array([255, 255, 255])
            # 노란색 필터
            yellow_lower = np.array([0, 85, 81])
            yellow_upper = np.array([190, 255, 255])
            # 흰색 필터링
            white_mask = cv2.inRange(hls, white_lower, white_upper)
            # 노란색 필터링
            yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
            # mask = cv2.bitwise_or(yellow_mask, white_mask)
            # masked = cv2.bitwise_and(img, img, mask = mask)
            # 두 결과를 합친다
            filtered = cv2.addWeighted(yellow_mask, 1, white_mask,1,0)
            cv2.imshow("a", filtered)
            cv2.waitKey(25)

            # ===== BEV =====
            x = img.shape[1]
            y = img.shape[0]

            bev_pts1 = np.array([[600,470],[740,470],[1080,680],[270,680]], np.float32)
            bev_pts2 = np.float32([[200,0],[1000,0],[980,720],[200,720]])
            matrix = cv2.getPerspectiveTransform(bev_pts1, bev_pts2)
            inv_matrix = cv2.getPerspectiveTransform(bev_pts2, bev_pts1)
            bev = cv2.warpPerspective(filtered, matrix,(x,y))
            cv2.imshow("Bird_Eye_View", bev)
            # cv2.waitKey(25)

            # ===== ROI =====
            shape = np.array(
                [[int(0.1*x), int(y)], [int(0.1*x), int(0.1*y)], [int(0.3*x), int(0.1*y)], [int(0.3*x), int(y)], [int(0.6*x), int(y)], [int(0.6*x), int(0.1*y)],[int(0.8*x), int(0.1*y)], [int(0.8*x), int(y)], [int(0.2*x), int(y)]])
            mask = np.zeros_like(bev)
            cv2.fillPoly(mask, np.int32([shape]), 255)
            roi = np.bitwise_and(bev, mask)
            cv2.imshow("m", roi)
            # cv2.waitKey(1000)

            # ===== Histogram =====
            histogram = np.sum(roi[roi.shape[0]//2:, :], axis=0)
            midpoint = np.int(histogram.shape[0]/2)
            left_current = np.argmax(histogram[:midpoint])
            right_current = np.argmax(histogram[midpoint:]) + midpoint
            
            # ===== Window Sliding =====
            out_img = np.dstack((roi, roi, roi))

            nwindows = 8
            window_height = np.int(roi.shape[0] / nwindows)
            nonzero = roi.nonzero()  # 선이 있는 부분의 인덱스만 저장 
            nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
            nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값 
            margin = 100
            minpix = 50
            left_lane = []
            right_lane = []
            color = [0, 255, 0]
            thickness = 2

            for w in range(nwindows):
                win_y_low = roi.shape[0] - (w + 1) * window_height  # window 윗부분
                win_y_high = roi.shape[0] - w * window_height  # window 아랫 부분
                win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
                win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
                win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위 
                win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
                good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
                good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
                left_lane.append(good_left)
                right_lane.append(good_right)
                cv2.imshow("oo", out_img)

                if len(good_left) > minpix:
                    left_current = np.int(np.mean(nonzero_x[good_left]))
                if len(good_right) > minpix:
                    right_current = np.int(np.mean(nonzero_x[good_right]))

            left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
            right_lane = np.concatenate(right_lane)

            leftx = nonzero_x[left_lane]
            lefty = nonzero_y[left_lane]
            rightx = nonzero_x[right_lane]
            righty = nonzero_y[right_lane]
            
            try:
                left_fit = np.polyfit(lefty, leftx, 2)
                # print(right_fit.shape)
                right_fit = np.polyfit(righty, rightx, 2)
                prev_left_fit=left_fit
                prev_right_fit=right_fit
            except:
                left_fit = prev_left_fit
                right_fit = prev_right_fit

            ploty = np.linspace(0, roi.shape[0] - 1, roi.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
            rtx = np.trunc(right_fitx)

            out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
            out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]
            
            cv2.imshow("A",out_img)
            # plt.plot(left_fitx, ploty, color = 'yellow')
            # plt.plot(right_fitx, ploty, color = 'yellow')
            # plt.xlim(0, 1280)
            # plt.ylim(720, 0)
            # plt.show()

            ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

            # ===== Drawing Line =====
            left_fitx = ret['left_fitx']
            right_fitx = ret['right_fitx']
            ploty = ret['ploty']

            warp_zero = np.zeros_like(roi).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            mean_x = np.mean((left_fitx, right_fitx), axis=0)
            pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

            cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
            cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))

            newwarp = cv2.warpPerspective(color_warp, inv_matrix, (img.shape[1], img.shape[0]))
            result = cv2.addWeighted(img, 1, newwarp, 0.4, 0)

            cv2.imshow("S", result)

        else:
            break   
else:
    print("Cannot not video")
video.release()
cv2.destroyAllWindows()
# ==================