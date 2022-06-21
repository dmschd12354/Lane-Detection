from codecs import ascii_decode
import cv2
import numpy as np

video = cv2.VideoCapture("road.mp4")
if video.isOpened():
    while True:
        vret, img = video.read()
        prev_left_fit = np.zeros(3,)
        prev_right_fit = np.zeros(3,)
        if vret:
            # ===== filtering =====
            '''
            특정 색상 영역 추출할 때 HSV 색 공간을 이용하는게 좋다.
            RGB는 어두운 사진에서 색상 영역 추출하면 잘 되지 않는다.
            '''
            # 흰색, 노란색 색상의 범위를 정해 해당하는 차선을 필터링한다.
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            # 흰색 필터
            ''' lower = ([minimum_blue, minimum_green, minimum_red]) / upper = ([maximum_blue, maximum_green, maximum_red])'''
            white_lower = np.array([20, 150, 20]) # Green 에 가까운 값 
            white_upper = np.array([255, 255, 255]) # 흰색  
            # 노란색 필터
            yellow_lower = np.array([0, 85, 81]) # 가장 어두운 노란색 값
            yellow_upper = np.array([190, 255, 255]); ''' B 채널이 255가 되면 흰색 / 0 이 되면 노란색 --> 적당히 190으로 upper bound를 정한다. '''
            # 흰색 필터링
            ''' cv2.inRange(img(hls), lowerbound, upperbound, dst=None) '''
            white_masked = cv2.inRange(hls, white_lower, white_upper)
            # 노란색 필터링
            yellow_masked = cv2.inRange(hls, yellow_lower, yellow_upper)
            # 두 결과를 합친다
            filtered = cv2.addWeighted(yellow_masked, 1, white_masked,1,0)
            # cv2.imshow("fil", filtered)
            # cv2.waitKey(25)

            # ===== BEV =====
            x = img.shape[1]
            y = img.shape[0]

            bev_pts1 = np.array([[630,380],[735,380],[1260,718],[260,718]], np.float32) # 임의로 설정 (환경에 따라 변환 필요)
            bev_pts2 = np.float32([[200,0],[1000,0],[1000,720],[200,720]]) # 이동할 output point
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
            # cv2.imshow("m", roi)
            # cv2.waitKey(1000)

            # ===== Histogram =====
            histogram = np.sum(roi[roi.shape[0]//2:, :], axis=0)
            midpoint = np.int(histogram.shape[0]/2)
            left_current = np.argmax(histogram[:midpoint]) # 왼쪽 차선의 시작 x값
            right_current = np.argmax(histogram[midpoint:]) + midpoint # 오른쪽 차선의 시작 x값
            ''' histogram을 보면 peak를 가진 곳이 왼쪽 차선과 오른쪽 차선이 있는 두 부분이다. 따라서 왼쪽 peak의 x값은 왼쪽 차선의 시작 x값이 된다. '''
            
            # ===== Window Sliding =====
            '''
            axis = -1 --> dstack (axis = -1을 기준(뒤)으로 합친다)
            axis = 0 --> vstack (0: 아래)
            axis = 1 --> hstack (1: 옆)
            '''
            out_img = np.dstack((roi, roi, roi)) # roi 3개를 병합해서 3차원 배열로 만든다

            nwindows = 8
            window_height = np.int(roi.shape[0] / nwindows)
            nonzero = roi.nonzero() # 선이 있는 부분의 인덱스만 저장 
            nonzero_y = np.array(nonzero[0]) # 선이 있는 부분 y의 인덱스 값
            nonzero_x = np.array(nonzero[1]) # 선이 있는 부분 x의 인덱스 값 
            margin = 100
            minpix = 50
            left_lane = []
            right_lane = []
            color = [0, 255, 0]
            thickness = 2

            for w in range(nwindows):
                win_y_low = roi.shape[0] - (w + 1) * window_height # window 윗부분 y값
                win_y_high = roi.shape[0] - w * window_height # window 아랫 부분 y값
                win_xleft_low = left_current - margin # 왼쪽 window 왼쪽 위 x값
                win_xleft_high = left_current + margin # 왼쪽 window 오른쪽 아래 x값
                win_xright_low = right_current - margin # 오른쪽 window 왼쪽 위 x값
                win_xright_high = right_current + margin # 오른쪽 window 오른쪽 아래 x값

                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness) # (왼쪽 window의 왼쪽 위 좌표), (왼쪽 window의 오른쪽 아래 좌표)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness) # (오른쪽 window의 왼쪽 위 좌표), (오른쪽 window의 오른쪽 아래 좌표)
                # nonzero에서 window안에 해당되는 좌표의 인덱스(nonzero에서의 인덱스)를 good_left에 저장
                good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
                # nonzero에서 window안에 해당되는 좌표의 인덱스를 good_right에 저장
                good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
                left_lane.append(good_left)
                right_lane.append(good_right)
                # cv2.imshow("Window Added", out_img)

                if len(good_left) > minpix:
                    left_current = np.int(np.mean(nonzero_x[good_left])) # 왼쪽 window안에 있는 x좌표들의 평균이 다음 iteration에서의 left_current가 된다
                    
                if len(good_right) > minpix:
                    right_current = np.int(np.mean(nonzero_x[good_right])) # 오른쪽 window안에 있는 x좌표들의 평균이 다음 iteration에서의 rights_current가 된다

            # left_lane/right_lane에 for loop에서 모은 good_left/good_right(인덱스)를 1차원 배열로 변환 후 저장
            left_lane = np.concatenate(left_lane) # np.concatenate() --> array를 1차원으로 합침 
            right_lane = np.concatenate(right_lane)

            # 추출된 좌표값들 저장
            leftx = nonzero_x[left_lane]
            lefty = nonzero_y[left_lane]
            rightx = nonzero_x[right_lane]
            righty = nonzero_y[right_lane]

            # 라인 검출 실패 시 대안: 전 이미지에서의 라인 유지
            try:
                left_fit = np.polyfit(lefty, leftx, 2) # polyfit은 2차 함수의 coefficients 반환
                right_fit = np.polyfit(righty, rightx, 2)
                prev_left_fit = left_fit
                prev_right_fit = right_fit
            except:
                left_fit = prev_left_fit
                right_fit = prev_right_fit

            ploty = np.linspace(0, roi.shape[0] - 1, roi.shape[0]) # plot을 위한 y값 (0~719)
            print(ploty)
            # 라인의 모든 x값 계산
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            ltx = np.trunc(left_fitx)  # np.trunc() --> 소수점 부분을 버림
            rtx = np.trunc(right_fitx)
            
            # 이미지에서 추출된 pixel(라인)들의 색 변환
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]
            
            # cv2.imshow("A",out_img)
            # plt.plot(left_fitx, ploty, color = 'yellow')
            # plt.plot(right_fitx, ploty, color = 'yellow')
            # plt.xlim(0, 1280)
            # plt.ylim(720, 0)
            # plt.show()

            ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

            # ===== Drawing Line =====
            left_fitx = ret['left_fitx'] # BEV에서 왼쪽 차선 x 좌표
            right_fitx = ret['right_fitx'] # BEV에서 오른쪽 차선 x 좌표
            ploty = ret['ploty'] # (0~719)

            warp_zero = np.zeros_like(roi).astype(np.uint8) # (720, 1280)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero)) # (720, 1280, 3)
            
            # pts_left, pts_right: 왼쪽, 오른쪽 차선의 모든 좌표값을 가진다. (1, 720, 2)
            # pts: pts_left 와 pts_right을 합친 것. (1, 1440, 2)
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))]) # np.vstack([left_fitx, ploty]).shape = (2,720) --> [[left_fitx], [ploty]]
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) # flipud: 상하반전 
            pts = np.hstack((pts_left, pts_right))

            mean_x = np.mean((left_fitx, right_fitx), axis=0)
            pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))]) # 두 차선 가운데 좌표 (1, 720, 2)
            # print(mean_x)
            # mid_fit = np.polyfit(mean_x, ploty, 1)
            # print(mid_fit)
            # mid_fitx = mid_fit[0] * ploty + mid_fit[1]
            # print(mid_fitx)
            # print(ploty.shape)
            # print(mid_fitx.shape)
            # color_warp[mid_fitx, ploty] = (0,255,255)
            # pts_mid = np.array([np.transpose(np.vstack([mid_fitx, ploty]))])
            # cv2.fillPoly(color_warp, np.int_([pts_mid]), (255,0,255))

            cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
            cv2.fillPoly(color_warp, np.int_([pts_mean]), (255, 255, 0))
            cv2.imshow("a", color_warp)

            # vec = pts_mean[0,0] - pts_mean[-1,-1] 
            # print(pts_mean.shape)  
            start = pts_mean[0,0]
            start = tuple(map(int, start))
            end = pts_mean[-1,-1]
            end = tuple(map(int,end))

            vecimg = cv2.arrowedLine(color_warp, start, end, [0,0,255], 10)
            cv2.imshow("vector", vecimg)
            # cv2.waitKey()

            # vec.reshape(3,1)
            # s = pts_mean[0,0]
            # news = np.reshape(np.append(s, 1),(3,1))   
            # # print(news)       
            
            # e = pts_mean[-1,-1]
            # newe = np.reshape(np.append(e, 1),(3,1))   
            # print(newe)
            # print(inv_matrix @ news)       
            # print(inv_matrix @ newe)
            
            newwarp = cv2.warpPerspective(color_warp, inv_matrix, (img.shape[1], img.shape[0]))
            result = cv2.addWeighted(img, 1, newwarp, 0.4, 0) # original 이미지에 복구한 이미지 합성

            cv2.imshow("S", result)
            cv2.waitKey(25)

        else:
            break   
else:
    print("Cannot not video")
video.release()
cv2.destroyAllWindows()
# ==================