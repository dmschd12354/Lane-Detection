import cv2
from matplotlib.pyplot import new_figure_manager
import numpy as np

prev_frame_left = []
prev_frame_right = []
global nframe
nframe = 0

def filtering(img):
    '''
    특정 색상 영역 추출할 때 HSV 색 공간을 이용하는게 좋다.
    RGB는 어두운 사진에서 색상 영역 추출하면 잘 되지 않는다.
    '''
    # 흰색, 노란색 색상의 범위를 정해 해당하는 차선을 필터링한다.
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) 
    cv2.imshow("HLS", hls)  
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
    cv2.imshow("w", white_masked)  
    # 노란색 필터링
    yellow_masked = cv2.inRange(hls, yellow_lower, yellow_upper)
    cv2.imshow("y", yellow_masked)  
    # 두 결과를 합친다
    filtered = cv2.addWeighted(yellow_masked, 1, white_masked,1,0)
    cv2.imshow("filtered", filtered)  

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret,filtered = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    # cv2.imshow("filtered", filtered)

    # ''' https://medium.com/@rjm2017/lane-detector-2-birds-eye-view-2d4edfb0f3bc '''
    # img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) 
    # img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)   
    
    # Rc = img[:,:,2]
    # Sc = img_hls[:,:,2]   
    # Bc = img_lab[:,:,2]  

    # ret1, dst1 = cv2.threshold(Rc, 180, 255, cv2.THRESH_BINARY)
    # ret2, dst2 = cv2.threshold(Sc, 80, 255, cv2.THRESH_BINARY)
    # ret3, dst3 = cv2.threshold(Bc, 180, 255, cv2.THRESH_BINARY)

    # a = cv2.bitwise_or(dst1, dst2)
    # filtered = cv2.bitwise_or(a, dst3)

    return filtered


def BEV(x, y, filtered):
    bev_pts1 = np.array([ [561,454],[709,454],[1180,684],[273,684] ], np.float32) # 임의로 설정 (환경에 따라 변환 필요) [577,463],[771,468],[1111,712],[218,699] [630,380],[735,380],[1260,718],[260,718]     [600,450],[730,450],[1100,720],[270,720] 
    bev_pts2 = np.float32([[200,0],[1000,0],[1000,720],[200,720]]) # 이동할 output point
    matrix = cv2.getPerspectiveTransform(bev_pts1, bev_pts2)
    inv_matrix = cv2.getPerspectiveTransform(bev_pts2, bev_pts1)
    bev = cv2.warpPerspective(filtered, matrix,(x,y))

    return bev, inv_matrix


def ROI(x, y, bev):
    shape = np.array([[int(0.1*x), int(y)], [int(0.1*x), int(0.1*y)], [int(0.3*x), int(0.1*y)], [int(0.3*x), int(y)], [int(0.6*x), int(y)], [int(0.6*x), int(0.1*y)],[int(0.8*x), int(0.1*y)], [int(0.8*x), int(y)], [int(0.2*x), int(y)]])
    mask = np.zeros_like(bev)
    cv2.fillPoly(mask, np.int32([shape]), 255)
    roi = np.bitwise_and(bev, mask)

    return roi


def hist(roi):
    histogram = np.sum(roi[roi.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    left_current = np.argmax(histogram[:midpoint]) # 왼쪽 차선의 시작 x값
    right_current = np.argmax(histogram[midpoint:]) + midpoint # 오른쪽 차선의 시작 x값
    ''' histogram을 보면 peak를 가진 곳이 왼쪽 차선과 오른쪽 차선이 있는 두 부분이다. 따라서 왼쪽 peak의 x값은 왼쪽 차선의 시작 x값이 된다. '''

    return left_current, right_current


def window_sliding(roi, rec_added, lane_added, left_current, right_current, prev_left_fit, prev_right_fit):
    '''
    axis = -1 --> dstack (axis = -1을 기준(뒤)으로 합친다)
    axis = 0 --> vstack (0: 아래)
    axis = 1 --> hstack (1: 옆)
    '''

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

        cv2.rectangle(rec_added, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness) # (왼쪽 window의 왼쪽 위 좌표), (왼쪽 window의 오른쪽 아래 좌표)
        cv2.rectangle(rec_added, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness) # (오른쪽 window의 왼쪽 위 좌표), (오른쪽 window의 오른쪽 아래 좌표)
        # nonzero에서 window안에 해당되는 좌표의 인덱스(nonzero에서의 인덱스)를 good_left에 저장
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        # nonzero에서 window안에 해당되는 좌표의 인덱스를 good_right에 저장
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        cv2.imshow("Window Added", rec_added)

        if len(good_left) > minpix:
            left_current = np.int(np.mean(nonzero_x[good_left])) # 왼쪽 window안에 있는 x좌표들의 평균이 다음 iteration에서의 left_current가 된다
            
        if len(good_right) > minpix:
            right_current = np.int(np.mean(nonzero_x[good_right])) # 오른쪽 window안에 있는 x좌표들의 평균이 다음 iteration에서의 rights_current가 된다

    # cv2.imshow("Rectangle added",rec_added)

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
    # 라인의 모든 x값 계산
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)  # np.trunc() --> 소수점 부분을 버림
    rtx = np.trunc(right_fitx)

    # 이미지에서 추출된 pixel(라인)들의 색 변환
    lane_added[lefty, leftx] = [255, 0, 0]
    lane_added[righty, rightx] = [0, 0, 255]
    
    cv2.imshow("Lane added",lane_added)

    ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty} 

    return ret, rec_added, lane_added


def Drawing_line(ret, lane_filled, lane_added, inv_matrix):
    left_fitx = ret['left_fitx'] # BEV에서 왼쪽 차선 x 좌표
    right_fitx = ret['right_fitx'] # BEV에서 오른쪽 차선 x 좌표
    ploty = ret['ploty'] # (0~719)

    # warp_zero = np.zeros_like(roi).astype(np.uint8) # (720, 1280)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero)) # (720, 1280, 3)

    # pts_left, pts_right: 왼쪽, 오른쪽 차선의 모든 좌표값을 가진다. (1, 720, 2)
    # pts: pts_left 와 pts_right을 합친 것. (1, 1440, 2)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))]) # np.vstack([left_fitx, ploty]).shape = (2,720) --> [[left_fitx], [ploty]]
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) # flipud: 상하반전 
    
    # %%%%% 전의 5개 프레임에서의 라인 좌표들의 평균을 이용해 부드럽게 표현 %%%%%
    global nframe
    if nframe < 5:
        prev_frame_left.append(pts_left)
        prev_frame_right.append(pts_right)
        nframe += 1
    
    temp_left = np.zeros((1,720,2))
    temp_right = np.zeros((1,720,2))
    for i in range(nframe):
        temp_left += prev_frame_left[i]
        temp_right += prev_frame_right[i]
    temp_left /= nframe
    temp_right /= nframe

    if nframe == 5:
        del prev_frame_left[0]
        del prev_frame_right[0]
        nframe -= 1
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    pts = np.hstack((temp_left, temp_right))
            
    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))]) # 두 차선 가운데 좌표 (1, 720, 2)

    cv2.fillPoly(lane_filled, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(lane_filled, np.int_([pts_mean]), (216, 168, 74)) # 255, 255, 0
    cv2.imshow("Lane filled", lane_filled)
            
    newwarp1 = cv2.warpPerspective(lane_added, inv_matrix, (img.shape[1], img.shape[0]))
    newwarp2 = cv2.warpPerspective(lane_filled, inv_matrix, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, 1, newwarp1, 1, 0) # original 이미지에 복구한 이미지 합성
    result = cv2.addWeighted(result, 1, newwarp2, 0.2, 0)

    # %%%%%% 진행 방향 벡터 ??? %%%%%%
    s = pts_mean[0,0]
    e = pts_mean[0,300]
    ee = np.reshape(np.append(e, 1),(3,)) 
    ss = np.reshape(np.append(s, 1),(3,)) 
    e_img = inv_matrix @ ee
    s_img = inv_matrix @ ss
    e_img = (e_img / e_img[2])
    s_img = (s_img / s_img[2])
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # %%%%%% 차선 중앙 이탈 (카메라가 차 중앙에 위치한다고 가정) %%%%%
    mid_in_warped = (pts_right[0,0,0] + pts_left[-1,-1,0])/2 # 차선 중심 in warped image (x-coor)
    temp = inv_matrix @ [mid_in_warped,720,1] # original image의 coordinate로 변환
    mid_road_in_img = (temp / temp[2])[1]
    mid_in_img = img.shape[1]/2 + 0 # 차의 중심 in original image (x-coor)
    cv2.line(result, (int(mid_in_img),680), (int(mid_in_img),720), (0,0,255), 5) # 차의 중심
    cv2.line(result, (int(mid_road_in_img),680), (int(mid_road_in_img),720), (255,0,0), 5) # 차선의 중심
    offset = str(mid_road_in_img - mid_in_img)
    cv2.putText(result, "offset: " +offset+"px", (50,80), cv2.FONT_HERSHEY_PLAIN, 2,(200,200,200), 2)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # cv2.arrowedLine(result, (int(mid_in_img),720), (int(mid_road_in_img),600), (0,255,0), 2) # 진행 방향 벡터
    cv2.putText(result, "Center of the car", (50,120), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 2)
    cv2.putText(result, "Center of the lane", (50,160), cv2.FONT_HERSHEY_PLAIN, 2,(255,0,0), 2)


    return result


video = cv2.VideoCapture("curved_road.mp4")

if video.isOpened():
    while True:
        vret, img = video.read()
        prev_left_fit = np.zeros(3,)
        prev_right_fit = np.zeros(3,)
        if vret:
            # ===== filtering =====
            filtered = filtering(img)
            # ===== BEV =====
            x = img.shape[1]
            y = img.shape[0]
            bev, inv_matrix = BEV(x, y, filtered)
            cv2.imshow("Bird_Eye_View", bev)
            # cv2.waitKey(25)
            # ===== ROI =====
            roi = ROI(x, y, bev)
            # cv2.imshow("m", roi)
            # cv2.waitKey(1000)
            # ===== Histogram =====
            left_current, right_current = hist(roi)            
            # ===== Window Sliding =====
            rec_added = np.dstack((roi, roi, roi)) # roi 3개를 병합해서 3차원 배열로 만든다
            lane_added = np.copy(rec_added)
            lane_filled = np.copy(rec_added)
            ret, rec_added, lane_added = window_sliding(roi, rec_added, lane_added, left_current, right_current, prev_left_fit, prev_right_fit)
            # ===== Drawing Line =====
            result = Drawing_line(ret, lane_filled, lane_added, inv_matrix)

            cv2.imshow("Result", result)
            cv2.waitKey(25)

        else:
            break   
else:
    print("Cannot not load video")
video.release()
cv2.destroyAllWindows()
