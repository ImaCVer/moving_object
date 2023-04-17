import cv2
import numpy as np

gray_prev = None
points1 = None
points2 = None
st = None

def acceptTrackedPoint(a,b,c):
    return (c == 1) and ((abs(a[0][0] -b[0][0])**2 + abs(a[0][1] -b[0][1])**2)> 1) and (abs(a[0][0] -b[0][0]) > 1)

def swap(a, b):
    return b, a

def tracking(frame):
    global gray_prev, points1, points2, st

    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = frame.copy()

    # 添加特征点
    points1 = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 2)
    initial = points1


    if gray_prev is None:
        gray_prev = gray.copy()

    # gray_prev = wrap_img(gray, gray_prev)

    # 光流金字塔，输出图二的特征点
    lk_params = dict(winSize=(15, 15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    points2, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, points1, None, **lk_params)
    #单独用这个也可以，感觉没多大差异
    # points2, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, points1, None)

    # 去掉一些不好的特征点
    k = 0
    for i in range(0, points2.size):
        if i >= st.size:
            break
        if acceptTrackedPoint(initial[i], points2[i],st[i]) == True:
            initial[k] = initial[i]
            points2[k] = points2[i]
            k=k+1

    # 显示特征点和运动轨迹
    # 选择good points
    good_new = initial[st == 1]
    good_old = points2[st == 1]
    # if len(good_new) > 8:
    #     F, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5)
    #     # good_old = points1[mask.ravel() == 1]
    #     # good_new = points2[mask.ravel() == 1]
    #     lines1 = cv2.computeCorrespondEpilines(good_new.reshape(-1, 1, 2), 2, F.T)
    #
    #     lines1 = lines1.reshape(-1, 3)
    #
    #     c = output.shape[1]
    #
    #     for r in lines1:
    #         # color = tuple(np.random.randint(0, 255, 3).tolist())
    #         x0, y0 = map(int, [0, -r[2] / r[1]])
    #         x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
    #         output = cv2.line(output, (x0, y0), (x1, y1), (60,125,0), 1)


    # 绘制跟踪框
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        if i >= k:
            break
        a, b = new.ravel()
        c, d = old.ravel()
        output = cv2.line(output, (int(a), int(b)), (int(c),int(d)),(0,0,255), 1)
        output = cv2.circle(output, (int(c), int(d)), 3, (0,255,0), -1)

    # 把当前跟踪结果作为下一此参考
    points2, points1 = swap(points2, points1)
    gray_prev, gray = swap(gray_prev, gray)
    return output

def rotate_img(image, angle):
    '''
    旋转图像
    :param image: 输入图像
    :param angle: 输入旋转角度
    :return: 返回旋转后的图像
    '''
    height, width = image.shape[:2]
    center = ((width-1)/2, (height-1)/2)

    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    bound_w = height * np.abs(rotate_matrix[0, 1]) + width * np.abs(rotate_matrix[0, 0])
    bound_h = height * np.abs(rotate_matrix[0, 0]) + width * np.abs(rotate_matrix[0, 1])
    bound_w = int(round(bound_w, 10))
    bound_h = int(round(bound_h, 10))

    rotate_matrix[0, 2] += (bound_w-1) / 2 - center[0]
    rotate_matrix[1, 2] += (bound_h-1) / 2 - center[1]
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    return rotated_image


if __name__ == "__main__":
    video = cv2.VideoCapture(r'video_1.mp4')
    fps = video.get(cv2.CAP_PROP_FPS)
    success = True
    while success:
        # 读帧
        success, frame = video.read()
        if success == False :
            break
        # frame = rotate_img(frame, -90)
        result = tracking(frame)
        cv2.imshow('result', result)  # 显示
        cv2.waitKey(100)
        # cv2.waitKey(int(1000 / int(fps)))  # 设置延迟时间
    cv2.destroyAllWindows()
    video.release()