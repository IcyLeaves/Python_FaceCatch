# encoding:utf-8
from collections import deque

import dlib
import numpy as np
import cv2


# 4/16 cy:

# 得到所有上色点的dist和dist_inside，绘制直方图（Part 1)
# 统计示例图的上色像素个数：5585,由此可得，在嘴唇内部进行bfs似乎可行
# 代码现存问题：上色部分还比较粗糙，最直接的bug是颜色会越界变成黑色(Part 2-2)
# 代码现存问题：嘴唇实际上是凹多边形，凸包则会导致上嘴唇的凹陷区域算作内部(Part 2-1)
# Part2-1解决：将凹多边形分成2个凸多边形，生成<外嘴唇-左凸包>和<外嘴唇-右凸包>
# ---最后上色区域为：(<外嘴唇-左凸包>内部+边界 或 <外嘴唇-右凸包>内部+边界) 且 <内嘴唇凸包>内部+边界
# 下一步：探究bfs（魔棒法）在边缘部分的可行性？


def rect_to_bb(rect):  # [psy]:获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def shape_to_np(shape, dtype="int"):  # [psy]:将包含68个特征的的shape转换为numpy array格式
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def resize(image, width=1200):  # [psy]:将待检测的image进行resize
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def getbgr(image, xx, yy):
    bb = 0
    gg = 0
    rr = 0
    bgr = image[yy, xx]
    for ii in range(-1, 1):
        for jj in range(-1, 1):
            if ii == 0 and jj == 0:
                continue
            tx = xx + ii
            ty = yy + jj
            tbgr = image[ty, tx]
            bb += tbgr[0]
            gg += tbgr[1]
            rr += tbgr[2]

    bb = int(bb / 8 / 3 + bgr[0] / 3 * 2)
    gg = int(gg / 8 / 3 + bgr[1] / 3 * 2)
    rr = int(rr / 8 / 3 + bgr[2] / 3 * 2)
    print(bb, gg, rr)
    return bgr


def bfs(image, edge_list, convex):
    que = deque()
    dis = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    for node in edge_list:
        que.append((node[0], node[1]))
        dis[node[0], node[1]] = -10

    cc = 0
    while que:
        cc += 1
        now = que.popleft()
        # nbgr = image[now[1], now[0]]
        if dis[now[0], now[1]] + 1 < 0:
            for xx in range(-1, 1):
                for yy in range(-1, 1):
                    ty = yy + now[1]
                    tx = xx + now[0]
                    if cv2.pointPolygonTest(convex, (tx, ty), False) < 0:
                        if dis[tx, ty] > dis[now[0], now[1]] + 1:
                            dis[tx, ty] = dis[now[0], now[1]] + 1
                            tbgr = image[ty, tx]
                            if tbgr[0] + 40 < 255:
                                image[ty, tx][0] = tbgr[0] + 40
                            if tbgr[1] - 10 < 255:
                                image[ty, tx][1] = tbgr[1] - 10
                            if tbgr[2] - 10 < 255:
                                image[ty, tx][2] = tbgr[2] - 10
                            if (tx, ty) not in que:
                                que.append((tx, ty))
    print(cc)
    return image


def soft_edge():  # [cy]:边缘自然化函数
    return False


def feature(path, color):
    detector = dlib.get_frontal_face_detector()  # [cy]:人脸检测仪
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # [cy]:关键点检测器
    image = cv2.imread(path)  # [cy]:读取 输入图像.jpg
    image = resize(image, width=1200)  # [cy]:缩放 图像，宽为1200
    print(image.shape)  # [cy]:image尺寸 (高,宽,3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # [psy]:转为灰度图
    # [cy]:
    # 传入灰度图像，检测出里面的所有脸，某张脸的矩形：rect=[(左上角坐标),(右下角坐标)]
    # OpenCV坐标
    # (0,0) - (100,0)
    #   |        |
    # (0,100) - (100,100)
    rects = detector(gray, 1)  # [psy]:灰度图里定位人脸
    shapes = []  # [psy]:shapes存储找到的人脸框，人脸框仅包含四个角数值如frontal_face_detector.png所示。
    for (i, rect) in enumerate(rects):  # [cy]:遍历所有脸的方框
        shape = predictor(gray, rect)  # [cy]:用关键点检测器检测出关键点们
        shape = shape_to_np(shape)  # [cy]:关键点们变成numpy数组
        shape = shape[48:]  # [cy]:关键点[48-67]是嘴唇区域
        shapes.append(shape)  # [cy]:把这张脸的嘴唇关键点插入到shapes
    # [psy]:
    # 图片转为hsv形式，色调（H），饱和度（S），亮度（V）
    # H:  0 — 180
    # S:  0 — 255
    # V:  0 — 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # [cy]:
    for shape in shapes:
        # [cy]:遍历每个嘴唇
        # [cy]:图像的高与宽
        sx, sy = image.shape[0], image.shape[1]
        # [cy]:外嘴唇左半边凸包
        hull_left = cv2.convexHull(np.concatenate((shape[0:4], shape[7:12])))
        # [cy]:外嘴唇右半边凸包
        hull_right = cv2.convexHull(shape[3:8])
        # [psy]:内嘴唇凸包
        hull_inside = cv2.convexHull(shape[12:])
        for xx in range(sx):
            for yy in range(sy):
                # [cy]:获得(xx,yy)到左外凸包的距离，正数说明在内部，measureDist:是否返回准确距离
                dist_left = cv2.pointPolygonTest(hull_left, (xx, yy), measureDist=False)
                # [cy]:获得(xx,yy)到右外凸包的距离，正数说明在内部，measureDist:是否返回准确距离
                dist_right = cv2.pointPolygonTest(hull_right, (xx, yy), measureDist=False)
                # [cy]:获得(xx,yy)到内凸包的距离，正数说明在内部，measureDist:是否返回准确距离
                dist_inside = cv2.pointPolygonTest(hull_inside, (xx, yy), measureDist=False)
                # [cy]:正在开发边缘自然化方法
                # add=soft_edge(hull, dist, dist_inside)
                # [cy]:(在外嘴唇左凸包以内或在外嘴唇右凸包以内)且在内嘴唇凸包以外为嘴唇
                if (dist_left >= 0 or dist_right >= 0) and dist_inside < 0:
                    image[yy, xx][0] = color[0]
                    image[yy, xx][1] = color[1]
                    image[yy, xx][2] += 10
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)  # [cy]:重新将图像转为BGR格式
    # image = cv2.GaussianBlur(image, (7, 7), 0)  # [cy]:高斯模糊图像，我认为没有用处
    # [cy]:可以查看凹多边形分成2个凸多边形
    # shape=shapes[0]
    # polygon1 = np.concatenate((shape[0:4], shape[7:12]))  # 48 49 50 51  55 56 57 58 59
    # polygon2 = shape[3:8] # 51 52 53 54 55
    # cv2.polylines(image, [polygon1], True, (0, 0, 0), 2)
    # cv2.polylines(image, [polygon2], True, (0, 0, 250), 2)
    return image


if __name__ == "__main__":
    input_image_path = "test2.jpg"  # [cy]:输入图像.jpg
    lipstick_color = [175, 150, 0]  # [cy]:嘴唇颜色
    image_output = feature(input_image_path, lipstick_color)  # [cy]:处理图像
    cv2.imshow("Output", image_output)  # [cy]:显示 输出图像2.jpg
    cv2.imwrite("process2+" + input_image_path, image_output)  # [cy]:保存 输出图像2.jpg
    cv2.waitKey(0)
