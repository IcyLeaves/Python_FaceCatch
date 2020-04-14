# encoding:utf-8
from collections import deque

import dlib
import numpy as np
import cv2

def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"): # 将包含68个特征的的shape转换为numpy array格式
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def resize(image, width=1200):  # 将待检测的image进行resize
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def getbgr(image,xx,yy):
    bb=0
    gg=0
    rr=0
    bgr = image[yy,xx]
    for ii in range(-1,1):
        for jj in range(-1,1):
            if(ii==0 and jj ==0):
                continue
            tx=xx+ii
            ty=yy+jj
            tbgr=image[ty,tx]
            bb+=tbgr[0]
            gg+=tbgr[1]
            rr+=tbgr[2]

    bb = int(bb/8/3 + bgr[0]/3*2)
    gg = int(gg/8/3+bgr[1]/3*2)
    rr = int(rr/8/3+bgr[2]/3*2)
    print(bb, gg, rr)
    return bgr


def bfs(image,edge_list,convex):
    que = deque()
    dis = np.zeros((image.shape[0],image.shape[1]), dtype=int)
    for node in edge_list:
        que.append((node[0],node[1]))
        dis[node[0],node[1]]=-10

    cc = 0
    while(que):
        cc+=1
        now = que.popleft()
        nbgr = image[now[1],now[0]]
        if (dis[now[0],now[1]]+1<0):
            for xx in range(-1,1):
                for yy in range(-1,1):
                    ty=yy+now[1]
                    tx=xx+now[0]
                    if(cv2.pointPolygonTest(convex,(tx,ty),False)<0):
                        if(dis[tx,ty]>dis[now[0],now[1]]+1):
                            dis[tx,ty]=dis[now[0],now[1]]+1
                            tbgr = image[ty,tx]
                            if(tbgr[0]+40<255):
                                image[ty, tx][0]=tbgr[0]+40
                            if (tbgr[1] - 10 < 255):
                                image[ty, tx][1] = tbgr[1] -10
                            if (tbgr[2] - 10 < 255):
                                image[ty, tx][2] = tbgr[2] -10
                            if( (tx,ty) not in que):
                                que.append((tx,ty))
            # image[now[1],now[0]]=[int(bb),int(gg),int(rr)]
    print(cc)
    return image

def feature():
    image_file = "test.jpg"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    image = cv2.imread(image_file)
    image = resize(image, width=1200)
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shapes = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        shape = shape[48:]
        shapes.append(shape)
        # (x, y, w, h) = rect_to_bb(rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(image, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for shape in shapes:
        print(shape)
        # id = 1
        # ss = (shape[2]+shape[13])/2
        # print(shape[2],image[shape[2][0],shape[2][1]])
        # print(shape[13],image[shape[13][0],shape[13][1]])
        # ss = ss.astype(np.int)
        # image = bfs(image,ss[0],ss[1])
        # print(ss)
        # bgr=image[ss[0],ss[1]]
        # print(bgr)
        # cv2.circle(image, (ss[0], ss[1]), 2, ((0 + bgr[0]) / 2, (0 + bgr[1]) / 2, (255 + bgr[2]) / 2), -1)
        # for (x, y) in shape:
        # #     # cv2.putText(image, "{}".format(id), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0),2)
        #     bgr = image[y,x]
        # #
        #
        #     cv2.circle(image, (x, y), 2, ((0+bgr[0])/2,(0+bgr[1])/2 , (255+bgr[2])/2), -1)
        sx = image.shape[0]
        sy = image.shape[1]
        hull = cv2.convexHull(shape)
        # cv2.drawContours(image, [hull], -1, (168, 100, 168), -1)
        cc = 0
        c2 = 0
        for xx in range(sx):
            for yy in range(sy):
                dist = cv2.pointPolygonTest(hull,(xx,yy),False)
                c2+=1
                # print(dist)
                if(dist>=0):
                    # image[yy,xx][0]=image[yy,xx][0]+50
                    # print(image[yy, xx][2])
                    if(image[yy, xx][2] + 40<=255):
                        # print('okkkkk')
                       image[yy, xx][2] = image[yy, xx][2] + 40
                    cc+=1
                    if (image[yy, xx][0] - 10 >=0):
                        image[yy, xx][0] = image[yy, xx][0] - 10
                    if (image[yy, xx][1] - 10 >=0):
                        image[yy, xx][1] = image[yy, xx][1] - 10
        edge_list=[]
        for xx in range(sx):
            for yy in range(sy):
                dist = cv2.pointPolygonTest(hull,(xx,yy),False)
                if(dist==0):
                    edge_list.append([xx,yy])
        # image = bfs(image,edge_list,hull)
        # print(cc,c2)

        # cv2.drawContours(image, [hull], -1, (168, 100, 168), -1)
        #     id+=1
    image=    cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imshow("Output", image)
    cv2.imwrite("process+"+image_file, image)
    cv2.waitKey(0)

if __name__ == "__main__":

    feature()
