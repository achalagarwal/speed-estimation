import numpy as np
import cv2
import cv2
import numpy as np
import pickle
# cap = cv2.VideoCapture("vtest.avi")
cap = cv2.VideoCapture("../speed_challenge_2017/data/last_two_mins_with_stops.mp4")
# cap2 = cv2.VideoCapture("../speed_challenge_2017/data/trimmed.mp4") 


ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

speeds = np.loadtxt("../speed_challenge_2017/data/train.txt")

from sklearn.linear_model import LinearRegression
model = LinearRegression()
# model.fit(x, speeds)
flows = []

frames = -1
try:
    while(1):
        frames += 1
        if frames == 10000:
            raise(ResourceWarning("too many frames"))
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        print(frames)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        if frames % 5 == 1:
            flows.append(flow.reshape(-1,))
        # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # hsv[...,0] = ang*180/np.pi/2
        # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        # cv2.imshow('frame2',rgb)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break
        # elif k == ord('s'):
        #     cv2.imwrite('opticalfb.png',frame2)
        #     cv2.imwrite('opticalhsv.png',rgb)
        prvs = next
except:
    speeds_train = speeds[::-1][0:frames][::-1][::5]
    model.fit(flows, speeds_train)
    pickle.dump(model, open("model_pickled", 'wb'))
    # pickle.dump(flows, open("flows_pickled", 'wb'))

    # speeds_train = speeds[0:len(flows)]

    # # speeds_test = speeds[0:len(flows)][200:400]

    # flows_train = flows[0:200]
    # flows_test = flows[200:400]
    
    # model.fit(flows_train,speeds_train)
    # preds = model.predict(flows_test)
    # print(preds)
    # print(speeds_test)
    # sc = model.score(flows_test, speeds_test)

    # print(sc)
cap.release()
cv2.destroyAllWindows()