import numpy as np
import cv2
import cv2
import numpy as np
import pickle
# cap = cv2.VideoCapture("vtest.avi")
cap = cv2.VideoCapture("../speed_challenge_2017/data/train_test.mp4")
# cap2 = cv2.VideoCapture("../speed_challenge_2017/data/trimmed.mp4") 


ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

speeds = np.loadtxt("../speed_challenge_2017/data/train.txt")

from sklearn.linear_model import LinearRegression
model = LinearRegression()
# model.fit(x, speeds)
feature_list = []

ignore_regions = pickle.load(open("regions.pkl", "rb"))
frame = -1
try:
    while(1):
        frame += 1
        if frame == 10000:
            raise(ResourceWarning("too many frames"))
        
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        print(frame)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        


        # create a masked array using ignore regions
        ## ignore regions in frame and frame+1

        # set mask to false
    
        mask = np.zeros(flow.shape, dtype=np.bool)
        mask3 = np.zeros((640,480,3), dtype=np.bool)
        regions = ignore_regions[1193+frame] + ignore_regions[1193+frame+1]
        for region in regions:
            
            # update mask
            mask[region[0][0]:region[1][0]+1,region[0][1]:region[1][1]+1,:] = 1
            mask3[region[0][0]:region[1][0]+1,region[0][1]:region[1][1]+1,:] = 1

        


        # mean the masked array along the rows and each row is a feature for LR
        masked_array = np.ma.array(flow, mask=mask)

        # ensure that the axis that is getting reduced is the width!
        # so there must be a feature for each height
        features = np.ma.filled(np.ma.mean(masked_array, axis=1), 0)
        feature_list.append(features.reshape(-1,))
        
        # temp = np.ma.filled(masked_array,0)
        # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # hsv[...,0] = ang*180/np.pi/2
        # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        # rgb_masked = np.ma.array(rgb, mask=mask3)
        # k = cv2.waitKey(0) & 0xff
        # cv2.destroyAllWindows()
        # cv2.namedWindow("frame2")
        # cv2.namedWindow("frame3")
        # cv2.imshow('frame2',np.ma.filled(rgb_masked, 0))
        # cv2.imshow('frame3', rgb)
        
        
        # if k == 27:
        #     break
        # elif k == ord('s'):
        #     cv2.imwrite('opticalfb.png',frame2)
        #     cv2.imwrite('opticalhsv.png',rgb)
        prvs = next
except Exception as e:

    model = pickle.load(open("model_pickled_07", 'rb'))
    # model.fit(feature_list, speeds[1193:1193+frame])
    # speeds_train = speeds[0:frames][::100]
    print(model.score(feature_list, speeds[1193:1193+frame]))
    pickle.dump(model, open("model_pickled_07", 'wb'))
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