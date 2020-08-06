import numpy as np
import cv2

vid = cv2.VideoCapture("../speed_challenge_2017/data/trimmed.mp4")
# img = cv2.pyrDown(cv2.imread('2011-volvo-s60_100323431_h.jpg', cv2.IMREAD_UNCHANGED))

grabbed, frame = vid.read()
img = frame
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# do i need gaussian blur
# gassian blur before or after b&w -- currently b&w works with 3 channels so work acc.



img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("b&w", img)

# blur = cv2.GaussianBlur(img,(5,5),0)
# cv2.imshow("blur", blur)

thresholds = [
    cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        cv2.THRESH_BINARY, 23, 2),
    cv2.threshold(img, 112, 255, cv2.THRESH_TRIANGLE)[1:],
    cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1:],
    cv2.threshold(img, 20, 255, cv2.THRESH_BINARY_INV)[1:],
    
]

for i, threshed_img in enumerate(thresholds):
    cv2.imshow(str(i), threshed_img)
    # cv2.waitkey(0)
threshed_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,57,2)
# ret, threshed_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# ret, threshed_img = cv2.threshold(img,
#                 112, 255, cv2.THRESH_TRIANGLE)
# find contours and get the external one

cv2.imshow("threshed", threshed_img)

# contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[-2:]

n=len(contours)-1
contours=sorted(contours,key=cv2.contourArea,reverse=True)[:n]

for c in contours:
    hull=cv2.convexHull(c)
    cv2.drawContours(img,[hull],0,(0,255,0),2)
    cv2.imshow('convex hull',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
#                cv2.CHAIN_APPROX_SIMPLE)

# with each contour, draw boundingRect in green
# a minAreaRect in red and
# a minEnclosingCircle in blue
for c in contours:
    # get the bounding rect
    print(c)
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # get the min area rect
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a red 'nghien' rectangle
    cv2.drawContours(img, [box], 0, (0, 0, 255))

    # finally, get the min enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    # convert all values to int
    center = (int(x), int(y))
    radius = int(radius)
    # and draw the circle in blue
    img = cv2.circle(img, center, radius, (255, 0, 0), 2)

print(len(contours))
cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

cv2.imshow("contours", img)

cv2.imshow("contours", img)

while True:
    key = cv2.waitKey(1)
    if key == 27: #ESC key to break
        break

cv2.destroyAllWindows()