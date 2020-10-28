import cv2
import numpy as np


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while (1):
        # get a frame
        ret, frame = cap.read()

        cropped = frame[230:320, 120:300]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((3, 3), np.float32) / 2
        # dst = cv2.filter2D(gray, -1, kernel)
        dilation = cv2.dilate(gray, kernel)

        ret, thresh1 = cv2.threshold(dilation, 89, 255, cv2.THRESH_BINARY_INV)
        blurred = cv2.GaussianBlur(thresh1, (3,3), 20)


        edged = cv2.Canny(blurred, 250, 255)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # img_s = np.zeros(cnts.shape)
        # img_s = cv2.drawContours(img_s, cnts, -1, (125, 25, 0), 5)

        # show a frame
        # print(cnts)
        # if len(cnts[0])<=8:
        for cnt in cnts[0]:
            min_rect = cv2.minAreaRect(cnt)  # min_area_rectangle
            min_rect = np.int0(cv2.boxPoints(min_rect))
            e1=cv2.drawContours(edged, [min_rect], 0, (255, 255, 255), 2)  # green

        # e1=cv2.drawContours(edged,cnts[0],-1,(255,255,255),1)
        cv2.imshow("capture", e1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
