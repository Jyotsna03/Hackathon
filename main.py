import cv2

image = cv2.imread("image.jpg")

new_image = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret, binary = cv2.threshold(gray, 100, 255,
                            cv2.THRESH_OTSU)

cv2.imshow('Binary image', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()


inverted_binary = ~binary
cv2.imshow('Inverted binary image', inverted_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()


contours, hierarchy = cv2.findContours(inverted_binary,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)


with_contours = cv2.drawContours(image, contours, -1, (255, 0, 255), 3)
cv2.imshow('Detected contours', with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Total number of contours detected: ' + str(len(contours)))


first_contour = cv2.drawContours(new_image, contours, 0, (255, 0, 255), 3)
cv2.imshow('First detected contour', first_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()


x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(first_contour, (x, y), (x + w, y + h), (255, 0, 0), 5)
cv2.imshow('First contour with bounding box', first_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()

for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    if (cv2.contourArea(c)) > 10:
        cv2.rectangle(with_contours, (x, y), (x + w, y + h), (255, 0, 0), 5)

cv2.imshow('All contours with bounding box', with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
