import cv2 as cv

img = cv.imread("/Users/emadhekar/Pictures/square.jpg")
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurScore = cv.Laplacian(grey, cv.CV_64F).var()
score = cv.quality.QualityBRISQUE_compute(img, "brisque_model_live.yml", "brisque_range_live.yml")

print(f' >> Blur Score: {blurScore}')
print(f' >> BRISQUE Score: {score}')

cv.namedWindow("Output", cv.WINDOW_NORMAL)
cv.imshow("Output", img)
k = cv.waitKey(0)
