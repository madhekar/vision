import cv2 as cv

img = cv.imread("/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/images.jpeg")
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurScore = cv.Laplacian(grey, cv.CV_64F).var()
score = cv.quality.QualityBRISQUE_compute(img, "brisque_model_live.yml", "brisque_range_live.yml")

print(f' >> Blur Score: {blurScore}')
print(f' >> BRISQUE Score: {score}')