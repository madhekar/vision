import cv2 as cv
import os

"""
https://pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/
"""

sample_path = "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup"

for rt, _, files in os.walk(sample_path, topdown=True):
    for file in files:
        img = cv.imread(os.path.join(rt, file))
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurScore = cv.Laplacian(grey, cv.CV_64F).var()
        score = cv.quality.QualityBRISQUE_compute(
            img,
            "/home/madhekar/work/home-media-app/models/brisque/brisque_model_live.yml",
            "/home/madhekar/work/home-media-app/models/brisque/brisque_range_live.yml",
        )

        print(f' >>file: {file} Blur Score: {blurScore}')
        print(f' >> BRISQUE Score: {score}')