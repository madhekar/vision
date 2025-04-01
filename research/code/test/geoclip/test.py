import os
import torch
from geoclip import GeoCLIP

model = GeoCLIP()

image_path = "/Users/emadhekar/Downloads/gettyimages-2147692990-170667a.jpg"


sample_path = "/Users/emadhekar/tmp/images"

for rt, _, files in os.walk(sample_path, topdown=True):
    print("Location,Latitude,Longitude")
    for file in files:
        top_pred_gps, top_pred_prob = model.predict(os.path.join(rt, file), top_k=1)

        #print("Top 5 GPS Predictions")
        #print("=====================")
        for i in range(1):
            lat, lon = top_pred_gps[i]
            #print(f"{file}, Prediction {i+1}: ({lat:.6f}, {lon:.6f}): Prob: {top_pred_prob[i]:.6f}")
            # print(f"Probability: {top_pred_prob[i]:.6f}")
            print(f"{file},{lat:.6f},{lon:.6f},{top_pred_prob[i]:.6f}")
            #print("* * *")