import os
from geoclip import GeoCLIP

model = GeoCLIP()
sample_path = "/Users/emadhekar/tmp/images"

for rt, _, files in os.walk(sample_path, topdown=True):
    print("Location,Latitude,Longitude")
    for file in files:
        top_pred_gps, top_pred_prob = model.predict(os.path.join(rt, file), top_k=1)
        if top_pred_prob[0] > 0.099:
           lat, lon = top_pred_gps[0]
           print(f"{file}-{top_pred_prob[0]:.6f},{lat:.6f},{lon:.6f}")
