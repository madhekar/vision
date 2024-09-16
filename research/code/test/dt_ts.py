import datetime

ts = 946717261.0
ts1 = 1481209858
print(datetime.datetime.fromtimestamp(ts))
print(datetime.datetime.fromtimestamp(ts1))
dt = datetime.datetime.now()
print(str(dt))

#sdt = datetime.datetime.strptime(str(dt), "%Y%m%d-%H%M%S")
print(dt.strftime("%Y%m%d-%H%M%S"))