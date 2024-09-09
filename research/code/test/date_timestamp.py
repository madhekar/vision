import datetime
from dateutil import parser

dt = "02:04:2024 29:34:15"

dto = datetime.datetime(parser.parse(dt))

print(str(dto))