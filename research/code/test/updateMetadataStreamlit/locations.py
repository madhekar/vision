
import os
import pandas as pd

class loc:
    def __init__(self) -> None:
        self.lat = None
        self.lon = None
        self.name = None
        self.locdict = dict()
    
    def addLocation(self, lat, lon, name, desc):
        if name not in self.locdict:
           self.locdict[name] = {'lat': lat, 'lon': lon, 'desc': desc}

    def getLocation(self, name):
        return self.locdict(name)       
    
    def loadLocations(self):
        df_loc = pd.read_csv("locations.csv")

