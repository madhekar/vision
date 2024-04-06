import datetime


class Perf:
    def __init__(self):
        self._start = None
        self._end = None
        self._nframes = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self    
    
    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._nframes += 1    

    def time_elapsed(self):
        return (self._end - self._start).total_seconds()

    def perf_in_fps(self):
        return self._nframes / self.time_elapsed()

        

