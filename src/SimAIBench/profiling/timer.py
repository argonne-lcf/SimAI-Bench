import time

class Timer:
    def __init__(self,label:str):
        self.label = label
        self.start = None
        self.end = None
        self.duration = None
    
    def start_timing(self):
        self.start = time.perf_counter()
    
    def stop_timing(self):
        self.end = time.perf_counter()
        self.duration = self.end - self.start
    
    def get_duration(self):
        return self.duration

    def __enter__(self):
        self.start_timing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_timing()