try:
    from taps.executor import *
    from taps.plugins import get_executor_configs
    TAPS_AVAILABLE=True
except:
    TAPS_AVAILABLE=False

class TapsExecutor:

    def __init__(self):
        pass