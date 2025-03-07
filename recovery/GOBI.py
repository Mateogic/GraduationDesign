from .Recovery import *

class GOBIRecovery(Recovery):
    def __init__(self, hosts, env, training = False):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.utilHistory = []

    def run_model(self, time_series, original_decision):
        return original_decision