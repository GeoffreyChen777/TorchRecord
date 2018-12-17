import time


class Timer(object):
    def __init__(self, iter_length):
        self.start_time = 0
        self.iter_length = iter_length

    def start(self):
        self.start_time = time.time()

    def stamp(self, step):
        time_duration = time.time() - self.start_time
        rest_time = time_duration / (step+1) * (self.iter_length - step - 1)
        cur_hour, cur_min, cur_sec = self.convert_format(time_duration)
        rest_hour, rest_min, rest_sec =  self.convert_format(rest_time)
        log_string = "[{}:{}:{} < {}:{}:{}]".format(cur_hour, cur_min, cur_sec, rest_hour, rest_min, rest_sec)
        return log_string

    def stop(self):
        time_duration = time.time() - self.start_time
        cur_hour, cur_min, cur_sec = self.convert_format(time_duration)
        log_string = "[{}:{}:{}]".format(cur_hour, cur_min, cur_sec)
        self.start_time = 0
        return log_string

    @staticmethod
    def convert_format(sec):
        hour = "{:02}".format(int(sec // 3600))
        minu = "{:02}".format(int(sec // 60))
        sec = "{:02}".format(int(sec % 60))
        return hour, minu, sec
