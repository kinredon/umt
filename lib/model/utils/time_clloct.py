import time
from collections import defaultdict
from prettytable import PrettyTable


class TimeColloct():
    def __init__(self):
        self._names = []
        self._start_timer = {}
        self._end_timer = {}
        self._time_gap = defaultdict(lambda: 0.0)

    def _is_duplicated(self, name):
        if name in self._names:
            raise f"The duplicated name: {name}"

    def start(self, name):
        if name not in self._names:
            self._names.append(name)
        self._start_timer[name] = time.time()

    def end(self, name):
        if name not in self._names:
            raise f"No such a timer: {name}"
        self._end_timer[name] = time.time()
        gap = self._end_timer[name] - self._start_timer[name]
        if gap < 0:
            raise f"Time gap is less than zero in timer {name}"
        self._time_gap[name] += gap

    def clloct(self):
        pt = PrettyTable()
        pt.field_names = self._names
        time_gap = []
        for name in self._names:
            time_gap.append(self._time_gap[name])
        pt.add_row(time_gap)
        pt.float_format = "0.3"
        print(pt)

    def clear(self):
        self.__init__()


if __name__ == '__main__':
    timer = TimeColloct()
    timer.start("test")
    time.sleep(5)
    timer.end("test")
    timer.clloct()
