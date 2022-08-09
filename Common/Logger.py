class AverageMeter:
    def __init__(self, name=None):
        self.value = 0
        self.count = 0

        self.name = name

    def __call__(self, value, n=1):
        self.value += value
        self.count += n

    def result(self):
        return round(self.value / max(1, self.count), 4)

    def reset(self):
        self.value = 0
        self.count = 0

class SumMeter:
    def __init__(self, name=None):
        self.value = 0
        self.count = 0

        self.name = name

    def __call__(self, value, n=1):
        self.value += value
        self.count += n

    def result(self):
        return round(self.value, 4)

    def reset(self):
        self.value = 0
        self.count = 0


class ListMeter:
    def __init__(self, name=None):
        self.counts = []

        self.name = name

    def __call__(self, step):
        self.counts.append(step)

    def result(self):
        return round(self.counts[-1], 4)

    def reset(self):
        self.counts = []

class ValueMeter:
    def __init__(self, name=None):
        self.value = 0
        self.name = name

    def __call__(self, value):
        self.value = value

    def result(self):
        return round(self.value, 3)

    def reset(self):
        self.value = 0


