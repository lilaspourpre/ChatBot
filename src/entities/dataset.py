
class Dataset:
    def __init__(self, x,y):
        self.generator = [(x[i], y[i]) for i in range(len(x))]

    def get_generator(self):
        for x,y in self.generator:
            yield x,y