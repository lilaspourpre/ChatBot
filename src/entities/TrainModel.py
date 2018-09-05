
class TrainModel:
    def __init__(self):
        pass

    def train_model(self, dataset, max_len, decode_size, batch_size, epochs):
        pass

    @staticmethod
    def decode(num, decode_size):
        result = [0] * decode_size
        result[num] = 1
        return result
