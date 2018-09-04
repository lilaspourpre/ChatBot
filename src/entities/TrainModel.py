from keras.preprocessing.sequence import pad_sequences
import numpy as np


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

    def generator(self, dataset, batch_size, max_len, decode_size, save=True):
        samples_per_epoch = len(dataset)
        number_of_batches = int(samples_per_epoch / batch_size)
        counter = 0
        while 1:
            dataset_batch = np.array(dataset[batch_size * counter:batch_size * (counter + 1)])
            x_batch = pad_sequences([d[0] for d in dataset_batch], maxlen=max_len, padding="post")
            y_batch_input = pad_sequences([d[1] for d in dataset_batch],
                                          maxlen=max_len, padding="post")
            y_batch = pad_sequences([[self.decode(i, decode_size) for i in d[1]] for d in dataset_batch],
                                    maxlen=max_len, padding="post")
            counter += 1
            yield [x_batch, y_batch_input], y_batch
            # restart counter to yeild data in the next epoch as well
            if counter == number_of_batches:
                counter = 0
