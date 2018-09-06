from keras.callbacks import Callback
import os


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

    def save_model(self):
        pass

    def save_json(self, target_directory, model_name, model):
        with open(os.path.join(target_directory, "{}_model.json".format(model_name)), "w") as j:
            j.write(model.to_json())


class WeightsSaver(Callback):
    def __init__(self, target_path, model_name):
        super(WeightsSaver, self).__init__()
        self.save_path = os.path.join(target_path, model_name+"_models")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def on_epoch_end(self, epoch, logs=None):
        name = 'weights_on_epoch_{}.h5'.format(epoch)
        self.model.save_weights(os.path.join(self.save_path, name))
