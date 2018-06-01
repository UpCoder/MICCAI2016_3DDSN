class Generator:
    def __init__(self, generated_data, batch_size, epoch_num=-1, resize=None):
        self.generated_data = generated_data
        self.total_size = len(self.generated_data[0])
        for i in range(len(self.generated_data)):
            if self.total_size != len(self.generated_data[i]):
                assert False
        self.batch_size = int(batch_size)
        self.epoch_num = epoch_num
        self.start = 0
        self.resize = resize

    def next_batch(self):
        if self.epoch_num == -1:
            while True:
                batch_data = []
                # print self.batch_size
                end = self.start + self.batch_size
                if end > self.total_size:
                    end = self.total_size
                for idx in range(len(self.generated_data)):
                    batch_data.append(self.generated_data[idx][self.start:end])
                self.start += self.batch_size
                if end >= self.total_size:
                    self.start = 0
                # print np.shape(image_batch), np.shape(keras.utils.to_categorical(label_batch, 10))
                yield batch_data
        else:
            while True:
                batch_data = []
                end = self.start + self.batch_size
                if end > self.total_size:
                    end = self.total_size
                for idx in range(len(self.generated_data)):
                    batch_data.append(self.generated_data[idx][self.start:end])
                self.start += self.batch_size
                if end >= self.total_size:
                    break
                yield batch_data
            yield None, None, None