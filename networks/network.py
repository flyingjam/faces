from abc import ABCMeta, abstractmethod

class Network(metaclass=ABCMeta):

    @abstractmethod
    def train_epoch(self, epoch = 0):
        pass

    @abstractmethod
    def sample(self, img = None):
        pass

class Result():
    def __init__(self, duration, loss, epoch, loss_formatter = lambda x: "{}".format(x)):
        self.duration = duration
        self.loss = loss
        self.formatter = loss_formatter
        self.epoch = epoch

    def __str__(self):
        loss_string = self.formatter(self.loss)
        return "Epoch {} | Loss {} | Epoch Duration {}".format(self.epoch, loss_string, self.duration)


