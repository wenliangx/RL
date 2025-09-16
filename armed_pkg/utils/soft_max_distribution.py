import numpy as np

class SoftMaxDistribution:

    def __init__(self, length, initial=None):
        self.length = length
        if initial is None:
            self._datas = np.zeros(length)
        else:
            self._datas = initial
        self._exp_datas = np.exp(self._datas)
        self._sum = np.sum(self._exp_datas)
        self._distribution = self._exp_datas / self._sum

    @property
    def distribution(self):
        # self._sum = np.sum(np.exp(self._datas))
        # self._distribution = np.exp(self._datas) / self._sum
        return self._distribution

    @property
    def datas(self):
        return self._datas
    
    @datas.setter
    def datas(self, values):
        self._datas = np.array(values)
        self._exp_datas = np.exp(self._datas)
        self._sum = np.sum(self._exp_datas)
        self._distribution = self._exp_datas / self._sum
    
    def set_data(self, index, value):
        self._datas[index] = value
        self._exp_datas = np.exp(self._datas)
        self._sum = np.sum(self._exp_datas)
        self._distribution = self._exp_datas / self._sum