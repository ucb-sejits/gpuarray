import numpy as np
import pycl
import collections

class MappedArray(np.ndarray):

    queues = {}
    dirty = collections.defaultdict(bool)

    def __allocate_buffer(self, device):
        queue = self.__get_queue(device)
        self.__buffers[device], evt = pycl.buffer_from_ndarray(queue, self, self.__buffers.get(device))
        return evt

    @classmethod
    def __get_queue(cls, device):
        if device in cls.queues:
            return cls.queues[device]
        if device:
            ctx = pycl.clCreateContext(devices=[device])
        else:
            ctx = pycl.clCreateContextFromType(device_type=pycl.cl_device_type.CL_DEVICE_TYPE_GPU)
        queue = pycl.clCreateCommandQueue(context=ctx, device=device)
        cls.queues[device] = queue
        return queue


    def device_to_gpu(self, device=None, wait=True):
        evt = self.__allocate_buffer(device)
        if wait:
            evt.wait()
        else:
            self.__waiting.append(evt)

    def gpu_to_device(self, device=None, wait=True):
        _, evt = pycl.buffer_to_ndarray(self.__get_queue(device), self.__buffers[device])
        if wait:
            evt.wait()
        else:
            self.__waiting.append(evt)

    def get_buffer(self, device):
        return self.__buffers.get(device)

    def __array_finalize__(self, obj):
        self.__buffers = {}
        self.__waiting = []
        self.dirty['host'] = True

    def wait(self):
        for evt in self.__waiting:
            evt.wait()
        del self.__waiting[:]

    def __setitem__(self, key, value):
        self.dirty['host'] = True
        super(MappedArray, self).__setitem__(key, value)

    def __setslice__(self, i, j, sequence):
        self.dirty['host'] = True
        super(MappedArray, self).__setitem__(i, j, sequence)

    def __iadd__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__iadd__(other)

    def __isub__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__isub__(other)

    def __imul__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__imul__(other)

    def __idiv__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__idiv__(other)

    def __iand__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__iand__(other)

    def __ior__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__ior__(other)

    def __ipow__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__ipow__(other)

    def __ixor__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__ixor__(other)

    def __ifloordiv__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__ifloordiv__(other)

    def __itruediv__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__itruediv__(other)

    def __imod__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__imod__(other)

    def __ilshift__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__ilshift__(other)

    def __irshift__(self, other):
        self.dirty['host'] = True
        return super(MappedArray, self).__irshift__(other)

