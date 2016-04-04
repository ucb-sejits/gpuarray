import numpy as np
import pycl


class MappedArray(np.ndarray):

    queues = {}

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

    def wait(self):
        for evt in self.__waiting:
            evt.wait()
        del self.__waiting[:]
