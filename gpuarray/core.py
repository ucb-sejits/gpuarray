import numpy as np
import pycl
import collections
import ctree


def get_gpu():
    try:
        name = None
        gpu_id = None
        if ctree.CONFIG.has_option("opencl", "gpu"):
            name = ctree.CONFIG.get("opencl", "gpu")
        if ctree.CONFIG.has_option("opencl", "gpu_id"):
            gpu_id = ctree.CONFIG.get("opencl", "gpu_id")
        if name is None:
            return pycl.clGetDeviceIDs(device_type=pycl.CL_DEVICE_TYPE_GPU)[0]
        else:
            for gpu in pycl.clGetDeviceIDs():
                if gpu.name == name:
                    return gpu
                if gpu.value == gpu_id:
                    return gpu
    except pycl.DeviceNotFoundError:
        return None


class MappedArray(np.ndarray):

    queues = {}
    existing = {}


    @staticmethod
    def hash_array(arr):
        return arr.__array_interface__['data']

    def __allocate_buffer(self, device=get_gpu()):
        queue = self.get_queue(device)
        self.__buffers[device.value], evt = pycl.buffer_from_ndarray(queue, self, self.get_buffer(device))
        return evt

    def __is_dirty(self, device=get_gpu()):
        if hasattr(device, "value"):
            return self.dirty[device.value]
        return self.dirty[device]

    def __copied(self, device=get_gpu()):
        if hasattr(device, "value"):
            return self.copied[device.value]
        return self.copied[device]

    def __set_copied(self, device=get_gpu(), val=True):
        if hasattr(device, "value"):
            self.copied[device.value] = val
        else:
            self.copied[device] = val

    def set_dirty(self, device=get_gpu(), dirty=True):
        if hasattr(device, "value"):
            self.dirty[device.value] = dirty
        else:
            self.dirty[device] = dirty

    @classmethod
    def get_queue(cls, device=get_gpu()):
        if device.value in cls.queues:
            return cls.queues[device.value]
        else:
            ctx = pycl.clCreateContext(devices=[device])
        queue = pycl.clCreateCommandQueue(context=ctx, device=device)
        cls.queues[device.value] = queue
        return queue

    def device_to_gpu(self, device=get_gpu(), wait=True, force=False):
        if self.__copied(device) and not self.__is_dirty(device) and not force:
            return
        print("DEVICE TO GPU")
        evt = self.__allocate_buffer(device)
        if wait:
            evt.wait()
        else:
            self.__waiting.append(evt)

        self.set_dirty(device, False)
        self.__set_copied(device, True)


    def gpu_to_device(self, device=get_gpu(), wait=True, force=False):
        if not self.__is_dirty("host") and not force:
            return
        print("GPU to DEVICE")
        _, evt = pycl.buffer_to_ndarray(self.get_queue(device), self.__buffers[device.value], out=self)
        if wait:
            evt.wait()
        else:
            self.__waiting.append(evt)

        self.set_dirty('host', False)

    def get_buffer(self, device=get_gpu()):
        return self.__buffers.get(device.value)

    def __array_finalize__(self, obj):
        if self.hash_array(obj) in self.existing:
            old = self.existing[self.hash_array(obj)]
            self.dirty = old.dirty
            self.copied = old.copied
            self.__buffers = old.__buffers
            self.__waiting = old.__waiting
            return

        elif not hasattr(self, "dirty"):
            self.__buffers = {}
            self.__waiting = []
            self.dirty = collections.defaultdict(lambda: True)
            self.copied = collections.defaultdict(bool)
            self.dirty['host'] = True
            self.copied['host'] = True
            self.existing[self.hash_array(obj)] = self

    def wait(self):
        for evt in self.__waiting:
            evt.wait()
        del self.__waiting[:]

    def __setitem__(self, key, value):
        self.dirty['host'] = True
        super(MappedArray, self).__setitem__(key, value)

    def __setslice__(self, i, j, sequence):
        self.dirty['host'] = True
        super(MappedArray, self).__setslice__(i, j, sequence)

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

    def __getitem__(self, item):
        output = super(MappedArray, self).__getitem__(item)
        if isinstance(output, MappedArray):
            output.dirty = self.dirty
        return output

    def __getslice__(self, i, j):
        output = super(MappedArray, self).__getslice__(i, j)
        if isinstance(output, MappedArray):
            output.dirty = self.dirty
        return output