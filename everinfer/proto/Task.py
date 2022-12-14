# automatically generated by the FlatBuffers compiler, do not modify

# namespace: 

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Task(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Task()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTask(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Task
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Task
    def Uuid(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # Task
    def UuidAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # Task
    def UuidLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Task
    def UuidIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # Task
    def Items(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .KVData import KVData
            obj = KVData()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Task
    def ItemsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Task
    def ItemsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def TaskStart(builder): builder.StartObject(2)
def Start(builder):
    return TaskStart(builder)
def TaskAddUuid(builder, uuid): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(uuid), 0)
def AddUuid(builder, uuid):
    return TaskAddUuid(builder, uuid)
def TaskStartUuidVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def StartUuidVector(builder, numElems):
    return TaskStartUuidVector(builder, numElems)
def TaskAddItems(builder, items): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(items), 0)
def AddItems(builder, items):
    return TaskAddItems(builder, items)
def TaskStartItemsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def StartItemsVector(builder, numElems):
    return TaskStartItemsVector(builder, numElems)
def TaskEnd(builder): return builder.EndObject()
def End(builder):
    return TaskEnd(builder)