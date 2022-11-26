# automatically generated by the FlatBuffers compiler, do not modify

# namespace: 

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Message(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Message()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsMessage(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Message
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Message
    def Auth(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = o + self._tab.Pos
            from Identity import Identity
            obj = Identity()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Message
    def DataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # Message
    def Data(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    # Message
    def Error(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from Error import Error
            obj = Error()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Message
    def Status(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

def MessageStart(builder): builder.StartObject(5)
def Start(builder):
    return MessageStart(builder)
def MessageAddAuth(builder, auth): builder.PrependStructSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(auth), 0)
def AddAuth(builder, auth):
    return MessageAddAuth(builder, auth)
def MessageAddDataType(builder, dataType): builder.PrependUint8Slot(1, dataType, 0)
def AddDataType(builder, dataType):
    return MessageAddDataType(builder, dataType)
def MessageAddData(builder, data): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(data), 0)
def AddData(builder, data):
    return MessageAddData(builder, data)
def MessageAddError(builder, error): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(error), 0)
def AddError(builder, error):
    return MessageAddError(builder, error)
def MessageAddStatus(builder, status): builder.PrependUint8Slot(4, status, 0)
def AddStatus(builder, status):
    return MessageAddStatus(builder, status)
def MessageEnd(builder): return builder.EndObject()
def End(builder):
    return MessageEnd(builder)