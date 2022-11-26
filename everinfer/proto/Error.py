# automatically generated by the FlatBuffers compiler, do not modify

# namespace: 

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Error(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Error()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsError(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Error
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Error
    def Kind(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # Error
    def Cause(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def ErrorStart(builder): builder.StartObject(2)
def Start(builder):
    return ErrorStart(builder)
def ErrorAddKind(builder, kind): builder.PrependUint8Slot(0, kind, 0)
def AddKind(builder, kind):
    return ErrorAddKind(builder, kind)
def ErrorAddCause(builder, cause): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(cause), 0)
def AddCause(builder, cause):
    return ErrorAddCause(builder, cause)
def ErrorEnd(builder): return builder.EndObject()
def End(builder):
    return ErrorEnd(builder)