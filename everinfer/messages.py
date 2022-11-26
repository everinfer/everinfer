from typing import Dict, Union, Any, Optional
import json
import uuid
import base64
import urllib
import numpy as np
import flatbuffers
from PIL import Image as PILImage
from everinfer.proto import (
    Message,
    Payload,
    Pipeline,
    Task,
    Image,
    Tensor,
    TensorType,
    RawData,
    KVData,
    SourceAlive,
    NodeAccept,
    Identity,
    Status,
)

IntoTask = Dict[str, Union[str, Image.Image, np.ndarray]]

MSG_KEY = "msg"
BIN_KEY = "bin"


class TaskMsg:
    def __init__(self, t_uuid: Optional[uuid.UUID], data: IntoTask):
        self.data = data
        self.uuid = t_uuid if t_uuid else uuid.uuid4()
        self.error = None

    def __getitem__(self, key):
        if self.error:
            raise Exception(self.error)
        return self.data[key]

    def serialize(self) -> bytes:
        builder = flatbuffers.Builder(1024)
        task_items = []
        for name, item in self.data.items():
            task_items.append(TaskMsg._se_item(builder, name, item))

        p_uuid = builder.CreateByteVector(self.uuid.bytes)
        Task.StartItemsVector(builder, len(task_items))
        for item in task_items:
            builder.PrependUOffsetTRelative(item)
        p_task_items = builder.EndVector()

        Task.Start(builder)
        Task.AddUuid(builder, p_uuid)
        Task.AddItems(builder, p_task_items)
        p_task = Task.End(builder)

        Message.Start(builder)
        Message.AddDataType(builder, Payload.Payload().Task)
        Message.AddData(builder, p_task)
        p_msg = Message.End(builder)
        builder.Finish(p_msg)
        return bytes(builder.Output())

    @staticmethod
    def _se_item(
        builder: flatbuffers.Builder,
        name: str,
        data: Union[str, PILImage.Image, np.ndarray],
    ) -> Any:
        if isinstance(data, str):
            raw = np.fromfile(data, dtype=np.uint8).tobytes()
            return TaskMsg._se_image(builder, name, raw)
        elif isinstance(data, PILImage.Image):
            raw = data.tobytes()
            return TaskMsg._se_image(builder, name, raw)
        elif isinstance(data, np.ndarray):
            return TaskMsg._se_numpy(builder, name, data)
        else:
            raise Exception(f"Unknown task item type {type(data)}")

    @staticmethod
    def _se_image(builder: flatbuffers.Builder, name: str, raw: bytes) -> Any:
        p_name = builder.CreateString(name)
        p_data = builder.CreateByteVector(raw)

        Image.Start(builder)
        Image.AddData(builder, p_data)
        p_image = Image.End(builder)

        KVData.Start(builder)
        KVData.AddName(builder, p_name)
        KVData.AddItemType(builder, RawData.RawData().Image)
        KVData.AddItem(builder, p_image)
        return KVData.End(builder)

    @staticmethod
    def _se_numpy(builder: flatbuffers.Builder, name: str, data: np.ndarray) -> Any:
        p_name = builder.CreateString(name)
        p_data = builder.CreateByteVector(data.tobytes())
        p_dims = builder.CreateNumpyVector(np.array(data.shape, dtype=np.int64))

        Tensor.Start(builder)
        Tensor.AddKind(builder, _fb_type(data.dtype))
        Tensor.AddDims(builder, p_dims)
        Tensor.AddData(builder, p_data)
        p_tensor = Tensor.End(builder)

        KVData.Start(builder)
        KVData.AddName(builder, p_name)
        KVData.AddItemType(builder, RawData.RawData().Tensor)
        KVData.AddItem(builder, p_tensor)
        return KVData.End(builder)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        msg = Message.Message.GetRootAs(data)
        if msg.DataType() != Payload.Payload().Task:
            raise Exception(f"Expected `Task` got payload type `{msg.DataType()}`")

        p_task = Task.Task()
        p_task.Init(msg.Data().Bytes, msg.Data().Pos)
        task_uuid = uuid.UUID(bytes=p_task.UuidAsNumpy().tobytes())

        if msg.Error():
            result = TaskMsg(task_uuid, dict())
            result.error = msg.Error().Cause().decode("utf-8")
            return result

        items = {}
        for i in range(p_task.ItemsLength()):
            p_item = p_task.Items(i)
            if p_item.ItemType() != RawData.RawData().Tensor:
                continue
            p_tensor = Tensor.Tensor()
            p_tensor.Init(p_item.Item().Bytes, p_item.Item().Pos)
            name = p_item.Name().decode("utf-8")
            data = np.frombuffer(
                p_tensor.DataAsNumpy().tobytes(), dtype=_np_type(p_tensor.Kind())
            )
            items[name] = data.reshape(p_tensor.DimsAsNumpy())
        return TaskMsg(task_uuid, items)


def _fb_type(t: np.dtype) -> TensorType:
    if t == np.dtype("float64"):
        return TensorType.TensorType().Double
    elif t == np.dtype("uint64"):
        return TensorType.TensorType().UInt64
    elif t == np.dtype("int64"):
        return TensorType.TensorType().Int64
    elif t == np.dtype("float32"):
        return TensorType.TensorType().Float
    elif t == np.dtype("uint32"):
        return TensorType.TensorType().UInt32
    elif t == np.dtype("int32"):
        return TensorType.TensorType().Int32
    elif t == np.dtype("float16"):
        return TensorType.TensorType().Float16
    elif t == np.dtype("uint16"):
        return TensorType.TensorType().UInt16
    elif t == np.dtype("int16"):
        return TensorType.TensorType().Int16
    elif t == np.dtype("bool"):
        return TensorType.TensorType().Bool
    elif t == np.dtype("uint8"):
        return TensorType.TensorType().UInt8
    elif t == np.dtype("int8"):
        return TensorType.TensorType().Int8
    else:
        return TensorType.TensorType().Undefined


def _np_type(t: TensorType) -> np.dtype:
    if t == TensorType.TensorType().Double:
        return np.dtype("float64")
    elif t == TensorType.TensorType().UInt64:
        return np.dtype("uint64")
    elif t == TensorType.TensorType().Int64:
        return np.dtype("int64")
    elif t == TensorType.TensorType().Float:
        return np.dtype("float32")
    elif t == TensorType.TensorType().UInt32:
        return np.dtype("uint32")
    elif t == TensorType.TensorType().Int32:
        return np.dtype("int32")
    elif t == TensorType.TensorType().Float16:
        return np.dtype("float16")
    elif t == TensorType.TensorType().UInt16:
        return np.dtype("uint16")
    elif t == TensorType.TensorType().Int16:
        return np.dtype("int16")
    elif t == TensorType.TensorType().Bool:
        return np.dtype("bool")
    elif t == TensorType.TensorType().UInt8:
        return np.dtype("uint8")
    elif t == TensorType.TensorType().Int8:
        return np.dtype("int8")
    else:
        raise Exception(f"Undefined tensor type {t}")


class AliveMsg:
    def __init__(
        self,
        s_uuid: uuid.UUID,
        p_uuid: uuid.UUID,
        active: bool,
        endpoint: Optional[str],
    ):
        self.s_uuid = s_uuid
        self.p_uuid = p_uuid
        self.active = active
        self.endpoint = endpoint

    def serialize(self) -> bytes:
        builder = flatbuffers.Builder(1024)

        p_p_uuid = builder.CreateByteVector(self.p_uuid.bytes)
        p_endpoint = builder.CreateString(self.endpoint) if self.endpoint else None

        SourceAlive.Start(builder)
        SourceAlive.AddActive(builder, self.active)
        SourceAlive.AddPipelineUuid(builder, p_p_uuid)
        if p_endpoint is not None:
            SourceAlive.AddEndpoint(builder, p_endpoint)
        p_alive = SourceAlive.End(builder)

        Message.Start(builder)
        Message.AddAuth(builder, IdentityMsg(self.s_uuid).build(builder))
        Message.AddDataType(builder, Payload.Payload().SourceAlive)
        Message.AddData(builder, p_alive)
        p_msg = Message.End(builder)
        builder.Finish(p_msg)
        return bytes(builder.Output())


class StatusMsg:
    @staticmethod
    def serialize(status: Status) -> bytes:
        builder = flatbuffers.Builder(1024)

        Message.Start(builder)
        Message.AddStatus(builder, status)
        p_msg = Message.End(builder)

        builder.Finish(p_msg)
        return bytes(builder.Output())

    @staticmethod
    def deserialize(data: bytes) -> Status:
        msg = Message.Message.GetRootAs(data)
        return msg.Status()

    @staticmethod
    def ok() -> bytes:
        return StatusMsg.serialize(Status.Status().Ok)

    @staticmethod
    def later() -> bytes:
        return StatusMsg.serialize(Status.Status().Later)


class NodeAcceptMsg:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @staticmethod
    def deserialize(data: bytes) -> Any:
        msg = Message.Message.GetRootAs(data)
        if msg.DataType() != Payload.Payload().NodeAccept:
            raise Exception(
                f"Expected `NodeAccept` got payload type `{msg.DataType()}`"
            )
        p_accept = NodeAccept.NodeAccept()
        p_accept.Init(msg.Data().Bytes, msg.Data().Pos)
        return NodeAcceptMsg(p_accept.Endpoint().decode("utf-8"))


class PipelineMsg:
    def __init__(
        self,
        n_stage: int,
        name: str,
        meta: Optional[Dict[str, Any]],
        s_uuid: Optional[uuid.UUID] = None,
        p_uuid: Optional[uuid.UUID] = None,
    ):
        self.name = name
        self.meta = meta
        self.s_uuid = s_uuid
        self.p_uuid = p_uuid
        self.n_stage = n_stage

    def serialize(self) -> bytes:
        builder = flatbuffers.Builder(1024)
        p_name = builder.CreateString(self.name)
        p_meta = (
            builder.CreateByteVector(json.dumps(self.meta).encode())
            if self.meta
            else None
        )
        p_p_uuid = builder.CreateByteVector(self.p_uuid.bytes) if self.p_uuid else None

        Pipeline.Start(builder)
        Pipeline.AddName(builder, p_name)
        Pipeline.AddNStage(builder, self.n_stage)
        if p_p_uuid:
            Pipeline.AddUuid(builder, p_p_uuid)
        if p_meta:
            Pipeline.AddMeta(builder, p_meta)
        p_pipeline = Pipeline.End(builder)

        Message.Start(builder)
        Message.AddAuth(builder, IdentityMsg(self.s_uuid).build(builder))
        Message.AddDataType(builder, Payload.Payload().Pipeline)
        Message.AddData(builder, p_pipeline)
        p_msg = Message.End(builder)

        builder.Finish(p_msg)
        return bytes(builder.Output())

    @staticmethod
    def deserialize(data: bytes) -> Any:
        msg = Message.Message.GetRootAs(data)
        if msg.DataType() != Payload.Payload().Pipeline:
            raise Exception(f"Expected `Pipeline` got payload type `{msg.DataType()}`")

        p_meta = Pipeline.Pipeline()
        p_meta.Init(msg.Data().Bytes, msg.Data().Pos)

        p_uuid = uuid.UUID(bytes=p_meta.UuidAsNumpy().tobytes())
        p_name = p_meta.Name().decode("utf-8") if p_meta.Name() else None
        p_n_stage = p_meta.NStage()
        p_pipe_meta = (
            json.loads(p_meta.MetaAsNumpy().tobytes())
            if not p_meta.MetaIsNone()
            else None
        )
        return PipelineMsg(p_n_stage, p_name, p_pipe_meta, p_uuid=p_uuid)


class IdentityMsg:
    def __init__(
        self,
        uuid: Optional[uuid.UUID] = None,
        pubkey: Optional[bytes] = None,
        data_hash: Optional[bytes] = None,
        signature: Optional[bytes] = None,
    ):
        self.uuid = uuid
        self.pubkey = pubkey
        self.data_hash = data_hash
        self.signature = signature

    def build(self, builder: flatbuffers.Builder) -> Any:
        uuid = (
            self.uuid.bytes if self.uuid else np.zeros((16), dtype=np.uint8).tobytes()
        )
        pubkey = (
            self.pubkey if self.pubkey else np.zeros((32), dtype=np.uint8).tobytes()
        )
        data_hash = (
            self.data_hash
            if self.data_hash
            else np.zeros((64), dtype=np.uint8).tobytes()
        )
        signature = (
            self.signature
            if self.signature
            else np.zeros((64), dtype=np.uint8).tobytes()
        )
        return Identity.CreateIdentity(builder, uuid, pubkey, data_hash, signature)


def b64_str(b: bytes, url_quote=True) -> str:
    b64_string = base64.b64encode(b).decode("ascii")
    if url_quote:
        return urllib.parse.quote(b64_string)
    return b64_string


def str_b64(s: str, url_quote=True) -> bytes:
    if url_quote:
        return base64.b64decode(urllib.parse.unquote(s))
    return base64.b64decode(s)
