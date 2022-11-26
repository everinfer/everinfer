from typing import List, Dict, Union, Optional, Tuple
import time
import json
import uuid
import queue
import zenoh
import numpy as np
from PIL import Image
from everinfer.messages import (
    MSG_KEY,
    TaskMsg,
    AliveMsg,
    StatusMsg,
    NodeAcceptMsg,
    str_b64,
)

IntoTask = Dict[str, Union[str, Image.Image, np.ndarray]]


class Engine:
    def __init__(
        self,
        p_uuid: uuid.UUID,
        s_uuid: uuid.UUID,
        session: zenoh.Session,
        always_on: bool = True,
        task_timeout: float = 60.0,
        endpoint: Optional[Tuple[str, int]] = None,
    ):
        self.p_uuid = p_uuid
        self.s_uuid = s_uuid
        self.session = session
        self.always_on = always_on
        self.task_timeout = task_timeout
        self.endpoint = endpoint

        self._tasks = queue.SimpleQueue()
        self._preds = queue.SimpleQueue()

        p_uuid_key = p_uuid.bytes.hex()
        s_uuid_key = s_uuid.bytes.hex()

        self._live_qry = session.declare_queryable(
            f"source/alive/{s_uuid_key}", self._live_qry_cb
        )
        self._join_qry = session.declare_queryable(
            f"join/{s_uuid_key}/{p_uuid_key}", self._join_qry_cb
        )
        self._tasks_qry = session.declare_queryable(
            f"task/{s_uuid_key}/{p_uuid_key}/*", self._task_qry_cb
        )
        self._preds_sub = session.declare_subscriber(
            f"pred/{s_uuid_key}/{p_uuid_key}",
            self._task_sub_cb,
            reliability=zenoh.Reliability.RELIABLE(),
        )
        self._fail_sub = session.declare_subscriber(
            f"fail/{s_uuid_key}/{p_uuid_key}",
            self._task_sub_cb,
            reliability=zenoh.Reliability.RELIABLE(),
        )
        self._wake_pub = session.declare_publisher(
            f"wake/{s_uuid_key}/{p_uuid_key}",
            congestion_control=zenoh.CongestionControl.DROP(),
        )

        self.p_uuid_key = p_uuid_key
        self.s_uuid_key = s_uuid_key

        if self.endpoint is not None:
            config = self.session.config()
            listen = json.loads(config.get_json(zenoh.config.LISTEN_KEY))
            listen = list(set(listen + [f"tcp/0.0.0.0:{self.endpoint[1]}"]))
            config.insert_json5(zenoh.config.LISTEN_KEY, json.dumps(listen))
            self.endpoint = f"tcp/{self.endpoint[0]}:{self.endpoint[1]}"

    def _task_qry_cb(self, query: zenoh.Query):
        t_base_key = f"task/{self.s_uuid_key}/{self.p_uuid_key}"
        try:
            t_uuid, task = self._tasks.get(block=False)
            t_bytes = TaskMsg(t_uuid, task).serialize()
            query.reply(zenoh.Sample(f"{t_base_key}/{t_uuid.bytes.hex()}", t_bytes))
        except queue.Empty:
            if self.always_on:
                query.reply(zenoh.Sample(f"{t_base_key}/none", StatusMsg.later()))
        except Exception:
            query.reply(zenoh.Sample(f"{t_base_key}/none", StatusMsg.ok()))

    def _join_qry_cb(self, query: zenoh.Query):
        params = query.selector.decode_parameters()
        if MSG_KEY in params:
            data: NodeAcceptMsg = NodeAcceptMsg.deserialize(str_b64(params[MSG_KEY]))
            config = self.session.config()
            peers = json.loads(config.get_json(zenoh.config.CONNECT_KEY))
            peers = list(set(peers + [data.endpoint]))
            config.insert_json5(zenoh.config.CONNECT_KEY, json.dumps(peers))
        query.reply(zenoh.Sample(query.key_expr, StatusMsg.ok()))

    def _task_sub_cb(self, sample: zenoh.Sample):
        self._preds.put(sample.payload)

    def _live_qry_cb(self, query: zenoh.Query):
        reply_expr = f"source/alive/{self.s_uuid_key}"
        is_active = self.always_on or not self._tasks.empty()
        alive_msg = AliveMsg(
            self.s_uuid,
            self.p_uuid,
            is_active,
            self.endpoint,
        ).serialize()
        query.reply(zenoh.Sample(reply_expr, alive_msg))

    def predict(self, tasks: List[IntoTask]):
        self._reset()
        task_keys = [uuid.uuid4() for _ in tasks]
        for (key, task) in zip(task_keys, tasks):
            self._tasks.put((key, task))
        results = {}
        self._wake_pub.put(StatusMsg.ok())
        while set(task_keys) != set(results.keys()):
            try:
                task = self._preds.get(block=True, timeout=self.task_timeout)
                task: TaskMsg = TaskMsg.deserialize(task)
                results[task.uuid] = task
            except queue.Empty:
                break
        return [Engine._collect(results, task_key) for task_key in task_keys]

    def _collect(results: Dict[uuid.UUID, TaskMsg], task_key: uuid.UUID):
        if task_key in results:
            task = results.get(task_key)
            return task.data if task.data else {"error": task.error}
        else:
            return {"error": "timeout"}

    def _reset(self):
        while True:
            try:
                self._tasks.get(block=False)
            except queue.Empty:
                break
        while True:
            try:
                self._preds.get(block=False)
            except queue.Empty:
                break
