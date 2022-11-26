from typing import List, Dict, Any, Optional, Union, Tuple
import uuid
import json
import requests
import zenoh
from everinfer.engine import Engine


class Client:
    def __init__(self, api_key: str, gateway: str = "135.181.212.173"):
        self.rest_api = f"http://{gateway}:3000"
        self.rest_client = requests.Session()
        self.rest_client.headers.update({"Authorization": f"Bearer {api_key}"})
        self.authenticate()

        proto_api = [f"tcp/{gateway}:50051"]
        config = zenoh.Config()
        config.insert_json5(zenoh.config.MODE_KEY, json.dumps("peer"))
        config.insert_json5(zenoh.config.CONNECT_KEY, json.dumps(proto_api))
        session = zenoh.open(config)
        self.session = session

    def authenticate(self):
        response = self.rest_client.get(f"{self.rest_api}/auth").json()
        if "error" in response:
            raise Exception(response["error"])
        self.identity = response
        self.uuid = uuid.UUID(hex=response["uuid"])

    def register_pipeline(
        self,
        name: str,
        stages: List[str],
        meta: Optional[Dict[str, Any]] = None,
        uuid: Optional[uuid.UUID] = None,
    ):
        payload = {
            "uuid": uuid,
            "name": name,
            "n_stage": len(stages),
            "meta": meta,
        }
        response = self.rest_client.post(
            f"{self.rest_api}/pipeline", json=payload
        ).json()
        p_stages = []
        for stage in stages:
            p_stages.append(("stage", open(stage, "rb")))
        self.rest_client.post(
            f"{self.rest_api}/pipeline/{response['uuid']}", files=p_stages
        ).json()
        return response

    def create_engine(
        self,
        pipeline_uuid: Union[uuid.UUID, str],
        always_on: bool = True,
        task_timeout: float = 60,
        endpoint: Optional[Tuple[str, int]] = None,
    ) -> Engine:
        if isinstance(pipeline_uuid, str):
            pipeline_uuid = uuid.UUID(hex=pipeline_uuid)
        payload = {
            "pipeline_uuid": pipeline_uuid.hex,
            "endpoint": f"tcp/{self.endpoint[0]}:{self.endpoint[1]}"
            if endpoint
            else None,
        }
        alive = self.rest_client.post(f"{self.rest_api}/source", json=payload).json()
        if "status" not in alive:
            raise Exception(alive)
        return Engine(
            pipeline_uuid, self.uuid, self.session, always_on, task_timeout, endpoint
        )
