import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict
import httpx
from app.core.config import settings
from app.core.credentials import credentials_manager

logger = logging.getLogger(__name__)

WASM_WORKER_PATH = Path(__file__).parent.parent / "wasm" / "pow_worker.cjs"


class PoWSolver:
    def __init__(self, worker_path: Path = WASM_WORKER_PATH):
        self.worker_path = worker_path
        if not self.worker_path.exists():
            raise FileNotFoundError(f"WASM воркер не найден по пути {self.worker_path}")

    async def get_challenge(self, client: httpx.AsyncClient, target_path: str = "/api/v0/chat/completion") -> Dict[str, Any]:
        url = f"{settings.DEEPSEEK_BASE_URL}/api/v0/chat/create_pow_challenge"
        headers = {
            "accept": "*/*",
            "authorization": credentials_manager.auth_header,
            "content-type": "application/json",
            "x-client-bundle-id": settings.CLIENT_BUNDLE_ID,
            "x-client-locale": settings.CLIENT_LOCALE,
            "x-client-platform": settings.CLIENT_PLATFORM,
            "x-client-timezone-offset": settings.CLIENT_TIMEZONE_OFFSET,
            "x-client-version": settings.CLIENT_VERSION,
            "user-agent": settings.USER_AGENT,
        }

        response = await client.post(url, json={"target_path": target_path}, headers=headers)
        response.raise_for_status()
        
        result = response.json() or {}
        data = result.get("data") if isinstance(result, dict) else {}
        if data is None:
            data = {}
        biz_data = data.get("biz_data") if isinstance(data, dict) else {}
        if biz_data is None:
            biz_data = {}

        challenge = None
        if isinstance(biz_data, dict):
            challenge = biz_data.get("challenge") or (biz_data if "algorithm" in biz_data else None)

        if not isinstance(result, dict) or result.get("code") != 0 or not challenge:
            raise ValueError(f"Ошибка при получении PoW challenge: {result}")

        return challenge

    async def solve_challenge(self, challenge_data: Dict[str, Any]) -> str:
        algorithm = challenge_data["algorithm"]
        challenge = challenge_data["challenge"]
        salt = challenge_data["salt"]
        difficulty = challenge_data["difficulty"]
        expire_at = challenge_data.get("expire_at", 0)
        signature = challenge_data["signature"]
        target_path = challenge_data.get("target_path", "/api/v0/chat/completion")

        prefix = f"{salt}_{expire_at}_"

        proc = await asyncio.create_subprocess_exec(
            "node",
            str(self.worker_path),
            challenge,
            prefix,
            str(difficulty),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"Ошибка вычисления PoW в WASM воркере: {err_msg}")

        answer_str = stdout.decode().strip()
        answer = int(float(answer_str))

        if answer < 0:
            raise RuntimeError(f"Не удалось найти решение PoW для сложности {difficulty}")

        pow_response = {
            "algorithm": algorithm,
            "challenge": challenge,
            "salt": salt,
            "answer": answer,
            "signature": signature,
            "target_path": target_path,
        }

        json_bytes = json.dumps(pow_response, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(json_bytes).decode("utf-8")

    async def get_pow_header(self, client: httpx.AsyncClient, target_path: str = "/api/v0/chat/completion") -> str:
        challenge_data = await self.get_challenge(client, target_path)
        return await self.solve_challenge(challenge_data)


pow_solver = PoWSolver()
