"""用真实相机帧验证终训权重与模拟器 WebSocket 接口。"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from math import isclose, isfinite
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from drive_car_simulator import HOST, MAX_SPEED, DriverModel, drive_session


async def check_simulator(model: DriverModel, image: bytes) -> dict[str, float]:
    speed = 10.0
    predicted_steering = model.predict(image)["steering"]
    expected_steering = max(-1.0, min(1.0, predicted_steering))
    expected_throttle = max(0.0, 1.0 - expected_steering**2 - (speed / MAX_SPEED)**2)
    telemetry = {
        "image": base64.b64encode(image).decode("ascii"),
        "speed": str(speed),
        "steering_angle": "0",
        "throttle": "0",
    }

    async def handle(websocket):
        await drive_session(websocket, model, None)

    async with asyncio.timeout(15):
        async with serve(handle, HOST, 0, compression=None, ping_interval=None) as server:
            port = server.sockets[0].getsockname()[1]
            url = f"ws://{HOST}:{port}/socket.io/?EIO=4&transport=websocket"
            async with connect(url, proxy=None) as client:
                handshake = await client.recv()
                if not handshake.startswith("0") or not json.loads(handshake[1:])["sid"]:
                    raise ValueError("模拟器握手失败")
                await client.send("2")
                if await client.recv() != "3":
                    raise ValueError("模拟器心跳失败")
                await client.send("42" + json.dumps(["telemetry", telemetry]))
                packet = await client.recv()

    if not packet.startswith("42"):
        raise ValueError("未收到驾驶事件")
    event, data = json.loads(packet[2:])
    if event != "steer" or not all(isinstance(data[key], str) for key in ("steering_angle", "throttle")):
        raise ValueError("驾驶指令格式有误")
    steering, throttle = float(data["steering_angle"]), float(data["throttle"])
    if not (isfinite(steering) and -1 <= steering <= 1 and isclose(steering, expected_steering, abs_tol=1e-6)):
        raise ValueError(f"转向指令 {steering} 与模型预期 {expected_steering} 不符")
    if not (isfinite(throttle) and 0 <= throttle <= 1 and isclose(throttle, expected_throttle, abs_tol=1e-6)):
        raise ValueError(f"油门指令 {throttle} 与车速控制预期 {expected_throttle} 不符")
    return {"steering_angle": steering, "throttle": throttle}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("expected_epoch", type=int)
    args = parser.parse_args()
    try:
        model = DriverModel(args.checkpoint)
        if model.epoch != args.expected_epoch:
            raise ValueError(f"权重记录了 {model.epoch} 轮，预期 {args.expected_epoch} 轮")
        control = asyncio.run(check_simulator(model, args.image.read_bytes()))
    except (ImportError, OSError, TimeoutError, ValueError, KeyError) as exc:
        parser.exit(1, f"模拟器验证失败：{exc}\n")
    print(json.dumps({"epoch": model.epoch, **control}))


if __name__ == "__main__":
    main()
