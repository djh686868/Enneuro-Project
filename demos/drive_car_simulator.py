"""用 EnNeuro 终训模型为 Udacity Windows 模拟器提供驾驶指令。"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "car_beyesian_full_20261110_005640" / "best_checkpoint.bin"
REFIT_DIR = ROOT / "artifacts" / "car_simulator_refit_100"
REFIT_MODEL = REFIT_DIR / "checkpoint.bin"
REFIT_VERIFIED = REFIT_DIR / "simulator_verified.json"
HOST, PORT = "127.0.0.1", 4567
MAX_SPEED, MIN_SPEED = 25.0, 10.0
LOGGER = logging.getLogger(__name__)


class DriverModel:
    def __init__(self, checkpoint: Path):
        sys.path.insert(0, str(ROOT / "code"))
        from eneuro.agent.car_executor import preprocess_camera_image
        from eneuro.base import Config, Tensor
        from eneuro.nn.resnet18_regression import ResNet18Regressor
        from eneuro.utils.serializer import Serializer

        config_file = checkpoint.with_suffix(".config.json")
        config = json.loads(config_file.read_text(encoding="utf-8"))["model_config"]
        if config["type"] != "ResNet18Regressor" or "steering" not in config["target_names"]:
            raise ValueError("需要包含转向预测的 ResNet18Regressor 权重")
        self.image_size = tuple(config["image_size"])
        self.target_names = config["target_names"]
        normalization = config["target_normalization"]
        self.means = [normalization[name]["mean"] for name in self.target_names]
        self.scales = [normalization[name]["scale"] for name in self.target_names]
        self.preprocess = preprocess_camera_image
        self.config = Config
        self.tensor = Tensor
        self.model = ResNet18Regressor(out_features=config["out_features"])
        self.epoch = Serializer.load_checkpoint_binary(checkpoint, self.model, SimpleNamespace())

    def predict(self, image: bytes) -> dict[str, float]:
        frame = self.preprocess(image, self.image_size)
        with self.config.no_grad(), self.config.test_mode():
            output = self.model(self.tensor(frame[None, ...])).data[0]
        values = {
            name: float(value) * scale + mean
            for name, value, scale, mean in zip(
                self.target_names, output, self.scales, self.means, strict=True
            )
        }
        if not all(map(isfinite, values.values())):
            raise ValueError("模型输出包含非有限值")
        return values


def event_packet(name: str, data: dict) -> str:
    return "42" + json.dumps([name, data], separators=(",", ":"))


def steer_packet(steering: float, throttle: float) -> str:
    return event_packet("steer", {"steering_angle": str(steering), "throttle": str(throttle)})


async def drive_session(websocket, model: DriverModel, image_folder: Path | None):
    from websockets.exceptions import ConnectionClosed

    url = urlsplit(websocket.request.path)
    if url.path != "/socket.io/" or parse_qs(url.query).get("transport") != ["websocket"]:
        await websocket.close(code=1008, reason="Expected simulator Socket.IO WebSocket")
        return

    sid = uuid4().hex
    speed_limit = MAX_SPEED
    frame_number = 0
    LOGGER.info("模拟器已连接")
    try:
        # 此 Unity 客户端的 URL 使用 EIO=4，但心跳由客户端发起，且直接发送默认命名空间的事件。
        handshake = {"sid": sid, "upgrades": [], "pingInterval": 25000, "pingTimeout": 20000}
        await websocket.send("0" + json.dumps(handshake, separators=(",", ":")))
        async for packet in websocket:
            if packet == "2":
                await websocket.send("3")
            elif isinstance(packet, str) and packet.startswith("42"):
                try:
                    event, *args = json.loads(packet[2:])
                    if event != "telemetry":
                        continue
                    if not args or not args[0]:
                        await websocket.send(event_packet("manual", {}))
                        continue
                    data = args[0]
                    speed = float(data["speed"])
                    if not isfinite(speed):
                        raise ValueError("模拟器车速不是有限值")
                    image = base64.b64decode(data["image"], validate=True)
                    steering = max(-1.0, min(1.0, model.predict(image)["steering"]))
                    speed_limit = MIN_SPEED if speed > speed_limit else MAX_SPEED
                    throttle = max(0.0, 1.0 - steering**2 - (speed / speed_limit)**2)
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    LOGGER.warning("当前帧无法用于推理，油门归零：%s", exc)
                    await websocket.send(steer_packet(0.0, 0.0))
                    continue

                frame_number += 1
                await websocket.send(steer_packet(steering, throttle))
                if image_folder is not None:
                    (image_folder / f"{sid}_{frame_number:06d}.jpg").write_bytes(image)
                if frame_number == 1 or frame_number % 100 == 0:
                    LOGGER.info("第 %d 帧，转向 %.3f，油门 %.3f，车速 %.1f mph",
                                frame_number, steering, throttle, speed)
    except ConnectionClosed:
        pass
    finally:
        LOGGER.info("模拟器已断开，共处理 %d 帧", frame_number)


async def serve_simulator(model: DriverModel, image_folder: Path | None):
    from websockets.asyncio.server import serve

    async def handle(websocket):
        await drive_session(websocket, model, image_folder)

    async with serve(handle, HOST, PORT, compression=None, ping_interval=None) as server:
        LOGGER.info("驾驶服务已启动，监听 %s:%d。请在模拟器中选择 Autonomous Mode。", HOST, PORT)
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?",
                        help="本项目生成的终训权重 .bin，需与同名 .config.json 放在一起")
    parser.add_argument("image_folder", type=Path, nargs="?", help="可选的摄像头图像保存目录")
    parser.add_argument("--check-image", type=Path, help="对一张图片推理并打印四项预测")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("websockets.server").setLevel(logging.WARNING)
    try:
        checkpoint = args.model or (REFIT_MODEL if REFIT_VERIFIED.is_file() else DEFAULT_MODEL)
        model = DriverModel(checkpoint)
        LOGGER.info("已加载 %d 轮终训权重 %s", model.epoch, checkpoint)
        if args.check_image:
            print(json.dumps(model.predict(args.check_image.read_bytes()), indent=2))
            return
        if args.image_folder is not None:
            args.image_folder.mkdir(parents=True, exist_ok=True)
        asyncio.run(serve_simulator(model, args.image_folder))
    except KeyboardInterrupt:
        pass
    except (ImportError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"启动失败：{exc}\n")


if __name__ == "__main__":
    main()
