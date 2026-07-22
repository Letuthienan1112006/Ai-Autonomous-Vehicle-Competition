"""TCP client for the Unity simulator JSON stream."""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, Optional


class SimulatorConnectionError(RuntimeError):
    """Raised when the simulator connection closes or returns invalid data."""


class SimulatorClient:
    """Exchange steering commands and JSON camera frames with the simulator.

    The original project assumed that one ``recv`` call always contained one
    complete JSON object. TCP does not guarantee message boundaries, so this
    implementation keeps a text buffer and decodes one complete JSON object
    at a time.
    """

    def __init__(self, host: str, port: int, timeout: Optional[float] = None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._buffer = ""
        self._decoder = json.JSONDecoder()

    def connect(self) -> None:
        if self._socket is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.timeout is not None:
            sock.settimeout(self.timeout)
        try:
            sock.connect((self.host, self.port))
        except OSError as exc:
            sock.close()
            raise SimulatorConnectionError(
                f"Could not connect to simulator at {self.host}:{self.port}."
            ) from exc
        self._socket = sock

    def send_control(self, steering_angle: float, speed: float) -> None:
        if self._socket is None:
            raise SimulatorConnectionError("Simulator client is not connected.")
        payload = f"{steering_angle} {speed}".encode("utf-8")
        self._socket.sendall(payload)

    def receive_json(self) -> Dict[str, Any]:
        if self._socket is None:
            raise SimulatorConnectionError("Simulator client is not connected.")

        while True:
            stripped = self._buffer.lstrip()
            leading_ws = len(self._buffer) - len(stripped)
            if stripped:
                try:
                    value, end = self._decoder.raw_decode(stripped)
                except json.JSONDecodeError:
                    pass
                else:
                    self._buffer = self._buffer[leading_ws + end :]
                    if not isinstance(value, dict):
                        raise SimulatorConnectionError(
                            f"Expected a JSON object, received {type(value).__name__}."
                        )
                    return value

            chunk = self._socket.recv(65536)
            if not chunk:
                raise SimulatorConnectionError("Simulator closed the connection.")
            try:
                self._buffer += chunk.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SimulatorConnectionError(
                    "Simulator returned data that is not valid UTF-8 JSON."
                ) from exc

    def exchange_control(self, steering_angle: float, speed: float) -> Dict[str, Any]:
        self.send_control(steering_angle, speed)
        return self.receive_json()

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None

    def __enter__(self) -> "SimulatorClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
