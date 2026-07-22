"""Tests for buffered simulator JSON decoding."""

from __future__ import annotations

import socket
import sys
import threading
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from communication.simulator_client import SimulatorClient  # noqa: E402


class SimulatorClientTests(unittest.TestCase):
    def test_receive_json_handles_fragmented_tcp_payload(self) -> None:
        client_socket, server_socket = socket.socketpair()
        client = SimulatorClient("unused", 0)
        client._socket = client_socket  # Inject a local socket for an isolated test.

        def sender() -> None:
            for chunk in (b'{"Img":"abc",', b'"Speed":42}'):
                server_socket.sendall(chunk)
            server_socket.close()

        thread = threading.Thread(target=sender)
        thread.start()
        try:
            payload = client.receive_json()
        finally:
            client.close()
            thread.join()

        self.assertEqual(payload, {"Img": "abc", "Speed": 42})

    def test_receive_json_keeps_second_message_in_buffer(self) -> None:
        client_socket, server_socket = socket.socketpair()
        client = SimulatorClient("unused", 0)
        client._socket = client_socket
        server_socket.sendall(b'{"frame":1}{"frame":2}')

        try:
            self.assertEqual(client.receive_json(), {"frame": 1})
            self.assertEqual(client.receive_json(), {"frame": 2})
        finally:
            client.close()
            server_socket.close()


if __name__ == "__main__":
    unittest.main()
