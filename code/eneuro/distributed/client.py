"""One-request TCP client for the local parameter server."""

import socket
from .protocol import send_message, recv_message


class PSClient:
    def __init__(self, host, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout
        self.connection = None

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def request(self, message, arrays=None):
        try:
            if self.connection is None:
                self.connection = socket.create_connection((self.host, self.port), timeout=self.timeout)
                self.connection.settimeout(self.timeout)
            sent = send_message(self.connection, message, arrays)
            reply, values, received = recv_message(self.connection)
        except (OSError, ValueError, ConnectionError):
            self.close()
            raise
        if reply.get('type') == 'ERROR':
            raise RuntimeError(reply.get('message', 'parameter server error'))
        return reply, values, sent, received
