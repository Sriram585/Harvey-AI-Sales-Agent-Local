"""Network guard preventing external network calls during offline execution."""

import socket
from contextlib import ContextDecorator
from typing import Any

STRICT_OFFLINE_MODE: bool = False

ALLOWED_LOOPBACK_HOSTS: set[str] = {
    "127.0.0.1",
    "localhost",
    "::1",
    "0.0.0.0",
}


class ExternalNetworkBlockedError(RuntimeError):
    """Raised when any code attempts an external network connection while in strict offline mode."""
    pass


def _is_loopback(host: Any) -> bool:
    """Check if target host is an allowed loopback/local address."""
    if not host:
        return True
    host_str = str(host).strip().lower()
    if host_str in ALLOWED_LOOPBACK_HOSTS:
        return True
    if host_str.startswith("127."):
        return True
    return False


class StrictOfflineSandbox(ContextDecorator):
    """Context manager and decorator that blocks external network socket connections."""

    def __init__(self):
        self._orig_create_connection = None
        self._orig_socket_connect = None

    def __enter__(self):
        global STRICT_OFFLINE_MODE
        STRICT_OFFLINE_MODE = True
        self._orig_create_connection = socket.create_connection
        self._orig_socket_connect = socket.socket.connect

        def guarded_create_connection(address, *args, **kwargs):
            host = address[0] if isinstance(address, (tuple, list)) else address
            if not _is_loopback(host):
                raise ExternalNetworkBlockedError(
                    f"External network connection blocked in strict offline mode: {host}"
                )
            return self._orig_create_connection(address, *args, **kwargs)

        def guarded_socket_connect(sock_self, address):
            host = address[0] if isinstance(address, (tuple, list)) else address
            if not _is_loopback(host):
                raise ExternalNetworkBlockedError(
                    f"External network connection blocked in strict offline mode: {host}"
                )
            return self._orig_socket_connect(sock_self, address)

        socket.create_connection = guarded_create_connection
        socket.socket.connect = guarded_socket_connect
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global STRICT_OFFLINE_MODE
        STRICT_OFFLINE_MODE = False
        if self._orig_create_connection is not None:
            socket.create_connection = self._orig_create_connection
        if self._orig_socket_connect is not None:
            socket.socket.connect = self._orig_socket_connect
        return False
