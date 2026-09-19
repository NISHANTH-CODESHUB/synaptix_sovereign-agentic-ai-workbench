"""Egress monitor + canary.

This reports the TRUTH about the machine's connectivity. On a laptop with Wi-Fi
on it will say OPEN — which is correct, and is the point: a green badge means the
block was actually tested, not merely asserted.

Two independent signals:
  canary   — deliberately attempts an outbound TCP connect every few seconds and
             asserts it FAILS. A silent packet counter proves nothing; a failing
             canary proves the gap.
  sockets  — counts this process tree's non-loopback, non-LAN connections.
"""
from __future__ import annotations
import ipaddress
import socket
import threading
import time

import psutil

CANARY_TARGETS = [("1.1.1.1", 443), ("8.8.8.8", 53)]
LAN_NETS = [ipaddress.ip_network(n) for n in
            ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
             "169.254.0.0/16", "::1/128", "fe80::/10")]


def _is_local(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return True
    return any(ip in n for n in LAN_NETS)


class EgressMonitor:
    def __init__(self, period: float = 5.0):
        self.period = period
        self.state = {
            "status": "unknown",          # blocked | open | unknown
            "canary_last_check": None,
            "canary_detail": "",
            "external_connections": 0,
            "external_peers": [],
            "checks": 0,
            "blocked_attempts": 0,        # EgressBlocked raised by the app guard
            "since": time.time(),
        }
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()

    # -------------------------------------------------------------- canary
    def _canary(self) -> tuple[bool, str]:
        """Returns (blocked, detail). blocked=True means every attempt failed."""
        details = []
        blocked = True
        for host, port in CANARY_TARGETS:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.5)
                s.connect((host, port))
                s.close()
                blocked = False
                details.append(f"{host}:{port} REACHED")
            except Exception as e:
                details.append(f"{host}:{port} refused ({type(e).__name__})")
        return blocked, "; ".join(details)

    # -------------------------------------------------------------- sockets
    def _sockets(self) -> tuple[int, list[str]]:
        peers, me = [], psutil.Process()
        procs = [me] + me.children(recursive=True)
        pids = {p.pid for p in procs}
        try:
            for c in psutil.net_connections(kind="inet"):
                if c.pid in pids and c.raddr and not _is_local(c.raddr.ip):
                    peers.append(f"{c.raddr.ip}:{c.raddr.port}")
        except Exception:
            pass
        return len(peers), sorted(set(peers))[:20]

    def _loop(self):
        while not self._stop.wait(self.period):
            self.sample()

    def sample(self) -> dict:
        blocked, detail = self._canary()
        n, peers = self._sockets()
        self.state.update({
            "status": "blocked" if blocked else "open",
            "canary_last_check": time.strftime("%H:%M:%S"),
            "canary_detail": detail,
            "external_connections": n,
            "external_peers": peers,
            "checks": self.state["checks"] + 1,
        })
        return self.state

    def note_block(self, host: str):
        self.state["blocked_attempts"] += 1
        self.state["last_blocked_host"] = host


MONITOR = EgressMonitor()
