"""Read-only network inspection commands."""

NETWORK_INSPECTION_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("ip", "address", "show"),
    ("ip", "route", "show"),
    ("ss", "-lntup"),
    ("cat", "/etc/resolv.conf"),
)


def port_check_argv(host: str, port: int) -> list[str]:
    if not host or not 1 <= port <= 65535:
        raise ValueError("An exact host and valid port are required.")
    return ["nc", "-zvw5", host, str(port)]
