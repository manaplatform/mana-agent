"""Read-only storage inspection commands."""

STORAGE_INSPECTION_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("df", "-hP"),
    ("df", "-iP"),
    ("findmnt", "--json"),
    ("lsblk", "--json", "--output", "NAME,SIZE,FSTYPE,MOUNTPOINTS,TYPE"),
)
