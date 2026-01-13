def val_class(level: str) -> str:
    return {"ok": "val-ok", "moderate": "val-mod", "severe": "val-sev"}[level]


def initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][0].upper()
    return (parts[0][0] + parts[-1][0]).upper()
