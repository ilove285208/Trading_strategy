import sysconfig
from pathlib import Path


def _bootstrap_stdlib_signal():
    stdlib_signal_path = Path(sysconfig.get_path("stdlib")) / "signal.py"
    source = stdlib_signal_path.read_text(encoding="utf-8")
    globals()["__file__"] = str(stdlib_signal_path)
    globals()["__package__"] = ""
    exec(compile(source, str(stdlib_signal_path), "exec"), globals(), globals())


if __name__ == "signal":
    _bootstrap_stdlib_signal()
else:
    from signal_cli import main

    if __name__ == "__main__":
        main()
