"""Encode NumPy scalars from the frozen analyzer as ordinary JSON numbers."""
import json

import numpy as np


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def write(path, data):
    path.write_text(json.dumps(data,default=scalar,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")

