from pathlib import Path
import numpy as np
import re

try:
    import openvino as ov
except Exception:
    ov = None


BASE = Path(__file__).resolve().parent.parent / "models"


class InferenceEngine:

    def __init__(self):
        self.compiled = None
        self.mode = "heuristic-fallback"

        if ov:
            model_path = BASE / "semantic_classifier.xml"

            if model_path.exists():
                try:
                    core = ov.Core()
                    model = core.read_model(str(model_path))
                    self.compiled = core.compile_model(model, "CPU")
                    self.mode = "openvino-cpu"
                except Exception as exc:
                    print(f"OpenVINO initialization failed: {exc}")

    def classify(self, text: str) -> str:
        text = text.strip()

        # -------------------------------------------------
        # Deterministic list detection
        # -------------------------------------------------
        # Supports:
        # 1. Python
        # 2. Java
        # 3. C++
        #
        # - Python
        # * Java
        # • C++
        # -------------------------------------------------

        if re.match(r"^\s*(?:\d+[\.\)]|[-*•])\s+\S+", text):
            return "list"

        # -------------------------------------------------
        # OpenVINO neural classification
        # -------------------------------------------------

        if self.compiled:
            x = self.features(text).reshape(1, -1).astype(np.float32)

            result = self.compiled([x])[self.compiled.output(0)]

            labels = [
                "paragraph",
                "heading",
                "list",
            ]

            return labels[int(np.argmax(result))]

        # -------------------------------------------------
        # Heuristic fallback
        # -------------------------------------------------

        words = text.split()

        if (
            len(words) <= 12
            and not text.endswith((".", "?", "!"))
            and re.search(r"[A-Z]", text)
        ):
            return "heading"

        return "paragraph"

    @staticmethod
    def features(text: str):
        words = text.split()
        chars = len(text)

        return np.array(
            [
                len(words),
                chars,
                sum(c.isupper() for c in text),
                text.count(","),
                text.count("."),
                text.count(":"),
            ],
            dtype=np.float32,
        )

    def status(self):
        return {
            "mode": self.mode,
            "device": "CPU" if self.compiled else "fallback",
            "model": "semantic_classifier" if self.compiled else None,
        }


_engine = InferenceEngine()


def get_inference_engine():
    return _engine