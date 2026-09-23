import re


class Correctness:
    def __call__(self, completion, judgement):
        return float(judgement["correct"])


class Format:
    def __call__(self, completion, judgement):
        return float(
            bool(
                re.search(
                    r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>",
                    completion,
                    re.DOTALL,
                )
            )
        )


class Numeric:
    def __call__(self, completion, judgement):
        if "<answer>" not in completion:
            return 0.0
        answer = completion.split("<answer>")[-1].split("</answer>")[0]
        try:
            float(answer.replace(",", "").strip())
            return 1.0
        except ValueError:
            return 0.0
