"""Tokenizer stubs with real chat-template shapes, for CPU-only tests."""

import re

MISTRAL_V3 = ("<s>[INST] {user}[/INST]", " {assistant}</s>")
CHATML = (
    "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
    "{assistant}<|im_end|>\n",
)
SPECIAL = re.compile(r"<s>|</s>|\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>")


class StubTokenizer:
    """Word-piece-like tokenizer: a leading space merges into the next word.

    That mirrors SentencePiece/BPE closely enough that ``" yes"`` is one token,
    which is what makes the decision-token position meaningful.
    """

    eos_token = "</s>"
    pad_token_id = 0

    def __init__(self, template=MISTRAL_V3):
        self.prompt_template, self.reply_template = template
        self.vocab: dict[str, int] = {"<pad>": 0}
        self.inverse: dict[int, str] = {0: "<pad>"}

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab)
            self.inverse[self.vocab[piece]] = piece
        return self.vocab[piece]

    def __call__(self, text, add_special_tokens=False):
        pieces = []
        for part in re.split(f"({SPECIAL.pattern})", text):
            if SPECIAL.fullmatch(part or ""):
                pieces.append(part)
            elif part:
                pieces.extend(re.findall(r" ?\w+| ?[^\w\s]|\s", part))
        return {"input_ids": [self._id(piece) for piece in pieces]}

    def decode(self, ids, skip_special_tokens=False):
        pieces = [self.inverse[i] for i in ids]
        if skip_special_tokens:
            pieces = [piece for piece in pieces if not SPECIAL.fullmatch(piece)]
        return "".join(pieces)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        text = self.prompt_template.format(user=messages[0]["content"])
        if len(messages) > 1:
            text += self.reply_template.format(assistant=messages[1]["content"])
        return text
