from types import SimpleNamespace

import torch
from _stubs import StubTokenizer
from datasets import Dataset

from src.pubmedqa_loraplus import evaluation


class BatchTokenizer(StubTokenizer):
    def batch_decode(self, rows, skip_special_tokens=False):
        return [
            self.decode(
                [token for token in row.tolist() if token != self.pad_token_id],
                skip_special_tokens=skip_special_tokens,
            )
            for row in rows
        ]


class ProseFirstModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.prose_token_id = tokenizer(" The")["input_ids"][0]
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.generation_batch_sizes = []
        self.scoring_batch_sizes = []

    @staticmethod
    def rationale(prompt):
        if "Unmarked question" in prompt:
            return "The phrase Answer: appears inside the explanation."
        return "The evidence supports the third option."

    def generate(self, input_ids, attention_mask, **kwargs):
        assert kwargs["stop_strings"] == ["\nAnswer:"]
        assert kwargs["max_new_tokens"] == 24
        assert kwargs["do_sample"] is False
        self.generation_batch_sizes.append(len(input_ids))
        continuations = []
        for row, mask in zip(input_ids, attention_mask):
            prompt = self.tokenizer.decode(row[mask.bool()].tolist())
            assert "Reason briefly" in prompt
            assert "Answer: <letter>" in prompt
            text = " " + self.rationale(prompt)
            if "Unmarked question" not in prompt:
                text += "\nAnswer: C"
            continuations.append(self.tokenizer(text)["input_ids"])
        width = max(map(len, continuations))
        padded = [ids + [self.tokenizer.pad_token_id] * (width - len(ids)) for ids in continuations]
        return torch.cat((input_ids, torch.tensor(padded)), dim=1)

    def forward(self, input_ids, attention_mask, position_ids, logits_to_keep):
        self.scoring_batch_sizes.append(len(input_ids))
        logits = torch.zeros(len(input_ids), 1, len(self.tokenizer.vocab))
        for index, (row, mask) in enumerate(zip(input_ids, attention_mask)):
            text = self.tokenizer.decode(row[mask.bool()].tolist())
            prompt, _, completion = text.partition("[/INST]")
            if completion == f" {self.rationale(prompt)}\nAnswer:":
                logits[index, 0, self.tokenizer.vocab[" C"]] = 5
            else:
                logits[index, 0, self.tokenizer.vocab[" A"]] = 5
                logits[index, 0, self.prose_token_id] = 10
        return SimpleNamespace(logits=logits)


def test_mmlu_scores_prose_first_answers_and_counts_missing_markers(monkeypatch):
    dataset = Dataset.from_dict(
        {
            "question": ["Marked question", "Unmarked question with details", "Another question"],
            "choices": [["one", "two", "three", "four"]] * 3,
            "subject": ["anatomy", "clinical_knowledge", "medical_genetics"],
            "answer": [2, 2, 0],
        }
    )
    monkeypatch.setattr(evaluation, "load_dataset", lambda *args, **kwargs: dataset)
    tokenizer = BatchTokenizer()
    model = ProseFirstModel(tokenizer)
    config = SimpleNamespace(
        mmlu_dataset_id="stub",
        mmlu_dataset_config="all",
        mmlu_split="test",
        mmlu_revision=None,
        mmlu_eval_samples=12,
        chat_template_kwargs={},
        eval_batch_size=2,
        rationale_max_new_tokens=24,
    )

    result = evaluation.evaluate_mmlu_medical(model, tokenizer, config)

    assert result["overall_accuracy"] == 1 / 3
    assert result["answer_line_accuracy"] == 2 / 3
    assert result["answer_line_unmarked"] == 1
    assert result["total_samples"] == 3
    assert result["subject_accuracies"]["anatomy"] == 0.0
    assert result["subject_results"]["anatomy"] == {"correct": 0, "total": 1, "answer_line": 1}
    assert result["subject_results"]["clinical_knowledge"] == {
        "correct": 0,
        "total": 1,
        "answer_line": 1,
    }
    assert result["subject_results"]["medical_genetics"] == {
        "correct": 1,
        "total": 1,
        "answer_line": 0,
    }
    assert model.generation_batch_sizes == [2, 1]
    assert model.scoring_batch_sizes == [2, 2, 1, 1]
