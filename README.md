# Mistral-7B Fine-tuning for Medical Question Answering

This project fine-tunes the Mistral-7B-Instruct model on the PubMedQA dataset for medical question answering using Parameter Efficient Fine-Tuning (PEFT) with LoRA adapters.

## 🎯 Project Overview

- **Base Model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Dataset**: PubMedQA (Biomedical Question Answering)
- **Method**: LoRA (Low-Rank Adaptation)
- **Task**: Medical Question Answering (Yes/No/Maybe classification)

## 📊 Model Details

### Tokenizer Configuration
- **Vocabulary Size**: 32,768 tokens
- **Special Tokens**: 1,024 control tokens ([control_0] to [control_1023])
- **Padding Token**: Set to EOS token
- **Max Length**: 512 tokens

### LoRA Configuration
```json
{
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ]
}
```

### Training Parameters
- **Trainable Parameters**: 41,943,040 (0.58% of total)
- **Total Parameters**: 7,289,966,592
- **Training Regime**: bfloat16 mixed precision
- **Optimizer**: AdamW
- **Learning Rate**: 2e-4
- **Batch Size**: 4 per device
- **Gradient Accumulation**: 4 steps
- **Epochs**: 3

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Using the Notebook
Open and run [`mistral-7b-instruct-finetune.ipynb`](mistral-7b-instruct-finetune.ipynb) for step-by-step fine-tuning.

### Using the Script
```bash
python main.py
```

### Loading the Fine-tuned Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer = AutoTokenizer.from_pretrained("./mistral-pubmedqa-finetune")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./mistral-pubmedqa-adapter")

# Example usage
context = "Regular exercise has been shown to reduce cardiovascular disease risk."
question = "Does exercise reduce heart disease risk?"
prompt = f"<s>[INST] Context: {context}\n\nQuestion: {question} [/INST]"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📈 Performance

### PubMedQA Results
- **Accuracy**: 73.0%
- **F1 Score**: 70.23%

### MMLU Medical Tasks
- **Overall Accuracy**: 56.0%
- **Anatomy**: 56.0%

## 📁 File Structure

```
├── main.py                           # Main training script
├── mistral-7b-instruct-finetune.ipynb # Jupyter notebook
├── mistral-pubmedqa-finetune/        # Fine-tuned model
│   ├── adapter_config.json
│   ├── tokenizer_config.json
│   └── ...
├── mistral-pubmedqa-adapter/         # LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── evaluation_results.json          # Evaluation metrics
├── pubmedqa_confusion_matrix.png    # Confusion matrix visualization
└── requirements.txt                 # Dependencies
```

## 🔬 Technical Implementation

The fine-tuning process involves:

1. **Data Preprocessing**: Format PubMedQA examples into Mistral's instruction format
2. **LoRA Application**: Apply low-rank adapters to attention and MLP layers
3. **Training**: Use custom data collator for causal language modeling
4. **Evaluation**: Test on both PubMedQA and MMLU medical tasks

### Prompt Format
```
<s>[INST] Context: {medical_context}

Question: {question} [/INST] {answer}</s>
```

## 🛠️ Dependencies

- transformers >= 4.0.0
- datasets >= 4.0.0
- peft >= 0.11.1
- torch
- numpy
- scikit-learn
- tqdm

## 📝 Citation

```bibtex
@misc{mistral-pubmedqa-2024,
  title={Fine-tuning Mistral-7B for Medical Question Answering},
  author={Aber Islam},
  year={2024},
  note={Fine-tuned on PubMedQA dataset using LoRA}
}
```
