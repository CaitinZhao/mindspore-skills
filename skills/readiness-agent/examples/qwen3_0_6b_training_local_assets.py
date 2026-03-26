from pathlib import Path

from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


WORKSPACE = Path(__file__).resolve().parents[2]
ASSET_ROOT = WORKSPACE / ".readiness-assets"
MODEL_PATH = ASSET_ROOT / "models" / "Qwen__Qwen3-0.6B"
DATASET_PATH = ASSET_ROOT / "datasets" / "karthiksagarn__astro_horoscope"

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
dataset = load_from_disk(str(DATASET_PATH))
if isinstance(dataset, DatasetDict):
    train_source = dataset["train"]
else:
    train_source = dataset


def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )


tokenized = train_source.map(tokenize, batched=True, remove_columns=train_source.column_names)
tokenized = tokenized.train_test_split(test_size=0.1)

model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), torch_dtype="auto")
model = model.to("npu")

training_args = TrainingArguments(
    output_dir=str(WORKSPACE / "qwen3-finetuned"),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
