import json
from datasets import Dataset
from unsloth.chat_templates import get_chat_template, standardize_sharegpt


class TrainingDatasetBuilder:
    def __init__(self, dataset, tokenizer, chat_template: str = "qwen3-instruct"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.chat_template = chat_template

        # Run the entire pipeline automatically
        self._run_pipeline()

    def _parse_conversations(self, example: dict) -> dict:
        """Ensure conversations field is parsed from JSON string if needed."""
        if isinstance(example.get("conversations"), str):
            example["conversations"] = json.loads(example["conversations"])
        return example

    def _formatting_prompts_func(self, examples):
        """Convert conversations into chat-formatted text."""
        convos = examples["conversations"]
        texts = [
            self.tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    def _strip_newline(self, example):
        """Remove trailing newlines from text."""
        example["text"] = example["text"].rstrip("\n")
        return example

    def _run_pipeline(self):
        """Execute the full dataset preparation pipeline."""
        # Step 1: Parse conversations
        self.dataset = self.dataset.map(self._parse_conversations)

        # Step 2: Standardize ShareGPT format
        self.dataset = standardize_sharegpt(self.dataset)

        # Step 3: Attach chat template to tokenizer
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=self.chat_template,
        )

        # Step 4: Apply formatting
        self.dataset = self.dataset.map(self._formatting_prompts_func, batched=True)

        # Step 5: Clean text
        self.dataset = self.dataset.map(self._strip_newline)

    def get_dataset(self):
        """Return the fully prepared dataset."""
        return self.dataset
