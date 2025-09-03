from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
from unsloth.chat_templates import train_on_responses_only


# Load environment variables from .env file
load_dotenv()


class TrainModel:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=config.get("max_seq_length", 2048),
            dtype=config.get("dtype", None),
            load_in_4bit=config.get("load_in_4bit", True),
        )

        # Wrap with LoRA/PEFT
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=config.get("lora_r", 16),
            target_modules=config.get("target_modules", ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config.get("seed", 3407),
        )

    def train(self, training_dataset, validation_dataset, output_dir):
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.get("max_seq_length", 2048),
            packing=False,
            args=TrainingArguments(
                num_train_epochs=self.config.get("num_train_epochs", 3),
                per_device_train_batch_size=self.config.get("per_device_train_batch_size", 2),
                gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),
                warmup_steps=self.config.get("warmup_steps", 10),
                learning_rate=self.config.get("lr", 2e-5),
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=self.config.get("logging_steps", 1),
                optim=self.config.get("optim", "adamw_torch"),
                weight_decay=self.config.get("weight_decay", 0.01),
                lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine_with_restarts"),
                seed=self.config.get("seed", 42),
                output_dir=output_dir,
                report_to=self.config.get("report_to", "none"),
                eval_strategy=self.config.get("eval_strategy", "steps"),
                eval_steps=self.config.get("eval_steps", 10),
                save_strategy=self.config.get("save_strategy", "steps"),
                save_steps=self.config.get("save_steps", 10),
                save_only_model=self.config.get("save_only_model", True),
                save_total_limit=self.config.get("save_total_limit", 1),
                load_best_model_at_end=self.config.get("load_best_model_at_end", True),
                metric_for_best_model=self.config.get("metric_for_best_model", "eval_loss"),
                greater_is_better=self.config.get("greater_is_better", False),
            ),
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
            )
        
        stats = trainer.train()
        return stats