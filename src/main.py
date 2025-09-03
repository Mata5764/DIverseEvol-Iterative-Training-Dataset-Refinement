import os
import yaml
import logging
from datasets import load_dataset, Dataset
from utils import sample_data, hf_dataset_to_json
from train import TrainModel
from formatter import TrainingDatasetBuilder
from sampling import DiversitySampler

# -------------------- CONFIG & PATHS -------------------- #
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def get_path(key):
    """Helper function to build absolute paths from config"""
    return os.path.join(BASE_DIR, config.get(key))

os.makedirs(get_path("selected_dir"), exist_ok=True)
os.makedirs(get_path("entire_dir"), exist_ok=True)
os.makedirs(get_path("checkpoints_dir"), exist_ok=True)

# Create logs directory and setup logging
logs_dir = os.path.join(BASE_DIR, "models", "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging to file and console
log_file = os.path.join(logs_dir, "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)




# -------------------- LOAD VALIDATION DATA -------------------- #
validation_dataset = load_dataset(
    "myfi/parser_dataset_ner_val_v1.9",
    split="train",
    token="HF_TOKEN_PLACEHOLDER"
)
logging.info(f"Validation dataset size: {len(validation_dataset)}")

# -------------------- LOAD FULL TRAINING DATA -------------------- #
training_dataset = load_dataset(
    config.get("hf_dataset_name", "myfi/parser_dataset_ner_v1.16"),
    split=config.get("hf_dataset_split", "train"),
    token=config.get("HF_TOKEN_PLACEHOLDER")
)
logging.info(f"Training dataset size: {len(training_dataset)}")
logging.info(f"Starting DiverseEvol pipeline with {config.get('rounds', 3)} rounds")

# -------------------- ITERATIVE TRAINING PIPELINE -------------------- #
cumulative_training_samples = Dataset.from_list([])

for round_num in range(config.get("rounds", 3)):
    logging.info(f"===== ROUND {round_num} =====")

    if round_num == 0:
        # Randomly sample INIT_LABEL_NUM examples
        sampled_dataset = sample_data(training_dataset, config.get("init_label_num", 100))
    else:
        # Select new samples using best model from previous round
        prev_round_dir = os.path.join(get_path("checkpoints_dir"), f"round_{round_num-1}")
        checkpoint_dir = [d for d in os.listdir(prev_round_dir) if d.startswith("checkpoint-")][0]
        best_model_dir = os.path.join(prev_round_dir, checkpoint_dir)
        sampler = DiversitySampler(model_dir=best_model_dir, config=config)

        candidate_dataset = training_dataset
        sampled_dataset = sampler.select_diverse_samples(
            current_dataset=cumulative_training_samples,
            candidate_dataset=candidate_dataset,
            k=config.get("diverse_k", 100),
            text_field="conversations"
        )
        
        # Clean up previous checkpoint after successful sampling
        import shutil
        shutil.rmtree(prev_round_dir)
        logging.info(f"Deleted previous checkpoint: {prev_round_dir}")

    # Update cumulative dataset (always HF Dataset)
    cumulative_training_samples = Dataset.from_list(
        list(cumulative_training_samples) + list(sampled_dataset)
    )

    # Save selected & cumulative datasets
    round_selected_file = os.path.join(get_path("selected_dir"), f"round_{round_num}.json")
    hf_dataset_to_json(sampled_dataset, round_selected_file)

    round_entire_file = os.path.join(get_path("entire_dir"), f"round_{round_num}.json")
    hf_dataset_to_json(cumulative_training_samples, round_entire_file)

    logging.info(f"Training dataset size after round {round_num}: {len(cumulative_training_samples)}")

    train_model = TrainModel(config.get("base_model", "unsloth/Qwen3-4B-Instruct-2507"), config)

    # Format datasets
    train_builder = TrainingDatasetBuilder(
        dataset=cumulative_training_samples,
        tokenizer=train_model.tokenizer,
        chat_template=config.get("chat_template", "qwen3-instruct")
    )
    formatted_train_dataset = train_builder.get_dataset()

    val_builder = TrainingDatasetBuilder(
        dataset=validation_dataset,
        tokenizer=train_model.tokenizer,
        chat_template=config.get("chat_template", "qwen3-instruct")
    )
    formatted_val_dataset = val_builder.get_dataset()

    # Train model from scratch each round, with round-specific checkpoints
    round_checkpoint_dir = os.path.join(get_path("checkpoints_dir"), f"round_{round_num}")
    os.makedirs(round_checkpoint_dir, exist_ok=True)

    
    training_stats = train_model.train(
        training_dataset=formatted_train_dataset,
        validation_dataset=formatted_val_dataset,
        output_dir=round_checkpoint_dir
    )

    # Extract and log the best eval loss
    if hasattr(training_stats, 'metrics') and 'eval_loss' in training_stats.metrics:
        best_eval_loss = training_stats.metrics['eval_loss']
        logging.info(f"Round {round_num} - Best eval loss: {best_eval_loss:.4f}")
    
    logging.info(f"Completed training round {round_num}")

logging.info("🎉 DiverseEvol pipeline completed successfully!")
logging.info(f"Final training dataset size: {len(cumulative_training_samples)}")
logging.info(f"All logs saved to: {log_file}")
