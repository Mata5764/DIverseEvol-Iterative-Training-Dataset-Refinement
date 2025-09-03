import os
import torch
import numpy as np
from torch.nn.functional import normalize
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
import json


class DiversitySampler:
    def __init__(self, model_dir, config=None, device=None):
        """
        Initialize with the saved model checkpoint.

        Args:
            model_dir (str): Path to the saved model directory.
            config (dict, optional): Configuration dictionary.
            device (str, optional): Device to use ("cuda" or "cpu").
        """
        self.config = config or {}
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=self.config.get("max_seq_length", 2048),
            dtype=self.config.get("dtype", None),
            load_in_4bit=self.config.get("load_in_4bit", True),
        )
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _mean_pool_embeddings(self, texts, batch_size=None):
        """
        Compute mean pooled embeddings for a list of texts.

        Args:
            texts (list[str]): List of input sentences.
            batch_size (int): Batch size for embedding computation.

        Returns:
            np.ndarray: Array of embeddings.
        """
        batch_size = batch_size or self.config.get("embedding_batch_size", 8)
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.model(**tokens, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

                # Masked mean pooling
                mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden_states.size())
                summed = torch.sum(hidden_states * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = summed / counts

                mean_pooled = normalize(mean_pooled, p=2, dim=1)  # L2 normalize

            all_embeddings.append(mean_pooled.cpu().numpy())

        return np.vstack(all_embeddings)

    def select_diverse_samples(self, current_dataset, candidate_dataset, k=100, text_field="conversations"):
        """
        Select k most diverse samples from candidates.

        Args:
            current_dataset (Dataset): Current dataset (HF Dataset or list of dicts).
            candidate_dataset (Dataset): Candidate dataset (HF Dataset or list of dicts).
            k (int): Number of samples to select.
            text_field (str): Field containing text.

        Returns:
            list[dict]: Selected samples.
        """
        # Step 1: Extract human text from conversations
        current_texts = self._extract_human_texts(current_dataset[text_field])
        candidate_texts = self._extract_human_texts(candidate_dataset[text_field])

        # Step 2: Embed both datasets
        current_embeds = self._mean_pool_embeddings(current_texts)
        candidate_embeds = self._mean_pool_embeddings(candidate_texts)

        # Step 3: Compute min distance to existing training set
        dists = []
        for i, cand in enumerate(candidate_embeds):
            # Compute distances to all current embeddings
            dist = np.linalg.norm(current_embeds - cand, axis=1)
            min_dist = dist.min()
            dists.append((i, min_dist))

        # Step 4: Pick top-K with largest min_dist
        dists.sort(key=lambda x: x[1], reverse=True)  # largest min_dist first
        selected_indices = [idx for idx, _ in dists[:k]]

        # Step 5: Return HuggingFace Dataset subset
        selected_samples = [candidate_dataset[i] for i in selected_indices]

        return selected_samples
    
    def _extract_human_texts(self, conversations_list):
        """Extract human message text from conversations JSON strings."""
        human_texts = []
        for conv_str in conversations_list:
            try:
                conversations = json.loads(conv_str)
                # Find the human message
                for msg in conversations:
                    if msg.get("from") == "human":
                        human_texts.append(msg.get("value", ""))
                        break
                else:
                    # No human message found, use empty string
                    human_texts.append("")
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, use empty string
                human_texts.append("")
        return human_texts

    def save_selected(self, selected_samples, output_path):
        """
        Save selected samples to JSON file.

        Args:
            selected_samples (list[dict]): Samples to save.
            output_path (str): Path to save JSON.
        """
        import json
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(selected_samples, f, indent=2, ensure_ascii=False)


