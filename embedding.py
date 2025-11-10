"""
BERT embedding module for text encoding and semantic representation.
Provides tokenization and embedding generation for incident analysis.
"""

import json
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path

import torch
import numpy as np
from transformers import BertTokenizer, BertModel

logger = logging.getLogger(__name__)


class RealBertEmbedder:
    
    def __init__(self, model_name: str = None, model_dir: str = None):
        """
        Initialize BERT embedder with local or remote model loading.
        
        Args:
            model_name: HuggingFace model name (default: bert-base-uncased)
            model_dir: Local directory containing BERT model files
        """
        if model_dir is None:
            model_dir = "D:/incident_management/models/bert"
        
        if model_name is None:
            model_name = "bert-base-uncased"
        
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Try loading from local directory first
            if self._check_local_files():
                logger.info(f"Loading BERT from local directory: {model_dir}")
                self._load_local_model()
            else:
                # Fall back to downloading from HuggingFace
                logger.warning(f"Local BERT model not found in {model_dir}")
                logger.info(f"Downloading {model_name} from HuggingFace Hub...")
                self._download_and_load_model()
            
            self.model.eval()
            self.hidden_size = self.model.config.hidden_size
            logger.info(f"BERT loaded successfully (hidden_size: {self.hidden_size})")
            
        except Exception as e:
            logger.error(f"Failed to load BERT: {str(e)}")
            logger.error("Full error:", exc_info=True)
            raise
    
    def _check_local_files(self) -> bool:
        """
        Check if all required BERT model files exist locally.
        
        Returns:
            True if all files exist, False otherwise
        """
        required_files = ["config.json", "vocab.txt"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        
        # Check required files
        for file in required_files:
            if not (self.model_dir / file).exists():
                return False
        
        # Check at least one model file exists
        has_model_file = any((self.model_dir / f).exists() for f in model_files)
        
        return has_model_file
    
    def _load_local_model(self) -> None:
        """Load BERT model from local directory."""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                str(self.model_dir),
                local_files_only=True
            )
            
            self.model = BertModel.from_pretrained(
                str(self.model_dir),
                local_files_only=True
            ).to(self.device)
            
            logger.info("Successfully loaded BERT from local directory")
            
        except Exception as e:
            logger.error(f"Failed to load local BERT model: {str(e)}")
            raise
    
    def _download_and_load_model(self) -> None:
        """Download BERT model from HuggingFace and save locally."""
        try:
            # Ensure directory exists
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Download tokenizer
            logger.info("Downloading tokenizer...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            
            # Download model
            logger.info("Downloading model (this may take a few minutes)...")
            self.model = BertModel.from_pretrained(self.model_name).to(self.device)
            
            # Save to local directory
            logger.info(f"Saving model to {self.model_dir}...")
            self.tokenizer.save_pretrained(str(self.model_dir))
            self.model.save_pretrained(str(self.model_dir))
            
            logger.info("Model downloaded and saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to download BERT model: {str(e)}")
            raise
    
    def encode(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Encode text into BERT embedding vector.
        
        Args:
            text: Input text to encode
            max_length: Maximum sequence length (default: 512)
            
        Returns:
            Mean-pooled embedding vector as numpy array
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text input must be a non-empty string")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            masked_embeddings = last_hidden_state * attention_mask_expanded
            summed_embeddings = masked_embeddings.sum(dim=1)
            token_counts = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed_embeddings / token_counts
            embedding_vector = mean_pooled.squeeze(0).cpu().numpy()
            
            return embedding_vector
            
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise


# Alias for backward compatibility
BertEmbedder = RealBertEmbedder