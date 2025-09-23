
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from unidecode import unidecode
from difflib import get_close_matches
import warnings

warnings.filterwarnings("ignore")

# Model directory
MODEL_DIR = os.environ.get("MODEL_DIR", "model/prod")

# File paths
TORCH_MODEL_FILE = os.path.join(MODEL_DIR, "torch_regression_model.pkl")
PRENOM_AGE_ENCODER_FILE = os.path.join(MODEL_DIR, "prenom_age_encoder.pkl")
SCALER_FEATURES_FILE = os.path.join(MODEL_DIR, "scaler_features.pkl")
SCALER_TARGET_FILE = os.path.join(MODEL_DIR, "scaler_target.pkl")


class TorchRegression(nn.Module):
    """PyTorch regression model matching the trained architecture"""
    def __init__(self, input_size: int):
        super(TorchRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -------------------------
# Module aliasing for safe torch.load() in newer PyTorch versions
# This is crucial for loading the saved PyTorch model
# -------------------------
import sys

# Make TorchRegression available in __main__ namespace for pickle deserialization
sys.modules['__main__'].TorchRegression = TorchRegression

# Also make it available in current module
_current_module = sys.modules[__name__]
setattr(_current_module, "TorchRegression", TorchRegression)


# For newer torch versions, add to safe globals
try:
    import torch.serialization as _ts
    if hasattr(_ts, "add_safe_globals"):
        _ts.add_safe_globals({"TorchRegression": TorchRegression})
except Exception as e:
    print(f"[WARN] Could not register TorchRegression in torch safe globals: {e}")


def normalize_prename(prename: str) -> str:
    """Normalize prename to match training data format"""
    normalized_prename = unidecode(prename)
    normalized_prename = normalized_prename.upper()
    return normalized_prename


class AgeService:
    """Age prediction service using the working pipeline from script_test_age_pred_torch.py"""
    
    def __init__(self):
        self._load_model_and_artifacts()
        
    def _load_model_and_artifacts(self):
        """Load all required model files and artifacts"""
        try:
            # Load PyTorch model with weights_only=False to handle custom classes
            self.model_torch = torch.load(TORCH_MODEL_FILE, map_location='cpu', weights_only=False)
            self.model_torch.eval()
            
            # Load encoders and scalers
            with open(PRENOM_AGE_ENCODER_FILE, "rb") as file:
                self.prenom_age_encoder = pickle.load(file)
                
            with open(SCALER_FEATURES_FILE, "rb") as f:
                self.scaler_features = pickle.load(f)
                
            with open(SCALER_TARGET_FILE, "rb") as f:
                self.scaler_target = pickle.load(f)
                
            # Get valid prenoms for suggestions
            self.valid_prenom = list(self.prenom_age_encoder.keys())
            
            print("[MODEL] Successfully loaded all artifacts")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model artifacts: {e}")
            raise
    
    def _get_suggestions(self, prenom: str) -> List[Dict[str, Any]]:
        """Get name suggestions for invalid names"""
        suggestions = get_close_matches(prenom, self.valid_prenom, n=5, cutoff=0.7)
        return [{"prenom": suggestion, "available": True} for suggestion in suggestions]
    
    def predict_age(self, prenom: str) -> Dict[str, Any]:
        """
        Predict age for a given first name using the working pipeline
        
        Args:
            prenom: First name to predict age for
            
        Returns:
            Dict containing prediction results
        """
        if not prenom or not prenom.strip():
            return {
                "name": prenom,
                "age": None,
                "error": "Empty name provided"
            }
        
        # Normalize the name
        normalized_prenom = normalize_prename(prenom.strip())
        
        # Check if name exists in encoder
        if normalized_prenom not in self.valid_prenom:
            suggestions = self._get_suggestions(normalized_prenom)
            return {
                "name": prenom,
                "age": None,
                "error": "Name not found in training data",
                "suggestions": suggestions
            }
        
        try:
            # Get encoded value for the name
            preusuel_encoded = self.prenom_age_encoder.get(normalized_prenom)
            
            if preusuel_encoded is None:
                return {
                    "name": prenom,
                    "age": None,
                    "error": "Failed to encode name"
                }
            
            # Prepare input data (reshape to column vector)
            input_data_encoded = np.array(preusuel_encoded).reshape(-1, 1)
            
            # Scale the features
            input_data_encoded_scaled = self.scaler_features.transform(input_data_encoded)
            
            # Convert to tensor
            input_tensor = torch.tensor(input_data_encoded_scaled, dtype=torch.float32)
            
            # Model prediction
            with torch.no_grad():
                predicted_age_scaled = self.model_torch(input_tensor).numpy()
            
            # Inverse transform to get actual age
            predicted_age = self.scaler_target.inverse_transform(predicted_age_scaled)[0][0]
            
            # Clamp to reasonable range
            predicted_age = max(0.0, min(120.0, float(predicted_age)))
            
            return {
                "name": prenom,
                "age": round(predicted_age, 1),
                "normalized_name": normalized_prenom
            }
            
        except Exception as e:
            return {
                "name": prenom,
                "age": None,
                "error": f"Prediction failed: {str(e)}"
            }
            
    def predict(self, name: str) -> Dict[str, Any]:
        """Wrapper for predict_age to match service interface"""
        return self.predict_age(name)


# Create singleton service instance
service = AgeService()
