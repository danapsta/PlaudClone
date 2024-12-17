# src/audio/speaker_identity.py
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from pyannote.audio import Inference
import pickle
from dataclasses import dataclass

@dataclass
class SpeakerProfile:
    name: str
    embeddings: List[np.ndarray]
    audio_samples: List[str]  # Paths to reference audio files

class SpeakerIdentifier:
    def __init__(self, auth_token: str, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the embedding model
        self.embedding_model = Inference(
            "pyannote/embedding", 
            use_auth_token=auth_token
        ).to(self.device)
        
        self.speakers: Dict[str, SpeakerProfile] = {}
        self.similarity_threshold = 0.75  # Adjust this for stricter/looser matching
        
    def add_speaker(self, name: str, audio_path: Path) -> None:
        """Add a new speaker profile from reference audio."""
        embedding = self.embedding_model({"audio": str(audio_path)})
        
        if name in self.speakers:
            self.speakers[name].embeddings.append(embedding)
            self.speakers[name].audio_samples.append(str(audio_path))
        else:
            self.speakers[name] = SpeakerProfile(
                name=name,
                embeddings=[embedding],
                audio_samples=[str(audio_path)]
            )
            
    def identify_speaker(self, audio_segment: np.ndarray) -> Optional[str]:
        """Identify a speaker from an audio segment."""
        if not self.speakers:
            return None
            
        # Get embedding for the segment
        segment_embedding = self.embedding_model({"audio": audio_segment})
        
        # Compare with known speakers
        best_match = None
        highest_similarity = -1
        
        for name, profile in self.speakers.items():
            # Compare with all embeddings for this speaker
            similarities = [
                self.cosine_similarity(segment_embedding, ref_embedding)
                for ref_embedding in profile.embeddings
            ]
            avg_similarity = np.mean(similarities)
            
            if avg_similarity > highest_similarity and avg_similarity > self.similarity_threshold:
                highest_similarity = avg_similarity
                best_match = name
                
        return best_match
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    def save_profiles(self, path: Path) -> None:
        """Save speaker profiles to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.speakers, f)
            
    def load_profiles(self, path: Path) -> None:
        """Load speaker profiles from disk."""
        with open(path, 'rb') as f:
            self.speakers = pickle.load(f)