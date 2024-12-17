# src/audio/speaker_identity.py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from pyannote.audio import Inference
import pickle
from dataclasses import dataclass
import soundfile as sf
import tempfile
import os


@dataclass
class SpeakerProfile:
    name: str
    embeddings: List[np.ndarray]
    audio_samples: List[str]  # Paths to reference audio files


class SpeakerIdentifier:
    def __init__(self, auth_token: str, device: Optional[torch.device] = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # Load the embedding model
        self.embedding_model = Inference(
            "pyannote/embedding",
            use_auth_token=auth_token
        ).to(self.device)

        self.speakers: Dict[str, SpeakerProfile] = {}
        self.similarity_threshold = 0.3  # Adjust this for stricter/looser matching

    def add_speaker(self, name: str, audio_path: Path) -> None:
        """Add a new speaker profile from reference audio."""
        embedding_feature = self.embedding_model({"audio": str(audio_path)})
        # Convert SlidingWindowFeature to numpy array
        embedding = np.mean(embedding_feature.data, axis=0)

        if name in self.speakers:
            self.speakers[name].embeddings.append(embedding)
            self.speakers[name].audio_samples.append(str(audio_path))
        else:
            self.speakers[name] = SpeakerProfile(
                name=name,
                embeddings=[embedding],
                audio_samples=[str(audio_path)]
            )

    def identify_speaker(self, audio_segment: np.ndarray, sample_rate: int = 16000) -> Tuple[Optional[str], float]:
        """Identify a speaker from an audio segment."""
        if not self.speakers:
            print("No speaker profiles loaded!")
            return None, 0.0
            
        try:
            print(f"Available speakers: {list(self.speakers.keys())}")
            
            # Save the audio segment temporarily
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_segment, sample_rate)
                print(f"Saved temporary segment to {tmp_file.name}")
                
            # Get embedding for the segment
            embedding_feature = self.embedding_model({"audio": tmp_file.name})
            segment_embedding = np.mean(embedding_feature.data, axis=0)
            print(f"Generated embedding with shape: {segment_embedding.shape}")
            print(f"Embedding type: {type(segment_embedding)}")
            print(f"Embedding dtype: {segment_embedding.dtype}")
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            # Compare with known speakers
            best_match = None
            highest_similarity = -1.0
            
            for name, profile in self.speakers.items():
                similarities = []
                for ref_embedding in profile.embeddings:
                    if hasattr(ref_embedding, 'data'):
                        ref_embedding = np.mean(ref_embedding.data, axis=0)
                    print(f"Reference embedding shape: {ref_embedding.shape}")
                    print(f"Reference embedding type: {type(ref_embedding)}")
                    
                    try:
                        similarity = self.cosine_similarity(segment_embedding, ref_embedding)
                        print(f"Raw similarity value: {similarity}, type: {type(similarity)}")
                        similarities.append(float(similarity))
                        print(f"Similarity with {name}: {float(similarity):.3f}")
                    except Exception as e:
                        print(f"Error calculating similarity: {e}")
                        print(f"Segment embedding example values: {segment_embedding[:5]}")
                        print(f"Reference embedding example values: {ref_embedding[:5]}")
                        continue
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    print(f"Average similarity for {name}: {avg_similarity:.3f}")
                    
                    if avg_similarity > highest_similarity and avg_similarity > self.similarity_threshold:
                        highest_similarity = avg_similarity
                        best_match = name
            
            if best_match:
                print(f"Identified speaker: {best_match} with confidence: {highest_similarity:.3f}")
            else:
                print(f"No speaker identified (highest similarity: {highest_similarity:.3f})")
                
            return best_match, highest_similarity
                
        except Exception as e:
            print(f"Error in speaker identification: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, 0.0

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.sum(a * b)  # Element-wise multiplication and sum
        norm_a = np.sqrt(np.sum(a * a))
        norm_b = np.sqrt(np.sum(b * b))
        similarity = dot_product / (norm_a * norm_b)
        return similarity  # Will be a scalar value

    def save_profiles(self, path: Path) -> None:
        """Save speaker profiles to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.speakers, f)

    def load_profiles(self, path: Path) -> None:
        """Load speaker profiles from disk."""
        with open(path, 'rb') as f:
            self.speakers = pickle.load(f)
