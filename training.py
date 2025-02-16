import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import os
from tqdm import tqdm
from datasets import load_dataset
import torch.nn as nn
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

class EmotionEncoder:
    """Encodes emotions into embeddings using a predefined emotion set."""
    def __init__(self):
        self.emotions = {
            "neutral": 0,
            "happy": 1,
            "sad": 2,
            "angry": 3,
            "fearful": 4,
            "disgust": 5,
            "surprised": 6
        }
        self.emotion_embeddings = torch.nn.Embedding(len(self.emotions), 256)  # 256-dim emotion embedding
    
    def encode(self, emotion: str) -> torch.Tensor:
        if emotion not in self.emotions:
            emotion = "neutral"  # default to neutral if emotion not found
        emotion_idx = torch.tensor([self.emotions[emotion]])
        return self.emotion_embeddings(emotion_idx)

class AudioPreprocessor:
    """Handles audio preprocessing including normalization and augmentation."""
    def __init__(self, sample_rate: int = 16000, max_length_seconds: float = 30.0):
        self.sample_rate = sample_rate
        self.max_length = int(max_length_seconds * sample_rate)
        self.hop_length = 256  # Standard hop length for mel spectrograms
        self.max_mel_length = self.max_length // self.hop_length
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=self.hop_length,
            n_mels=80
        )
        
    def process(self, wav: torch.Tensor, augment: bool = True) -> torch.Tensor:
        # Ensure audio length is consistent
        if wav.size(-1) > self.max_length:
            wav = wav[..., :self.max_length]
        else:
            # Pad with zeros if too short
            pad_length = self.max_length - wav.size(-1)
            wav = torch.nn.functional.pad(wav, (0, pad_length))
            
        # Normalize audio
        wav = wav / (torch.max(torch.abs(wav)) + 1e-8)
        
        if augment:
            # Random volume adjustment
            wav = wav * (0.8 + 0.4 * torch.rand(1))
            
            # Random noise addition
            if torch.rand(1) < 0.5:
                noise = torch.randn_like(wav) * 0.01
                wav = wav + noise
        
        # Convert to mel spectrogram
        mel = self.mel_transform(wav)
        
        # Ensure mel spectrogram length is consistent
        if mel.size(-1) > self.max_mel_length:
            mel = mel[..., :self.max_mel_length]
        else:
            pad_length = self.max_mel_length - mel.size(-1)
            mel = torch.nn.functional.pad(mel, (0, pad_length))
            
        return mel

class ZonosHFDataset(Dataset):
    def __init__(
        self, 
        dataset_name="mozilla-foundation/common_voice_17_0", 
        language="uz", 
        split="train", 
        sampling_rate=16000,
        emotion_labels_path: Optional[str] = None
    ):
        self.sampling_rate = sampling_rate
        self.language = language
        self.audio_processor = AudioPreprocessor(sampling_rate)
        
        print(f"Loading {dataset_name} dataset...")
        self.dataset = load_dataset(dataset_name, language, split=split)
        print(f"Dataset loaded with {len(self.dataset)} samples")
        
        # Load emotion labels if provided
        self.emotion_labels = {}
        if emotion_labels_path and os.path.exists(emotion_labels_path):
            with open(emotion_labels_path, 'r') as f:
                self.emotion_labels = json.load(f)
    
    def __len__(self):
        return len(self.dataset)
    
    def get_emotion(self, idx: int) -> List[float]:
        """Get emotion vector for a sample, defaulting to neutral if not found."""
        if str(idx) in self.emotion_labels:
            return self.emotion_labels[str(idx)]
        return [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]  # default neutral
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load and process audio
        audio = item['audio']
        wav = torch.FloatTensor(audio['array'])
        sr = audio['sampling_rate']
        
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
            
        # Resample if necessary
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
        
        # Process audio
        mel_spectrogram = self.audio_processor.process(wav)
        
        # Get emotion vector
        emotion = self.get_emotion(idx)
        emotion_tensor = torch.tensor(emotion)
        
        return {
            'audio': wav,
            'mel_spectrogram': mel_spectrogram,
            'text': item['sentence'],
            'language': self.language,
            'emotion': emotion_tensor
        }

def train_zonos(
    model_path="Zyphra/Zonos-v0.1-transformer",
    dataset_name="mozilla-foundation/common_voice_17_0",
    language="uz",
    output_dir="checkpoints",
    batch_size=8,
    learning_rate=1e-4,
    num_epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    emotion_labels_path: Optional[str] = None
):
    # Initialize model
    model = Zonos.from_pretrained(model_path, device=device)
    model.train()
    
    # Create dataset and dataloader
    dataset = ZonosHFDataset(
        dataset_name, 
        language, 
        emotion_labels_path=emotion_labels_path
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=lambda x: {
            'audio': torch.nn.utils.rnn.pad_sequence(
                [s['audio'].squeeze(0)[:16000*30] for s in x],  # Limit to 30 seconds
                batch_first=True
            ).unsqueeze(1),
            'mel_spectrogram': torch.stack([s['mel_spectrogram'] for s in x]),
            'text': [s['text'] for s in x],
            'language': [s['language'] for s in x],
            'emotion': torch.stack([s['emotion'] for s in x])
        }
    )
    
    # Initialize optimizer with warmup
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.1
    )
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            
            # Process each sample in the batch
            batch_loss = 0
            for audio, mel_spec, text, language, emotion in zip(
                batch['audio'], 
                batch['mel_spectrogram'],
                batch['text'], 
                batch['language'],
                batch['emotion']
            ):
                # Move tensors to device
                audio = audio.to(device)
                mel_spec = mel_spec.to(device)
                emotion = emotion.to(device)
                
                # Create speaker embedding
                speaker = model.make_speaker_embedding(audio.squeeze(0), dataset.sampling_rate)
                
                # Prepare conditioning with emotion
                cond_dict = make_cond_dict(
                    text=text,
                    speaker=speaker,
                    language=language,
                    emotion=emotion.tolist()  # Convert tensor to list for make_cond_dict
                )
                conditioning = model.prepare_conditioning(cond_dict)
                
                # Get target codes using the autoencoder
                with torch.no_grad():
                    target_codes = model.autoencoder.encode(audio)
                
                # Forward pass
                output = model(conditioning)
                
                # Calculate losses
                token_loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    target_codes.view(-1)
                )
                
                # Add mel spectrogram reconstruction loss
                mel_loss = torch.nn.functional.mse_loss(
                    model.autoencoder.decode(output.argmax(dim=-1)),
                    mel_spec
                )
                
                # Combined loss
                loss = token_loss + 0.1 * mel_loss
                batch_loss += loss
            
            # Average loss over batch
            batch_loss = batch_loss / batch_size
            
            # Backward pass
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            total_loss += batch_loss.item()
        
        # End of epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"zonos_checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Start training
    train_zonos(
        model_path="Zyphra/Zonos-v0.1-transformer",
        dataset_name="mozilla-foundation/common_voice_17_0",
        language="uz",
        output_dir="checkpoints",
        batch_size=8,
        num_epochs=10,
        emotion_labels_path="emotion_labels.json"  # Optional path to emotion labels
    ) 