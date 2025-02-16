import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import os
from tqdm import tqdm
from datasets import load_dataset

class ZonosHFDataset(Dataset):
    def __init__(self, dataset_name="mozilla-foundation/common_voice_17_0", language="uz", split="train", sampling_rate=16000):
        self.sampling_rate = sampling_rate
        print(f"Loading {dataset_name} dataset...")
        self.dataset = load_dataset(dataset_name, language, split=split)
        print(f"Dataset loaded with {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load and process audio
        # Common Voice stores paths to audio files, so we need to load them
        audio = item['audio']
        wav = torch.FloatTensor(audio['array'])
        sr = audio['sampling_rate']
        
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)  # Add channel dimension
            
        # Resample if necessary
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
        
        return {
            'audio': wav,
            'text': item['sentence'],  # Common Voice uses 'sentence' instead of 'text'
            'language': language  # Common Voice language is set during dataset loading
        }

def train_zonos(
    model_path="Zyphra/Zonos-v0.1-transformer",
    dataset_name="mozilla-foundation/common_voice_17_0",
    language="uz",
    output_dir="checkpoints",
    batch_size=8,
    learning_rate=1e-4,
    num_epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Initialize model
    model = Zonos.from_pretrained(model_path, device=device)
    model.train()
    
    # Create dataset and dataloader
    dataset = ZonosHFDataset(dataset_name, language)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=lambda x: {
            'audio': torch.nn.utils.rnn.pad_sequence([s['audio'] for s in x], batch_first=True),
            'text': [s['text'] for s in x],
            'language': [s['language'] for s in x]
        }
    )
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            
            # Process each sample in the batch
            batch_loss = 0
            for audio, text, language in zip(batch['audio'], batch['text'], batch['language']):
                # Move audio to device
                audio = audio.to(device)
                
                # Create speaker embedding
                speaker = model.make_speaker_embedding(audio, dataset.sampling_rate)
                
                # Prepare conditioning
                cond_dict = make_cond_dict(
                    text=text,
                    speaker=speaker,
                    language=language
                )
                conditioning = model.prepare_conditioning(cond_dict)
                
                # Get target codes using the autoencoder
                with torch.no_grad():
                    target_codes = model.autoencoder.encode(audio)
                
                # Forward pass
                output = model(conditioning)
                
                # Calculate loss (assuming model outputs logits for next token prediction)
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    target_codes.view(-1)
                )
                batch_loss += loss
            
            # Average loss over batch
            batch_loss = batch_loss / batch_size
            
            # Backward pass
            batch_loss.backward()
            optimizer.step()
            
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
        num_epochs=10
    ) 