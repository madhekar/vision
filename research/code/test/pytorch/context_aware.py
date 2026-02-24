import torch
from torch.utils.data import Dataset, DataLoader

class DocumentDataset(Dataset):
    def __init__(self, documents):
        # documents is a list of lists: [[s1, s2, ...], [s1, s2, ...]]
        self.data = []
        for doc_id, doc in enumerate(documents):
            for seq_id, segment in enumerate(doc):
                self.data.append({
                    'segment': segment,
                    'doc_id': doc_id,
                    'seq_id': seq_id
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # 'batch' is a list of tuples/dicts returned by __getitem__
    segments = [item['segment'] for item in batch]
    doc_ids = [item['doc_id'] for item in batch]
    
    # Example: Tokenize segments (assume simple split for example)
    tokens = [seg.split() for seg in segments]
    
    # In a real scenario, you would use a tokenizer, pad tokens,
    # and create masks here based on doc_ids to ensure context
    
    return {
        'input_tokens': tokens, # Padded tensors in real use
        'doc_ids': torch.tensor(doc_ids)
    }

# Dummy Data: 2 documents
docs = [
    ["Document one paragraph one.", "Document one paragraph two."],
    ["Document two sentence one.", "Document two sentence two.", "Document two sentence three."]
]

dataset = DocumentDataset(docs)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Example usage
for batch in dataloader:
    print(batch['doc_ids'])
    # Output will show which document the batched segments belong to
