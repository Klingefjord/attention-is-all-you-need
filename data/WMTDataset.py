
import torch
from torch import utils

def create_data_loaders(data_pct=.01, batch_size, vocab_size, num_workers=1):
    def _create_vocab(training_data, validation_data, src_iso='en', tgt_iso='de'):
        src_tokens = Counter()
        tgt_tokens = Counter()

        BOS = '<bos>'
        EOS = '<eos>'
        PAD = '<pad>'
        UNK = '<unk>'

        for _, doc in tqdm(enumerate(training_data + validation_data), position=0, leave=True):
            src_tokens += Counter([w for w in doc[src_iso].split(' ')])
            tgt_tokens  += Counter([w for w in doc[tgt_iso].split(' ')])
                    
        min_frequency = 1
        base_tokens = [PAD, BOS, EOS, UNK] # important that PAD is first in this list in order for zero-padding to work properly
        tgt_vocab_strings  = base_tokens + [token for token, count in src_tokens.most_common(vocab_size - len(base_tokens) - 1)  if count >= min_frequency]
        src_vocab_strings = base_tokens + [token for token, count in tgt_tokens.most_common(vocab_size - len(base_tokens) - 1) if count >= min_frequency]

        # create a dictionary with a default of -1 for word not existing in our vocab
        tgt_vocab  = defaultdict(lambda: -1, { value: key for key, value in enumerate(tgt_vocab_strings)})
        src_vocab = defaultdict(lambda: -1, { value: key for key, value in enumerate(src_vocab_strings)})

        print(f"Created {src_iso} vocab of size {len(src_vocab)}. Most common words are {src_vocab_strings[:10]}")
        print(f"Created {tgt_iso} vocab of size {len(tgt_vocab)}. Most common words are {tgt_vocab_strings[:10]}")

        return (src_vocab, tgt_vocab, src_vocab_strings, tgt_vocab_strings)

    training_data = wmt_dataset(train=True)[:math.floor(len(wmt_dataset(train=True)) * data_pct)]
    # validation dataset is orders of magnitude smaller than training dataset for some reason
    validation_data = wmt_dataset(dev=True)[:math.floor(len(wmt_dataset(dev=True)) * data_pct * 5)]
    src_vocab, tgt_vocab, src_vocab_strings, tgt_vocab_strings = _create_vocab(training_data, validation_data)

    return (train_data_loader(batch_size, src_vocab, tgt_vocab, training_data, num_workers=num_workers), \
        validation_data_loader(batch_size, src_vocab, tgt_vocab, validation_data, num_workers=num_workers))


def train_data_loader(batch_size, src_vocab, tgt_vocab, training_data, num_workers=1):
    """Intialize a training dataset loader from the vocabs and training data"""
    dataset = WMTDataset(training_data, 'en', 'de', src_vocab, tgt_vocab)
    return utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, collate_fn=collate_fn, num_workers=num_workers)
    

def validation_data_loader(batch_size, src_vocab, tgt_vocab, validation_data, num_workers=1):
    """Intialize a validation dataset loader from the vocabs and training data"""
    dataset = WMTDataset(validation_data, 'en', 'de', src_vocab, tgt_vocab)
    return utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, collate_fn=collate_fn, num_workers=num_workers)

    
def collate_fn(batch):
    """
    Zero-pad the batches to be of the same dimensionality and add a src mask to dim=2 and tgt mask to dim=3
    """
    def _src_mask(src, pad_idx=0):
        return (src != pad_idx).unsqueeze(-2)
        
    def _tgt_mask(tgt, pad_idx=0):
        tgt_mask = (tgt != pad_idx).unsqueeze(-2)
        return tgt_mask & subsequent_mask(tgt.shape[-1]).type_as(tgt_mask.data)

    batch_len = len(max(batch, key=lambda entry: len(entry[-1]))[-1])
    batch = [F.pad(b, (0, batch_len - len(b[-1]))) for b in batch]
    batch = torch.stack(batch)

    src_mask_tensor = _src_mask(batch[:, 0, :]).type_as(batch)
    tgt_mask_tensor = _tgt_mask(batch[:, 1, :]).type_as(batch)
    return torch.cat((batch, src_mask_tensor, tgt_mask_tensor), dim=1)


class WMTDataset(utils.data.Dataset):
    """WMT pytorch Dataset wrapper"""

    def __init__(self, data, src_iso, tgt_iso, src_vocab, tgt_vocab, \
            pad_idx=0, start_idx=1, end_idx=2, unk_idx=3):
        super(WMTDataset, self).__init__()
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.unk_idx=unk_idx
        
        self.data = [self._numericalize(doc, src_iso, tgt_iso, src_vocab, tgt_vocab) for doc in data]
        self.data = sorted(self.data, key=lambda entry: len(entry[-1]))

    def _zero_pad(self, sequence_1, sequence_2):
        """Adds pad idx to the shorter sequence until they are the same length"""
        if len(sequence_1) > len(sequence_2):
            sequence_2 = F.pad(sequence_2, (self.pad_idx, len(sequence_1) - len(sequence_1)))
        elif len(sequence_2) > len(sequence_1):  
            sequence_1 = F.pad(sequence_1, (self.pad_idx, len(sequence_2) - len(sequence_1)))
        return sequence_1, sequence_2
        
    def _numericalize(self, doc, src_iso, tgt_iso, src_vocab, tgt_vocab):
        """
        Return a zero-padded tensor with a numericalized src sequence 
        in dim 0 and a numericalized tgt sequence in dim 1
        """

        # append EOS token
        src_sequence = [src_vocab[t] if src_vocab[t] != -1 \
            else self.unk_idx for t in doc[src_iso].split(' ')]
        src_sequence.append(self.end_idx)
        src_sequence = torch.tensor(src_sequence)

        # append BOS and EOS tokens
        tgt_sequence = [tgt_vocab[t] if tgt_vocab[t] != -1 \
            else self.unk_idx for t in doc[tgt_iso].split(' ')]
        tgt_sequence.insert(0, self.start_idx)
        tgt_sequence.append(self.end_idx)
        tgt_sequence = torch.tensor(tgt_sequence)

        return torch.stack(self._zero_pad(src_sequence, tgt_sequence))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]