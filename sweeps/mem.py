"""
How do different models perform on a task of repeating the whole sequence in the same order or arbitrary permuted?
"""
from pathlib import Path
import torch

from train import train, parser, device
from train_tape import Tapes
from delta import LM
from multiquery_ar import sequence_recall


def make_sequence_recall_tapes(num_examples=100_000, permuted=False, vocab_size=256, batch_size=32, random_keys=False, seq_len=64):
    seq_len = seq_len * 2 # double the sequence length due to stacking
    num_train_batches = num_examples // batch_size
    num_train_examples = num_train_batches*batch_size
    num_valid_batches = 3_000 // batch_size
    num_valid_examples = num_valid_batches*batch_size
    valid_inputs, valid_targets, _ = sequence_recall(vocab_size=vocab_size, num_examples=num_valid_examples, input_seq_len=seq_len, seed=43, stacked=True, random_keys=random_keys, permuted=permuted)
    train_inputs, train_targets, vocab_size = sequence_recall(vocab_size=vocab_size, num_examples=num_train_examples, input_seq_len=seq_len, seed=42, stacked=True, random_keys=random_keys, permuted=permuted)

    class Repeat:
        def __init__(self, inputs, targets, count=100000):
            self.inputs = inputs
            self.targets = targets
            self.count = count

        def __len__(self):
            return len(self.inputs) * self.count

        def __getitem__(self, i):
            input, target = self.inputs[i % len(self.inputs)], self.targets[i % len(self.targets)]
            return input.long(), target.long()

    tapes = Tapes(
        vocab_size=vocab_size,
        seq_len=seq_len,
        train=Repeat(train_inputs.view(num_train_batches, batch_size, -1, 2).to(device),
                     train_targets.view(num_train_batches, batch_size, -1).to(device)),
        valid=Repeat(valid_inputs.view(num_valid_batches, batch_size, -1, 2).to(device),
                     valid_targets.view(num_valid_batches, batch_size, -1).to(device), count=1),
    )
    print('mem: one epoch takes', num_train_batches, 'steps')

    i, t = tapes.train[0]
    print(i.shape, t.shape, 'train shapes')
    print('effective vocab size', vocab_size)
    return tapes, vocab_size


def run(*, run_id='42', lr=1e-3, steps=100_000, num_examples=100_000, num_layers=2, dim=32, seed=3407,
         permuted=True, vocab_size=256, batch_size=32, random_keys=False, seq_len=64):
    args = parser.parse_args()
    args.exp = Path(args.exp.substitute(run_id=run_id, **vars(args)))
    args.exp.mkdir(parents=True, exist_ok=True)
    args.lr = lr
    args.steps = steps

    tapes, vocab_size = make_sequence_recall_tapes(
        num_examples,
        permuted=permuted,
        vocab_size=vocab_size,
        batch_size=batch_size,
        random_keys=random_keys,
        seq_len=seq_len
    )

    torch.manual_seed(seed)

    model = LM(vocab_size=vocab_size, dim=dim, num_layers=num_layers)
    print(model)
    opt = torch.optim.AdamW(model.parameter_groups(), lr=args.lr, betas=(0.9, 0.999), fused=False)
    train(model, tapes, opt, args=args)


if __name__ == '__main__':
    run()