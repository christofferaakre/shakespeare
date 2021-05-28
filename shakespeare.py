import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# set to True to enable training
# if set to False, training is skipped,
# and the weights are loaded from the session storage.
# this way, you can train the network once, and then
# set train = False, and experiment with it without
# training it again

train = True

# how many characters to predict at once when training
seq_length = 100

hidden_size = 250
num_layers = 1

# training 10 epochs took me about 15 minutes with GPU acceleration
n_epochs = 10
lr = 0.01

# path in session storage to save state to
PATH = './shakespeare_net_pth'

# use GPU if available, else use CPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)


class CustomDataset(Dataset):
    def __init__(self, data_file: str):
        self.data = open(data_file, 'r').read()
        vocab = sorted(set(self.data))
        self.vocab_size = len(vocab)
        self.char2idx = {ch: idx for idx, ch in enumerate(vocab)}
        self.idx2char = {idx: ch for idx, ch in enumerate(vocab)}
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        x = self.char2idx[self.data[i]]
        x = torch.tensor([x])
        x = F.one_hot(x, num_classes=self.vocab_size)
        
        # FloatTensor because the input needs to be type Float
        x = x.type(torch.FloatTensor)
        
        # return next character, or current character if there is no
        # next character
        t = self.char2idx[self.data[i + (i < (self.__len__() - 1))]]
        
        t = torch.tensor([t])
        return (x.to(device), t.to(device))

class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers = 1):
        super(Model, self).__init__()
        self.n_layers = num_layers
        self.vocab_size = vocab_size
        # input shape: (seq_length, 1, vocab_size)
        # output shape: (seq_length, 1, hidden_size)
        self.lstm = nn.LSTM(self.vocab_size,
                            hidden_size,
                            num_layers,
                            batch_first=False
                            )
        # input shape: (N, *, hidden_size)
        # output shape: (N, *, vocab_size)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, input, states_0=None):
        output, (hn, cn) = self.lstm(input, states_0)
        scores = self.linear(output)
        return scores, (hn, cn)

    def generate_sample(self, x, length=500):
        x = x.view(1, 1, self.vocab_size)
        h = torch.zeros(self.n_layers, 1, hidden_size).to(device)
        c = torch.zeros(self.n_layers, 1, hidden_size).to(device)
        text = ''
        for i in range(length):
            scores, (h, c) = self.forward(x, (h, c))
            probs = F.softmax(scores, dim=2).view(self.vocab_size)
            pred = torch.tensor(list(WeightedRandomSampler(probs, 1, replacement=True)))
            x = F.one_hot(pred, num_classes=self.vocab_size)
            x = x.view(1, 1, self.vocab_size).type(torch.FloatTensor).to(device)
            next_character = idx2char[pred.item()]
            text += next_character
        return text

    def init_state(self):
        return (
            torch.zeros(num_layers, 1, hidden_size).to(device),
            torch.zeros(num_layers, 1, hidden_size).to(device)
        )

dataset = CustomDataset(data_file='drive/MyDrive/colab/shakespeare.txt')
char2idx = dataset.char2idx
idx2char = dataset.idx2char
vocab_size = dataset.vocab_size

loader = DataLoader(dataset=dataset,
                    # we are not actually batching the data,
                    # we are just doing one training example at a
                    # time. this is just a trick
                    # so we can use Datalaoder to cut up
                    # the data nicely for us so we don't have
                    # to do it ourselves
                    batch_size=seq_length,
                    # data shuffling can be useful in many instances,
                    # but in this case it would fuck up everything
                    # since each example is a single character (or 100)
                    # because of the 'batching', and we need to preserve
                    # context
                    shuffle=False
                    )



model = Model(vocab_size=vocab_size,
              hidden_size=hidden_size,
              num_layers=num_layers
              ).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

# TRAINING

sample_input = None

n_batches = len(dataset) // seq_length
print(f'n_batches: {n_batches}')

if train:
    for epoch in range(n_epochs):
        state = model.init_state()
        i = 0
        for char, next_char in loader:  
            #x = x.view(seq_length, 1, vocab_size).to(device)
            pred, (h, c) = model(char, state)
            loss = criterion(pred.squeeze(dim=1), next_char.squeeze(dim=1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            state = (h.detach(), c.detach())

            i += 1

            if i % 100 == 0:
                print(f'Epoch: {epoch+1} / {n_epochs} batch {i+1} / {n_batches} Loss: {loss.item()}')
            if i % 500 == 0:
                sample_input = char[0]
                sample = model.generate_sample(sample_input, length=500)
                print(sample)


    torch.save(model.state_dict(), PATH)
    print(f'finished training and saved state to {PATH}')

# load state
model = Model(vocab_size=vocab_size,
              hidden_size=hidden_size,
              num_layers=num_layers
              ).to(device)
model.load_state_dict(torch.load(PATH))

print(sample)
# generate a bunch of text and save it to output.txt
with open('output.txt', 'w') as file:
    for i in range(100):
        sample = model.generate_sample(char[1], length=1000)
        file.write(sample)
        file.write('\n---------------------------------------\n')
