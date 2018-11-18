import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    """
    Encoder:
        Use the pretrained resnet101 replace the last fc layer and re-train the last fc layer.
    """

    def __init__(self, attention=False, embed_size=256, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.attention = attention
        self.enc_image_size = encoded_image_size
        resnet = models.resnet101(pretrained=True)
        if self.attention:
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(resnet.children())[:-2]
            self.resnet = nn.Sequential(*modules)
            # Resize image to fixed size to allow input images of variable size
            self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        else:
            # change the output dimension of the last fc and only requires grad for the weight and bias
            self.resnet = resnet
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        """
        Forward propagation.
        @param 
            images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        return: 
            image embeddings:
                if attention: (batch_size, encoded_image_size, encoded_image_size, 2048)
                else: (batch_size, embed_size)
        """
        if self.attention:
            with torch.no_grad():
                out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
                out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
                out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
            return out
        else:
            out = self.resnet(images)  # (batch_size, embed_size)
            return out


class StatefulLSTM(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatefulLSTM, self).__init__()

        self.lstm = nn.LSTMCell(in_size, out_size)
        self.out_size = out_size

        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):

        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = Variable(torch.zeros(state_size)).cuda()
            self.h = Variable(torch.zeros(state_size)).cuda()
        self.h, self.c = self.lstm(x, (self.h, self.c))

        return self.h


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20, stateful=True):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
