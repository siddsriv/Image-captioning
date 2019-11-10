import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)    
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, drop_prob = 0.2):
        #Find why this line is giving an error
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers, dropout = drop_prob, batch_first = True)
        self.linear = nn.Linear(hidden_size,vocab_size)
        
    
    def forward(self, features, captions):
        captions = captions[:,:-1] 
        embeddings = self.embedding(captions)
        total_input = torch.cat((features.unsqueeze(1),embeddings),1)
        lstm_out, self.hidden = self.lstm(total_input)
        outputs = self.linear(lstm_out)
        return outputs
    


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        hidden = (torch.zeros(self.num_layers,1,self.hidden_size).to(inputs.device),
                  torch.zeros(self.num_layers,1,self.hidden_size).to(inputs.device))
        
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs,hidden)    
            # 1*1*vocab_dim
            output_vocab = self.linear(lstm_out)    
            # 1*vocab_dim
            output_vocab = output_vocab.squeeze(1)  
            output_word = output_vocab.argmax(1)                
            output.append(output_word.item())
            # 1*1*embed_dim
            inputs = self.embedding(output_word.unsqueeze(0))   
        return output