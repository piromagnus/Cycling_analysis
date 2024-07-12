
import torch
import torch.nn as nn

# Step 3: Define the LSTM Model
class RPE_LSTM(nn.Module):
    def __init__(self,input_size, embedding_dim, hidden_size, num_layers, output_size,seq_length,num_heads=2):
        super(RPE_LSTM, self).__init__()
        self.embedding = nn.Linear(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_size*seq_length, output_size)
        self.h_fc = nn.Linear(hidden_size+1, hidden_size)
        self.c_fc = nn.Linear(hidden_size+1, hidden_size)
        self.norm_hidden = nn.LayerNorm(hidden_size+1)
        self.init_weights()


    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        for name, param in self.attention.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x,h,c):#,recup):
        """
        recup size : #batch_size, ou #batch_size,seq_length
        """
        x = self.embedding(x)
        # h= self.hidden_merge(torch.cat((h,h_r),dim=2))
        # c= self.hidden_merge(torch.cat((c,c_r),dim=2))
        # if len(recup.shape)==1:
        #     recup = recup.unsqueeze(1)
        #     #expand to number of layers
        #     recup = recup.unsqueeze(0).expand(num_layers,-1,-1)
        
        lstm_out, (h_res, c_res) = self.lstm(x, (h, c))
        # print(h_res.shape,recup.shape)
        # h_res =self.norm_hidden(torch.cat((h_res,recup),dim=2))
        # c_res =self.norm_hidden(torch.cat((c_res,recup),dim=2))
        # h_next=self.h_fc(h_res)
        # c_next=self.c_fc(c_res)
        batch_size = x.size(0)
        lstm_out = self.norm(lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = lstm_out + attn_out  # Residual connection
        attn_out = self.norm(attn_out)
        out = self.fc(attn_out[:,:, :].reshape(batch_size, -1)).squeeze(1)
        return out,h_res,c_res#h_next,c_next


class ResNetBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.norm(out)
        return out + x
    
class ResNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(ResNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([ResNetBlock(hidden_size, hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.norm(out)
        for block in self.blocks:
            out = block(out)
        out = self.fc3(out)
        return out

class RPE_LSTM_h(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, output_size,):
        super(RPE_LSTM_h, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.resnet = ResNet(hidden_size*3, hidden_size, output_size, num_layers=num_layers)
        self.init_weights()


    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x,h,c):#,recup):
        """
        """
        # keep the last of the sequence for the resnet

        x = self.embedding(x)
        last= x[:,-1,:].unsqueeze(0)
        x= x[:,:-1,:]
        lstm_out, (h_res, c_res) = self.lstm(x, (h, c))
        res_in =torch.cat((c_res, h_res,last),dim=2)
        # print(res_in.shape)
        out = self.resnet(res_in)
        return out,h_res,c_res#h_next,c_next



class RPE_LSTM_h_symb(nn.Module):
    def __init__(self,input_size, hidden_size,resnet_width, num_layers, output_size,dropout=0.5):
        super(RPE_LSTM_h_symb, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.resnet = ResNet(hidden_size*3, resnet_width, output_size, num_layers=num_layers)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x,h,c):#,recup):
        """
        """
        # keep the last of the sequence for the resnet

        x = self.embedding(x)
        last= x[:,-1,:].unsqueeze(0)
        x= x[:,:-1,:]
        lstm_out, (h_res, c_res) = self.lstm(x, (h, c))
        res_in =torch.cat((c_res, h_res,last),dim=2)
        # print(res_in.shape)
        out = self.resnet(res_in)
        # print(out.shape)
        out=self.dropout(out)
        out=self.softmax(out)
        return out,h_res,c_res#h_next,c_next


class Encoder_Decoder(nn.Module):
    def __init__(self, encoder, decoder,input_size,embedding_dim):
        super(Encoder_Decoder, self).__init__()
        self.embedding = nn.Linear(input_size, embedding_dim)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x,h,c):
        x = self.embedding(x)
        last= x[:,-1,:].unsqueeze(0)
        x= x[:,:-1,:]
        h_out,c_out = self.encoder(x, h, c)
        dec_out= self.decoder(last, h_out, c_out)
        return dec_out
    

class ResNet_decoder(nn.Module):
    def __init__(self,hidden_size, output_size,dropout=0.5,num_layers=10,res_net_width=32):
        super(ResNet_decoder, self).__init__()
        self.resnet = ResNet(hidden_size*3, res_net_width, output_size, num_layers=num_layers)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = torch.nn.Dropout(p=dropout)
    
    def forward(self,x,h,c):
        res_in =torch.cat((c, h,x),dim=2)
        out = self.resnet(res_in)
        out=self.dropout(out)
        out=self.softmax(out)
        return out
