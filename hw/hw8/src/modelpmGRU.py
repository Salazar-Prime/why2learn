"""
For this embeddings adapted version of the GRUnet shown earlier, we can assume that
the 'input_size' for a tensor representing a word is always 300.
Source: https://blog.floydhub.com/gru-with-pytorch/
with the only modification that the final output of forward() is now
routed through LogSoftmax activation. 

"""
import torch.nn as nn
import torch
class custom_pmGRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, batch_size=1): 
        """
        -- input_size is the size of the tensor for each word in a sequence of words.  If you word2vec
                embedding, the value of this variable will always be equal to 300.
        -- hidden_size is the size of the hidden state in the RNN
        -- output_size is the size of output of the RNN.  For binary classification of 
                input text, output_size is 2.
        -- num_layers creates a stack of GRUs
        """
        # super(DLStudio.TextClassificationWithEmbeddings.GRUnetWithEmbeddings, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        # self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.pmgru = pmGRU(input_size, hidden_size, output_size, batch_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, h, sequence_end=False):
        out, h = self.pmgru(x, h, sequence_end)
        # out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        out = self.logsoftmax(out)
        return out, h

    def init_hidden(self):
        weight = next(self.parameters()).data
        #                  num_layers  batch_size    hidden_size
        hidden = weight.new(  2,          1,         self.hidden_size    ).zero_()
        return hidden

class pmGRU(nn.Module):
    """
    This GRU implementation is based primarily on a "Minimal Gated" version of a GRU as described in
    "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" by Joel Heck and Fathi 
    Salem. The Wikipedia page on "Gated_recurrent_unit" has a summary presentation of the equations 
    proposed by Heck and Salem.
    """
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        ## for forget gate:
        self.project1 = nn.Sequential( nn.Linear(self.input_size + self.hidden_size, self.hidden_size), nn.Sigmoid() )
        ## for interim out:
        self.project2 = nn.Sequential( nn.Linear( self.input_size + self.hidden_size, self.hidden_size), nn.Tanh() ) 
        ## for final out
        self.project3 = nn.Sequential( nn.Linear( self.hidden_size, self.output_size ), nn.Tanh() )                   

    def forward(self, x, h, sequence_end=False):
        combined1 = torch.cat((x, h), 1)
        forget_gate = self.project1(combined1)  
        interim =  forget_gate * h
        combined2  = torch.cat((x, interim), 1)
        output_interim =  self.project2( combined2 )
        output = (1 - forget_gate) * h  +  forget_gate * output_interim
        if sequence_end == False:
            return output, output
        else:
            final_out = self.project3(output)
            return final_out, final_out 

    def init_hidden(self):
        weight = next(self.parameters()).data                                                
        hidden = weight.new(self.batch_size, self.hidden_size).zero_()
        return hidden