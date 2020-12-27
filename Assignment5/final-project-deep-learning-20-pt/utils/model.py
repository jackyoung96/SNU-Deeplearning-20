import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# import custom modules
from utils.custom_modules import *




#######################################################################################################
# DO NOT CHANGE 
class RNN_ENCODER(nn.Module):
#######################################################################################################
    def __init__(self, ntoken, ninput=256, drop_prob=0.5, nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken        # size of the dictionary
        self.ninput = ninput        # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers      # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN.TYPE
        
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
            
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()


    def define_module(self):
        '''
        e.g., nn.Embedding, nn.Dropout, nn.LSTM
        '''
        self.embedding = nn.Embedding(self.ntoken, self.ninput) # ntoken : 총 단어 개수 , ninput : input dimension
        self.dropout = nn.Dropout(self.drop_prob)

        # GRU를 사용한다.
        self.GRU = nn.GRU(self.ninput, self.nhidden, self.nlayers, 
                        bias = True, batch_first = True,
                        dropout = self.drop_prob, bidirectional = self.bidirectional)
        
    def init_weights(self):
        # GRU 는 initialize가 필요한 듯 하다.
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return Variable(weight.new(self.nlayers * self.num_directions,
                                    bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        '''
        1. caption -> embedding
        2. pack_padded_sequence (embedding, cap_lens)
        3. for rnn, hidden is used
        4. sentence embedding should be returned
        '''
        emb = self.dropout(self.embedding(captions))
        # pack_padded_sequence : batch 마다 길이를 같게 하려고 padd를 넣는데, 이를 병렬처리 하기 위해서 sequence마다
        # pad가 몇개 끼어져 있는지 미리 기록해 두는 작업이다.
        emb = pack_padded_sequence(emb, cap_lens, batch_first= True)

        # GRU module에 input과 hidden을 넣은 후 최종 output과 hidden 출력
        output, hidden = self.GRU(emb, hidden)

        # pack_padded_sequence를 거꾸로 해주는 작업
        word_emb = pad_packed_sequence(output, batch_first= True) # word embedding
        # size = (batch x sequence length x input size)

        # hidden 을 사용한다. dim : (nlayer*n_direction , batch, nhidden)
        sent_emb = hidden.contiguous() # sentense embedding
        sent_emb = sent_emb.view(-1,self.nhidden*self.num_directions)
        # size = (batch*num layer x num hidden*direction)
        
        return word_emb, sent_emb

#######################################################################################################
# DO NOT CHANGE 
class CNN_ENCODER(nn.Module):
#######################################################################################################
    def __init__(self, nif, ntf):
        super(CNN_ENCODER, self).__init__()
        '''
        nef: size of image feature
        '''
        self.nif = nif
        self.ntf = ntf

        self.define_module()


    def define_module(self):
        '''
        '''
        # googleNet pretrained model 을 사용한다.
        model = models.inception_v3(pretrained= True)
        # for param in model.parameters():
        #     param.requires_grad = False
        # model.eval()

        # # 각 layer를 저장해둔다.
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.maxpool1 = model.maxpool1
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.maxpool2 = model.maxpool2
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        self.avgpool = model.avgpool
        self.dropout = model.dropout

        self.fc = nn.Linear(2048, self.ntf)
        self.emb_feature = conv1x1(768, self.nif)

        
    def forward(self, x):
        '''
            input :
            inception v3 의 경우 3*299*299 input에 대해서 짜여진 network이므로, 
            초기 input을 upsampling하여 사용한다.

            output : 
            중간에 auxiliary 부분에서 feature를 추출하고, 
            1x1 convolution 을 통과하여 N x nef x 17 x 17 feature map을 추출한다.
            최종 class 결과 또한 N x nef 사이즈이다.

        '''
        x = nn.Upsample(size=(299,299),mode='bilinear')(x) 
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        
        # 중간 feature를 추출하여 사용 (image regional feature)
        features = x

        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, start_dim=1)
        # N x 2048
        img_class = self.fc(x)
        # N x 1000 (num_classes)
        if not features is None:
            features = self.emb_feature(features)


        # features = (N X nef X 17 X 17)
        # img_class = (N x 1000)
        return img_class, features


#######################
# Generator Part
#######################

#######################################################################################################
# DO NOT CHANGE  
class GENERATOR(nn.Module):
#######################################################################################################
    def __init__(self, nif, ntf):
        super(GENERATOR, self).__init__()
        '''
        '''
        # image feature channel
        self.nif = nif
        # text feature dim
        self.ntf = ntf

        # image feature를 Nx512x16x16로 변환
        self.conv2x2 = nn.Sequential(
            nn.Conv2d(self.nif,512,2,1) 
        )

        # image feature + text feature 
        # text feature dim = ( N x hidden dim )
        # Residual block을 통과
        self.resblocks = nn.Sequential(
            conv3x3(512+128,512), 
            nn.BatchNorm2d(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512), # 4 residual block like original SISGAN
        )


        # decoder same as original SISGAN
        # N x 3 x 128 x 128로 변환
        self.decoder = nn.Sequential(
            upBlock(512,256), # scale 2배로 만들어줌
            upBlock(256,128),
            upBlock(128,64),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

        # conditioning augmentation
        self.mu = nn.Sequential(
            nn.Linear(self.ntf, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.log_sigma = nn.Sequential(
            nn.Linear(self.ntf, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )




    def forward(self, cnn_code, sent_emb, z=None):
        """
        """

        # Reparameterize trick
        z_mean = self.mu(sent_emb)
        z_logvar = self.log_sigma(sent_emb)
        if z is None:
            z = torch.randn(sent_emb.size(0),128)
        if next(self.parameters()).is_cuda:
            z = z.cuda()
        sent_emb = z_mean + z_logvar.exp() * Variable(z)

        # sent_emb를 ( N x 128 x 16 x 16 ) 으로 만들어줌
        sent_emb = sent_emb.unsqueeze(-1).unsqueeze(-1)
        cnn_code = self.conv2x2(cnn_code)
        sent_emb = sent_emb.repeat(1,1,cnn_code.size(2),cnn_code.size(3))
        # concatenate image feature and text feature
        fusion = torch.cat((cnn_code,sent_emb), dim=1)
        # Residual blocks
        fusion = self.resblocks(fusion)

        # decoder
        fake_img = self.decoder(fusion)
        
        return fake_img, (z_mean, z_logvar)




######################
# Discriminator part
######################

#######################################################################################################
# DO NOT CHANGE 
class DISCRIMINATOR(nn.Module):
#######################################################################################################
    def __init__(self, ntf , b_jcu=True):
        super(DISCRIMINATOR, self).__init__()
        '''
        '''
        # text feature dimension
        self.ntf = ntf
        dim = 128
        # Original discriminator encoder of SISGAN
        # result dimension = (N x 512 x 4 x 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, dim, 4, 2, padding=1), # 62
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(dim, 2*dim, 4, 2, padding=1, bias=False), # 29
            nn.BatchNorm2d(2*dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(2*dim, 4*dim, 4, 2, padding=1, bias=False), # 12
            nn.BatchNorm2d(4*dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(4*dim, 8*dim, 4, 2, padding=1, bias=False), # 4
            nn.BatchNorm2d(8*dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(8*dim, 16*dim, 4, 2, padding = 1, bias=False),
            nn.BatchNorm2d(16*dim)
        ) 

        # residual branch in SISGAN
        self.residual_branch = nn.Sequential(
            nn.Conv2d(16*dim, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(128, 16*dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(16*dim)
        )

        # classifier in SISGAN
        self.classifier = nn.Sequential(
            conv1x1(16*dim + 128, 128, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(128, 1, 4)
        )

        # text feature compression
        self.compression = nn.Sequential(
            nn.Linear(self.ntf, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, img, sent_emb):
        '''
        '''
        img_feature = self.encoder(img)
        # residual branch를 한 번 거친 것을 사용해준다.
        img_feature = F.leaky_relu(self.residual_branch(img_feature), 0.2)
        
        sent_emb = self.compression(sent_emb)
        sent_emb = sent_emb.unsqueeze(-1).unsqueeze(-1)
        sent_emb = sent_emb.repeat(1,1,img_feature.size(2), img_feature.size(3))

        fusion = torch.cat((img_feature, sent_emb), dim=1)
        output = self.classifier(fusion).squeeze()
        # output = torch.sigmoid(output)

        return output
