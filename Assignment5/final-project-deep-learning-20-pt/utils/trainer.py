import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
from miscc.config import cfg
from PIL import Image

import numpy as np
import os
import time

#################################################
# DO NOT CHANGE 
from utils.model import RNN_ENCODER, CNN_ENCODER, GENERATOR, DISCRIMINATOR
#################################################
from utils.loss import cosine_similarity
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from utils.custom_modules import visual_semantic_loss, pairwise_ranking_loss

                
class condGANTrainer(object):
    def __init__(self, output_dir, train_dataloader, test_dataloader, n_words, dataloader_for_wrong_samples=None,
                 log=None, writer=None):
        self.output_dir = output_dir
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.dataloader_for_wrong_samples = dataloader_for_wrong_samples
        
        self.batch_size = cfg.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        
        self.n_words = n_words # size of the dictionary

        self.log = log if log is not None else print
        self.writer = writer

    
    def prepare_data(self, data, txt_encoder=None):
        """
        Prepares data given by dataloader
        e.g., x = Variable(x).cuda()
        """
        #################################################
        # TODO
        # this part can be different, depending on which algorithm is used
        #################################################

        imgs = data['img']
        captions = data['caps']
        captions_lens = data['cap_len']
        class_ids = data['cls_id']
        keys = data['key']
        sentence_idx = data['sent_ix']

        # sort data by the length in a decreasing order
        # the reason of sorting data can be found in https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html 
        sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
        sentence_idx = sentence_idx[sorted_cap_indices].numpy()
        real_imgs = []
        for i in range(len(imgs)):
            imgs[i] = imgs[i][sorted_cap_indices]
            
            if cfg.CUDA:
                real_imgs.append(Variable(imgs[i]).cuda())
            else:
                real_imgs.append(Variable(imgs[i]))
        captions = captions[sorted_cap_indices].squeeze()
        class_ids = class_ids[sorted_cap_indices].numpy()
        keys = [keys[i] for i in sorted_cap_indices.numpy()]

        # SISGAN에서 mismatch caption, synthesis caption이 필요하므로 이를 만들어줌
        #self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        #if txt_encoder is None:
        #    state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER),
        #                        map_location=lambda storage, loc: storage)
        #    self.text_encoder.load_state_dict(state_dict)
        #if cfg.CUDA:
        #    self.text_encoder = self.text_encoder.cuda()
            
        
        #text_feature = self.text_encoder(captions, captions_lens, self.text_encoder.init_hidden(cfg.BATCH_SIZE))
        #text_feature = text_feature.squeeze().data.cpu().numpy()
        # captions_mismatch : 아예 다른 것
        captions_mismatch = torch.Tensor(np.roll(captions, 1, axis=0)).type(torch.LongTensor)
        # caption_relevant : 반으로 나눈 후 반만 다르게
        captions_split = np.split(captions, [captions.shape[0]//2])
        captions_relevant = torch.Tensor(np.concatenate([
            np.roll(captions_split[0],-1,axis=0),
            captions_split[1]
        ])).type(torch.LongTensor)

        if cfg.CUDA:
            captions = Variable(captions).cuda()
            sorted_cap_lens = Variable(sorted_cap_lens).cuda()
            captions_mismatch = Variable(captions_mismatch).cuda()
            captions_relevant = Variable(captions_relevant).cuda()
        else:
            captions = Variable(captions)
            sorted_cap_lens = Variable(sorted_cap_lens)
            captions_mismatch = Variable(captions_mismatch)
            captions_relevant = Variable(captions_relevant)

        return [real_imgs, captions, captions_mismatch, captions_relevant, sorted_cap_lens, class_ids, keys, sentence_idx]
    

    def pretrain(self):
        """
        e.g., for epoch in range(cfg.TRAIN.MAX_EPOCH):
                  for step, data in enumerate(self.train_dataloader, 0):
                      x = self.prepare_data()
                      .....
        """
        #################################################
        # TODO: Implement text guided image manipulation
        # TODO: You should remain log something during training (e.g., loss, performance, ...) with both log and writer
        # Ex. log('Loss at epoch {}: {}'.format(epoch, loss)')
        # Ex. writer.add_scalar('Loss/train', loss, epoch)

        self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        self.image_encoder = CNN_ENCODER(cfg.CNN.EMBEDDING_DIM, cfg.TEXT.EMBEDDING_DIM)

        if cfg.CUDA:
            self.text_encoder = self.text_encoder.cuda()
            self.image_encoder = self.image_encoder.cuda()

        #####################################
        # Pretraining for RNN, CNN Encoder
        #####################################
        self.text_encoder.train()
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.image_encoder.fc.parameters():
            p.requires_grad = True
        self.image_encoder.train()
        param_t = self.text_encoder.parameters()
        param_i = self.image_encoder.fc.parameters()

        lr = 0.002
        optimizer_t = optim.Adam(param_t, lr=lr, betas=(0.5,0.999))
        optimizer_i = optim.Adam(param_i, lr=lr, betas=(0.5,0.999))
        
        minimum_loss = 10000

        for epoch in range(cfg.TRAIN.PRE_EPOCH):
            avg_loss = 0
            for step, data in enumerate(self.train_dataloader, 0):
                img,caps,caps_mis,caps_rel,cap_length,*_ = self.prepare_data(data)
                img = img[0]

                self.text_encoder.zero_grad()
                self.image_encoder.zero_grad()

                # Training
                img_feat,_ = self.image_encoder(img)
                word_feat,text_feat = self.text_encoder(caps,cap_length,self.text_encoder.init_hidden(cfg.BATCH_SIZE))
                loss = visual_semantic_loss(img_feat, text_feat)
                avg_loss += loss
                loss.backward()
                optimizer_t.step()
                optimizer_i.step()

                # Loss 출력
                if step % 10 ==0:
                    self.writer.add_scalar('Loss/pretrain', avg_loss/(step+1), step+1+epoch*len(self.train_dataloader))
                    print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                      % (epoch + 1, cfg.TRAIN.PRE_EPOCH, step + 1, len(self.train_dataloader), avg_loss / (step + 1)))
                    
                    if avg_loss/(step+1) < minimum_loss:
                        self.save_encoder()
                        minimum_loss = avg_loss/(step+1)

        # 학습 종료
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.eval()
        
        #################################################



    def train(self):
        """
        e.g., for epoch in range(cfg.TRAIN.MAX_EPOCH):
                  for step, data in enumerate(self.train_dataloader, 0):
                      x = self.prepare_data()
                      .....
        """

        
        # text encoder load
        self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER),
                                map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.RNN_ENCODER)

        # image encoder load
        self.image_encoder = CNN_ENCODER(cfg.CNN.EMBEDDING_DIM, cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.CNN_ENCODER),
                                map_location=lambda storage, loc: storage)
        self.image_encoder.load_state_dict(state_dict)
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', cfg.TRAIN.CNN_ENCODER)

        if cfg.CUDA:
            self.text_encoder = self.text_encoder.cuda()
            self.image_encoder = self.image_encoder.cuda()


        #####################################
        # Generator, Discriminator Training
        #####################################

        self.netG = GENERATOR(cfg.CNN.EMBEDDING_DIM, cfg.TEXT.EMBEDDING_DIM)
        D = DISCRIMINATOR(cfg.TEXT.EMBEDDING_DIM)
        if cfg.CUDA:
            self.netG = self.netG.cuda()
            D = D.cuda()
        g_optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5,0.9))
        d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5,0.999))
        g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, 100, 0.5)
        d_lr_scheduler = lr_scheduler.StepLR(d_optimizer, 100, 0.5)


        # Pre training for Generator

        

        for epoch in range(cfg.TRAIN.GEN_EPOCH):
            g_lr_scheduler.step()

            avg_G_loss = 0
            avg_kld = 0
            for step, data in enumerate(self.train_dataloader, 0):
                img,caps, caps_mis, caps_rel,cap_length,*_ = self.prepare_data(data)
                img = img[0]

                _,img_feat = self.image_encoder(img)
                _,caps = self.text_encoder(caps, cap_length, self.text_encoder.init_hidden(cfg.BATCH_SIZE))


                # Generator training
                self.netG.zero_grad()
                fake_img, (z_mean, z_logvar) = self.netG(img_feat, caps)
                # VAE에서 사용하는 D_KL
                kld = 0.5*torch.mean(torch.exp(2*z_logvar)+torch.pow(z_mean,2)-2*z_logvar -1)
                avg_kld += kld.item()
                kld.backward(retain_graph=True)
                fake_loss = F.mse_loss(fake_img, img)
                avg_G_loss += fake_loss.item()
                fake_loss.backward()
                g_optimizer.step()

                if step % 10 == 0:
                    self.writer.add_scalar('PreGeneratorLoss/Train', (avg_G_loss+avg_kld)/(step+1), step+1+epoch*len(self.train_dataloader))
                    print('Epoch [%d/%d], Iter [%d/%d], G_fake: %.4f, KLD: %.4f' 
                    % (epoch + 1, cfg.TRAIN.GEN_EPOCH, step + 1, len(self.train_dataloader), avg_G_loss / (step + 1), avg_kld / (step + 1)))
                    self.save_Generator("netG_pre.pth")



        # GAN training
        g_optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5,0.9))
        g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, 100, 0.5)


        G_min_loss = 10000

        for epoch in range(cfg.TRAIN.MAX_EPOCH):
            d_lr_scheduler.step()
            g_lr_scheduler.step()

            # training loop
            avg_D_real_loss = 0
            avg_D_real_m_loss = 0
            avg_D_fake_loss = 0
            avg_G_fake_loss = 0
            avg_kld = 0

            for step, data in enumerate(self.train_dataloader, 0):
                img,caps, caps_mis, caps_rel,cap_length,_,keys,sent_idx = self.prepare_data(data)
                img = img[0]

                _,img_feat = self.image_encoder(img)
                _,caps = self.text_encoder(caps, cap_length, self.text_encoder.init_hidden(cfg.BATCH_SIZE))

                caps_np = caps.squeeze().data.cpu().numpy()
                caps_mis = torch.Tensor(np.roll(caps_np, 1, axis=0))
                # caption_relevant : 반으로 나눈 후 반만 다르게
                caps_split = np.split(caps_np, [caps_np.shape[0]//2])
                caps_rel = torch.Tensor(np.concatenate([
                    np.roll(caps_split[0],-1,axis=0),
                    caps_split[1]
                ]))

                if cfg.CUDA:
                    caps_mis = caps_mis.cuda()
                    caps_rel = caps_rel.cuda()

                # fake latent
                ONES = Variable(1*torch.ones(img.size(0)).float())
                ZEROS = Variable(-1*torch.ones(img.size(0)))
                if cfg.CUDA:
                    ONES = ONES.cuda()
                    ZEROS = ZEROS.cuda()


                # discriminator training
                D.zero_grad()
                # 진짜 이미지에 진짜 caption은 맞다고 판정
                real_logit = D(img, caps)
                # real_loss = F.binary_cross_entropy_with_logits(real_logit, ONES)
                real_loss = F.mse_loss(real_logit , ONES) # SLGAN loss
                avg_D_real_loss += real_loss.item()
                # real_loss.backward()
                # 진짜 이미지에 다른 caption은 틀렸다고 판정
                real_m_logit = D(img, caps_mis)
                # real_m_loss = F.binary_cross_entropy_with_logits(real_m_logit, ZEROS)
                real_m_loss = F.mse_loss(real_m_logit , ZEROS)
                avg_D_real_m_loss += real_m_loss.item()
                # real_m_loss.backward()
                # 만들어낸 이미지는 틀렸다고 판정 (이 때 condition으로는 true label이 아닌 관련 라벨을 준다)
                fake_img, _ = self.netG(img_feat, caps_rel)
                fake_logit = D(fake_img.detach(), caps_rel)
                # fake_loss = cfg.TRAIN.FAKE_WEIGHT * F.binary_cross_entropy_with_logits(fake_logit,ZEROS)
                fake_loss = cfg.TRAIN.FAKE_WEIGHT * F.mse_loss(fake_logit , ZEROS)
                avg_D_fake_loss += fake_loss.item()
                # fake_loss.backward()

                # d_optimizer.step()


                d_loss = real_loss + (real_m_loss + fake_loss)
                d_loss.backward()
                d_optimizer.step()


                # Generator training
                self.netG.zero_grad()
                fake_img, (z_mean, z_logvar) = self.netG(img_feat, caps_rel)
                # VAE에서 사용하는 D_KL
                kld = 0.5*torch.mean(torch.exp(2*z_logvar)+torch.pow(z_mean,2)-2*z_logvar -1)
                avg_kld += kld.item()
                kld.backward(retain_graph=True)
                g_logit = D(fake_img, caps_rel)
                # fake image가 참이라고 받아들여 지도록
                # fake_loss = F.binary_cross_entropy_with_logits(fake_logit, ONES)
                g_loss = F.mse_loss(g_logit , ONES)
                avg_G_fake_loss += g_loss.item()
                # fake_loss.backward()

                g_loss.backward()

                g_optimizer.step()
                
                # g_optimizer.step()

                if step % 10 == 0:
                    D_grad = 0
                    D_param_count = 0
                    for param in D.parameters():
                        D_param_count += 1
                        D_grad += np.abs(param.grad.data.cpu().numpy().mean())
                    self.writer.add_scalar('D_gradient', D_grad/D_param_count, step+1 +epoch*len(self.train_dataloader))
                    G_grad = 0
                    G_param_count = 0
                    for param in self.netG.parameters():
                        if param.grad is not None:
                            G_param_count += 1
                            G_grad += np.abs(param.grad.data.cpu().numpy().mean())
                    self.writer.add_scalar('G_gradient', G_grad/G_param_count, step+1 +epoch*len(self.train_dataloader))

                    self.writer.add_scalar('D_real_Loss/Train', avg_D_real_loss/(step+1), step+1 +epoch*len(self.train_dataloader))
                    self.writer.add_scalar('D_real_m_Loss/Train', avg_D_real_m_loss/(step+1), step+1+epoch*len(self.train_dataloader))
                    self.writer.add_scalar('D_fake_Loss/Train', avg_D_fake_loss/(step+1), step+1+epoch*len(self.train_dataloader))
                    self.writer.add_scalar('G_fake_Loss/Train', avg_G_fake_loss/(step+1), step+1+epoch*len(self.train_dataloader))
                    self.writer.add_scalar('kld_Loss/Train', avg_kld/(step+1), step+1+epoch*len(self.train_dataloader))
                    # self.writer.add_scalar('Total Loss', {'D_loss':(avg_D_real_loss+avg_D_real_m_loss+avg_D_fake_loss)/(step+1),
                    #                                     'G_loss':(avg_G_fake_loss)/(step+1)}, step+1+epoch*len(self.train_dataloader))
                    print('Epoch [%d/%d], Iter [%d/%d], D_real: %.4f, D_mis: %.4f, D_fake: %.4f, G_fake: %.4f, KLD: %.4f' 
                    % (epoch + 1, cfg.TRAIN.MAX_EPOCH, step + 1, len(self.train_dataloader), avg_D_real_loss / (step + 1), 
                    avg_D_real_m_loss / (step + 1), avg_D_fake_loss / (step + 1), avg_G_fake_loss / (step + 1), avg_kld / (step + 1)))
                    if G_min_loss > avg_G_fake_loss/(step+1):
                        self.save_Generator(cfg.TRAIN.GENERATOR)
                        G_min_loss = avg_G_fake_loss/(step+1)

                if step == 0 :
                    im = fake_img[0].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    im.save(os.path.join(cfg.TEST.MID_TEST_IMAGES,'epoch_{}.png'.format(epoch)))
            


            
        #################################################

    def generate_data_for_eval(self):
        # load the text encoder model to generate images for evaluation
        self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER),
                                map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.RNN_ENCODER)
        self.text_encoder.eval()


        #################################################
        # TODO
        # this part can be different, depending on which algorithm is used
        #################################################

        # image encoder load
        self.image_encoder = CNN_ENCODER(cfg.CNN.EMBEDDING_DIM, cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.CNN_ENCODER),
                                map_location=lambda storage, loc: storage)
        self.image_encoder.load_state_dict(state_dict)
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', cfg.TRAIN.CNN_ENCODER)
        self.image_encoder.eval()


        # load the generator model to generate images for evaluation
        self.netG = GENERATOR(cfg.CNN.EMBEDDING_DIM, cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.GENERATOR),
                                map_location=lambda storage, loc: storage)
        # state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, "netG_pre.pth"),
        #                         map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(state_dict)
        for p in self.netG.parameters():
            p.requires_grad = False
        print('Load generator from:', cfg.TRAIN.GENERATOR)
        self.netG.eval()

        noise = Variable(torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM))

        if cfg.CUDA:
            self.text_encoder = self.text_encoder.cuda()
            self.image_encoder = self.image_encoder.cuda()
            self.netG = self.netG.cuda()
            noise = noise.cuda()

        for step, data in enumerate(self.test_dataloader, 0):
            imgs, captions,m,r,cap_lens, class_ids, keys, sent_idx = self.prepare_data(data)
            #################################################
            # TODO
            # word embedding might be returned as well
            # hidden = self.text_encoder.init_hidden(self.batch_size)
            # sent_emb = self.text_encoder(captions, cap_lens, hidden)
            # sent_emb = sent_emb.detach()
            #################################################
            img = imgs[0]
            noise.data.normal_(0, 1)

            _,img_feat = self.image_encoder(img)
            img_feat = img_feat.detach()
            _,text_feat = self.text_encoder(captions, cap_lens, 
                    self.text_encoder.init_hidden(cfg.BATCH_SIZE))
            text_feat = text_feat.detach()

            #################################################
            # TODO
            # this part can be different, depending on which algorithm is used
            # the main purpose is generating synthetic images using caption embedding and latent vector (noise)
            # fake_img = self.netG(noise, sent_emb, img_emb, ...)
            fake_imgs, _ = self.netG(img_feat, text_feat)
            #################################################

            # Save original img
            for j in range(self.batch_size):
                if not os.path.exists(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0])):
                    os.mkdir(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0]))
                if not os.path.exists(os.path.join(cfg.TEST.ORIG_TEST_IMAGES, keys[j].split('/')[0])):
                    os.mkdir(os.path.join(cfg.TEST.ORIG_TEST_IMAGES, keys[j].split('/')[0]))
                if not os.path.exists(os.path.join(cfg.TEST.ORIG_TEST_IMAGES, keys[j] + '.png')):
                    im = imgs[0][j].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    print(os.path.join(cfg.TEST.ORIG_TEST_IMAGES, keys[j] + '.png'))
                    im.save(os.path.join(cfg.TEST.ORIG_TEST_IMAGES, keys[j] + '.png'))

                im = fake_imgs[j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                print(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))
                im.save(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))

    
    def save_encoder(self):
        torch.save(self.text_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER))
        torch.save(self.image_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.CNN_ENCODER))

    def save_Generator(self, name=cfg.TRAIN.GENERATOR):
        """
        Saves models
        """
        torch.save(self.netG.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, name))
