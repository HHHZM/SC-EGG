import os
import torch
import wandb
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import datasets.image_util as util
from datasets.SUNDataLoader import SUNDataLoader
from models.Attention import DAZLE, MinMaxScaler_Torch
import models.utils as utils
import models.VAEGAN as VAEGAN
import models.classifier_images as classifier
from models.helper_func import eval_zs_gzsl
import numpy as np
import torch.nn as nn


wandb.init(project='SCEGG', config='wandb_config/config_sun.yaml')
config = wandb.config
config.lambda2 = config.lambda1
config.visual_dim = 2048
config.encoder_layer_sizes[0] = config.visual_dim
config.decoder_layer_sizes[-1] = config.visual_dim
config.latent_size = config.attSize
print('Config file from wandb:', config)

print("Random Seed: ", config.manualSeed)
random.seed(config.manualSeed)
np.random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)
torch.cuda.manual_seed_all(config.manualSeed)
cudnn.benchmark = True

dataloader = SUNDataLoader('./', config.device,
                           img_size=config.atten_imgsize,
                           use_unzip=config.atten_use_unzip)
dataloader.construct_torch_dataset(config)
dataloader_aux = util.DATA_LOADER(config)

#########################################
#### Stage 1: Local Attention
#########################################
assert config.is_atten_model 
# netA = DenseAttention(config, dataloader.w2v_att, dataloader.att).to(config.device)
# optimizerA = optim.Adam(netA.parameters(), lr=config.atten_lr, betas=(config.beta1, 0.999))
netA = DAZLE(dim_f=config.resSize,
                dim_v=config.atten_dim_v,
                init_w2v_att=dataloader.w2v_att,
                att=dataloader.att,
                normalize_att=dataloader.normalize_att,
                seenclass=dataloader.seenclasses,
                unseenclass=dataloader.unseenclasses,
                lambda_1=0.0,
                lambda_2=0.0,
                lambda_3=0.0,
                device=config.device,
                trainable_w2v=config.atten_trainable_w2v,
                normalize_V=config.atten_normalize_V, 
                normalize_F=config.atten_normalize_F, 
                is_conservative=False,
                prob_prune=0,
                desired_mass=1, 
                uniform_att_1=False, 
                uniform_att_2=True, 
                is_conv=False, 
                is_bias=False,
                # CLS
                att_compose_norm=config.atten_compose_norm, 
                att_compose_type=config.atten_compose_type,
                lambda_localCE=config.atten_lambda_localCE, 
                lambda_globalCE=config.atten_lambda_globalCE
                ).to(config.device)
optimizerA = optim.RMSprop(netA.parameters(),
                            lr=0.0001,
                            weight_decay=0.0001,
                            momentum=0.9)
if config.atten_pretrain:
    # training
    for i in range(0,config.atten_itnum):
        netA.train()
        optimizerA.zero_grad()
        batch_label, batch_feature, batch_att = dataloader.next_seen_batch(config.batch_size)
        out_package = netA(batch_feature)

        in_package = out_package
        in_package['batch_label'] = batch_label
        
        # out_package=model.compute_loss(in_package)
        out_package=netA.compute_loss_CLS(in_package)
        loss = out_package['loss']
        loss_CE_local = out_package['loss_CE_local']
        loss_CE_global = out_package['loss_CE_global']
        loss.backward()
        optimizerA.step()
        if i%100==0:
            print('-'*30)
            acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataloader, netA, config.device,
                                                          bias_seen=0, bias_unseen=0)
            print('iter {} loss {:.3f} '.format(i,loss.item()), end='')
            print('loss_CE_local {:.3f} '.format(loss_CE_local.item()), end='')
            print('loss_CE_global {:.3f} '.format(loss_CE_global.item()))
            print('acc_seen {:.3f} acc_novel {:.3f} H {:.3f} '.format(
                acc_seen, acc_novel, H), end='')  
            print('acc_zs {:.3f}'.format(acc_zs))
else: raise Exception

# MinMax Preprocessing
train_seen_set = torch.utils.data.TensorDataset(dataloader.data['train_seen']['resnet_features'])
train_seen_loader = torch.utils.data.DataLoader(train_seen_set, batch_size=config.batch_size)
train_seen_Hs = []
with torch.no_grad():
    for features in train_seen_loader:
        input_res = features[0].to(config.device)
        f_global = netA.extract_pooling_attributes_features(
            input_res, config.atten_compose_type_runtime)
        train_seen_Hs.append(f_global)
    train_seen_Hs = torch.cat(train_seen_Hs)
scale_min = torch.min(train_seen_Hs, axis=0)[0]
scale_max = torch.max(train_seen_Hs, axis=0)[0]

MinMaxScaler = MinMaxScaler_Torch(scale_min, scale_max).to(config.device)
netA.Scaler.load_state_dict(MinMaxScaler.state_dict())

#########################################
#### Stage 2: Local to Global
#########################################
class SCEN(nn.Module):
    def __init__(self, config, CLSDAZLE_paras):
        super(SCEN, self).__init__()  
        self.config = config
        self.netA = DAZLE(dim_f=config.resSize,
                             dim_v=config.atten_dim_v,
                             init_w2v_att=dataloader.w2v_att,
                             att=dataloader.att,
                             normalize_att=dataloader.normalize_att,
                             seenclass=dataloader.seenclasses,
                             unseenclass=dataloader.unseenclasses,
                             lambda_1=0.0,
                             lambda_2=0.0,
                             lambda_3=0.0,
                             device=config.device,
                             trainable_w2v=config.atten_trainable_w2v,
                             normalize_V=config.atten_normalize_V,
                             normalize_F=config.atten_normalize_F,
                             is_conservative=False,
                             prob_prune=0,
                             desired_mass=1,
                             uniform_att_1=False,
                             uniform_att_2=True,
                             is_conv=False,
                             is_bias=False,
                             # CLS
                             att_compose_norm=config.atten_compose_norm,
                             att_compose_type=config.atten_compose_type,
                             lambda_localCE=config.atten_lambda_localCE,
                             lambda_globalCE=config.atten_lambda_globalCE
                             )
        self.netA.load_state_dict(CLSDAZLE_paras)
        for p in self.netA.parameters():
            p.requires_grad = False
        self.ft_compose_cls=nn.Sequential(
            nn.Linear(config.resSize, 1024),
            # nn.LeakyReLU(),
            # nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(1024, config.nclass_all)).to(config.device)

    def compute_loss(self, out_package):
        local_cls_embed = out_package['local_cls_embed']
        cls_out = out_package['S_pp']
        loss_ts = F.mse_loss(cls_out, local_cls_embed, reduction='mean')
        # loss_ts = utils.WeightedL1(cls_out, local_cls_embed)
        return loss_ts

    def forward(self, input_res):
        f_global = self.netA.extract_pooling_attributes_features(
                    input_res, config.atten_compose_type_runtime)
        out_package = self.netA.forward(input_res)
        cls_out = self.ft_compose_cls(f_global)
        out_package['S_pp'] = cls_out
        return out_package

ts_model = SCEN(config, netA.state_dict().copy()).to(config.device)

optimizerFTCC = optim.RMSprop(
    ts_model.parameters(),
    lr=0.0001,
    weight_decay=0.0001,
    momentum=0.9)

if not config.atten_pretrain and not config.ts_train:
    ts_model.load_state_dict(torch.load(config.ts_model_path))
    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataloader, ts_model, config.device,
                                                    bias_seen=0, bias_unseen=0)
    print('acc_seen {:.3f} acc_novel {:.3f} H {:.3f} '.format(
        acc_seen, acc_novel, H), end='')  
    print('acc_zs {:.3f}'.format(acc_zs))
else:
    for i in range(0,config.ts_itnum):
        ts_model.train()
        optimizerFTCC.zero_grad()

        batch_label, input_res, input_att = dataloader.next_seen_batch(config.batch_size)
        input_res = input_res.to(config.device)
        input_att = input_att.to(config.device)

        package = ts_model(input_res)
        loss_ts = ts_model.compute_loss(package)

        package['batch_label'] = batch_label
        loss_CE1 = ts_model.netA.compute_CE_loss_non_conservative(package)['loss_CE']
        loss_CE2 = ts_model.netA.compute_CE_loss_conservative(package)['loss_CE']

        loss_ts.backward()
        optimizerFTCC.step()

        if i%100==0:
            print('-'*30)
            acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataloader, ts_model, config.device,
                                                            bias_seen=0, bias_unseen=0)
            print('iter {} '.format(i), end='')
            print('loss_ts {:.3f} '.format(loss_ts.item()), end='')
            print('loss_CE2 {:.3f} '.format(loss_CE2.item()), end='')
            print('loss_CE1 {:.3f} '.format(loss_CE1.item()), end='')
            print('')
            print('acc_seen {:.3f} acc_novel {:.3f} H {:.3f} '.format(
                acc_seen, acc_novel, H), end='')  
            print('acc_zs {:.3f}'.format(acc_zs))

#########################################
#### Stage 3: Train TF-VAEGAN
#########################################
# TF-VAEGAN
netE = VAEGAN.Encoder(config).to(config.device)
netG = VAEGAN.Generator(config).to(config.device)
netD = VAEGAN.Discriminator_D1(config).to(config.device)
noise = torch.FloatTensor(config.batch_size, config.nz).to(config.device)
one = torch.FloatTensor([1]).to(config.device)
mone = one * -1
optimizerE = optim.Adam(netE.parameters(), lr=config.lr)
optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
if config.is_dec:
    netDec = VAEGAN.AttDec(config, config.attSize).to(config.device)
    optimizerDec = optim.Adam(netDec.parameters(), lr=config.dec_lr, betas=(config.beta1, 0.999))
    if config.is_feedback:
        assert config.feedback_loop == 2
        netF = VAEGAN.Feedback(config).to(config.device)
        optimizerF = optim.Adam(netF.parameters(), lr=config.feed_lr, betas=(config.beta1, 0.999))

if config.is_load_vaegan:
    netE.load_state_dict(torch.load(os.path.join(config.vaegan_model_path, 'netE.pth')))
    netG.load_state_dict(torch.load(os.path.join(config.vaegan_model_path, 'netG.pth')))
    netD.load_state_dict(torch.load(os.path.join(config.vaegan_model_path, 'netD.pth')))
    netDec.load_state_dict(torch.load(os.path.join(config.vaegan_model_path, 'netDec.pth')))
        
# extract dense-semantic features
(train_att_feature,
 test_seen_att_feature,
 test_unseen_att_feature) = utils.inference_atten_f(config, netA, dataloader)

# train vaegan
lambda1 = config.lambda1
best_gzsl_acc = -1
best_zsl_acc = -1
for epoch in range(0,config.nepoch):
    for loop in range(0,config.feedback_loop):
        for i in range(0, dataloader.ntrain, config.batch_size):

            for p in netA.parameters():
                p.requires_grad = False
            for p in ts_model.parameters():
                p.requires_grad = False

            # Discriminator training 
            for p in netD.parameters(): #unfreeze discrimator
                p.requires_grad = True
            if config.is_dec:
                for p in netDec.parameters(): #unfreeze deocder
                    p.requires_grad = True

            # Train D1 and Decoder (and Decoder Discriminator)
            gp_sum = 0 #lAMBDA VARIABLE
            for iter_d in range(config.critic_iter):

                netD.zero_grad()
                if config.is_dec:
                    netDec.zero_grad()
                netA.zero_grad()

                batch_label, input_res, input_att = dataloader.next_seen_batch(config.batch_size)
                input_res = input_res.to(config.device)
                input_att = input_att.to(config.device)
                f_global = netA.extract_pooling_attributes_features(
                    input_res, config.atten_compose_type_runtime)
                # f_global = F.normalize(f_global, dim = 1)

                # update Decoder
                if config.is_dec:
                    recons = netDec(f_global)
                    R_cost = config.recons_weight * utils.WeightedL1(recons, input_att)
                    R_cost.backward(retain_graph=True)
                    optimizerDec.step()

                # update Discriminator
                criticD_real = netD(f_global, input_att)
                criticD_real = config.gammaD * criticD_real.mean()

                if config.encoded_noise:        
                    means, log_var = netE(f_global, input_att)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([config.batch_size, config.latent_size]).cpu()
                    eps = Variable(eps.to(config.device))
                    z = eps * std + means
                else:
                    noise.normal_(0, 1)
                    z = Variable(noise)

                if config.is_dec and config.is_feedback and loop == 1:
                    fake = netG(z, c=input_att)
                    dec_out = netDec(fake)
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake = netG(z, a1=config.a1, c=input_att, feedback_layers=feedback_out)
                else:
                    fake = netG(z, c=input_att)

                # fake = netG(z, c=input_att)
                criticD_fake = netD(fake.detach(), input_att)
                criticD_fake = config.gammaD * criticD_fake.mean()
                # gradient penalty
                gradient_penalty = config.gammaD * utils.calc_gradient_penalty(
                    config, netD, f_global, fake.data, input_att, lambda1)
                # if opt.lambda_mult == 1.1:
                gp_sum += gradient_penalty.data
                Wasserstein_D = criticD_real - criticD_fake
                #add Y here and #add vae reconstruction loss
                D_cost = criticD_fake - criticD_real + gradient_penalty
                D_cost.backward()
                optimizerD.step()

            gp_sum /= (config.gammaD * lambda1 * config.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                lambda1 /= 1.1

            # Generator training 
            # Train Generator and Decoder
            for p in netD.parameters(): #freeze discrimator
                p.requires_grad = False
            if config.is_dec and config.recons_weight > 0 and config.freeze_dec:
                for p in netDec.parameters(): #freeze decoder
                    p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            netA.zero_grad()
            if config.is_dec:
                netDec.zero_grad()
                if config.is_feedback:
                    netF.zero_grad()

            # next_seen_batch
            batch_label, input_res, input_att = dataloader.next_seen_batch(config.batch_size)
            input_res = input_res.to(config.device)
            input_att = input_att.to(config.device)
            f_global = netA.extract_pooling_attributes_features(
                    input_res, config.atten_compose_type_runtime)

            means, log_var = netE(f_global, input_att)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([config.batch_size, config.latent_size]).cpu()
            eps = Variable(eps.to(config.device))
            z = eps * std + means #torch.Size([64, 312])
            if config.is_dec and config.is_feedback and loop == 1:
                recon_x = netG(z, c=input_att)
                dec_out = netDec(recon_x)
                dec_hidden_feat = netDec.getLayersOutDet()
                feedback_out = netF(dec_hidden_feat)
                recon_x = netG(z, a1=config.a1, c=input_att, feedback_layers=feedback_out)
            else:
                recon_x = netG(z, c=input_att)
            # recon_x = netG(z, c=input_att)

            # minimize E 3 with this setting feedback will update the loss as well
            vae_loss_seen = utils.loss_fn(recon_x, f_global, means, log_var)
            errG = config.vae_loss_weight * vae_loss_seen

            if config.is_rec_cls_loss:
                netA_package_ref = {}
                netA_package_ref['S_pp'] = ts_model.ft_compose_cls(f_global)
                netA_package_ref['batch_label'] = batch_label
                loss_rec_ce_ref = netA.compute_CE_loss_conservative(netA_package_ref)['loss_CE']
                netA_package = {}
                netA_package['S_pp'] = ts_model.ft_compose_cls(recon_x)
                netA_package['batch_label'] = batch_label
                loss_rec_ce = netA.compute_CE_loss_conservative(netA_package)['loss_CE']
                errG += config.rec_seen_cls_weight * loss_rec_ce
            if config.is_gen_unseen_cls_loss:
                unseen_labels = ts_model.netA.unseenclass[
                    np.random.randint(ts_model.netA.unseenclass.size(0),size=batch_label.size(0))]
                att_unseen = dataloader.attribute[unseen_labels]
                noise.normal_(0, 1)
                noisev = Variable(noise)
                gen_unseen = netG(noisev, c=att_unseen)
                netA_package_gen = {}
                netA_package_gen['S_pp'] = ts_model.ft_compose_cls(gen_unseen)
                netA_package_gen['batch_label'] = unseen_labels
                loss_gen_unseen_ce = netA.compute_CE_loss_conservative(netA_package_gen)['loss_CE']
                errG += config.gen_unseen_cls_weight * loss_gen_unseen_ce
            if config.is_gen_seen_cls_loss:
                pass
            
            if config.encoded_noise:
                criticG_fake = netD(recon_x,input_att).mean()
                fake = recon_x 
            else:
                noise.normal_(0, 1)
                noisev = Variable(noise)
                if config.is_dec and config.is_feedback and loop == 1:
                    fake = netG(noisev, c=input_att)
                    dec_out = netDec(recon_x) #Feedback from Decoder encoded output
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake = netG(noisev, a1=config.a1, c=input_att, feedback_layers=feedback_out)
                else:
                    fake = netG(noisev, c=input_att)
                # fake = netG(noisev, c=input_att)
                criticG_fake = netD(fake,input_att).mean()

            G_cost = - criticG_fake
            errG += config.gammaG * G_cost    
            if config.is_dec:
                netDec.zero_grad()
                recons_fake = netDec(fake)
                R_cost = utils.WeightedL1(recons_fake, input_att)
                errG += config.recons_weight * R_cost

            if config.is_dec and config.dec_unseen:
                unseen_labels = ts_model.netA.unseenclass[
                    np.random.randint(ts_model.netA.unseenclass.size(0),
                                        size=batch_label.size(0))]
                att_unseen = dataloader.attribute[unseen_labels]
                noise.normal_(0, 1)
                noisev = Variable(noise)
                gen_unseen = netG(noisev, c=att_unseen)
                recons_att_unseen = netDec(gen_unseen)
                R_cost_unseen = utils.WeightedL1(recons_att_unseen, att_unseen)
                errG += config.recons_weight_unseen * R_cost_unseen

            errG.backward()
            # write a condition here
            optimizerE.step()
            optimizerG.step()
            # optimizerA.step()
            # not train decoder at feedback time
            if config.is_dec and config.is_feedback and loop == 1:
                optimizerF.step()
            if config.is_dec and config.recons_weight > 0 and not config.freeze_dec:
                optimizerDec.step() 


    if epoch % config.test_freq_epoch != 0:
        continue
        
    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f, ' % (
        epoch, config.nepoch, D_cost.data.item(), G_cost.data.item(),
        Wasserstein_D.data.item(), vae_loss_seen.data.item()), end=" ")
    if config.is_rec_cls_loss:
        print('loss_ref_ce: %.4f, ' % (loss_rec_ce_ref.item()), end='')
        print('loss_rec_ce: %.4f, ' % (loss_rec_ce.item()), end='')
    if config.is_gen_unseen_cls_loss:
        print('loss_gen_unseen_ce: %.4f, ' % (loss_gen_unseen_ce.item()), end='')
    if config.is_gen_seen_cls_loss:
        pass

    netG.eval()
    netA.eval()
    if config.is_dec:
        netDec.eval()
        if config.is_feedback:
            netF.eval()
    
    syn_feature, syn_label = utils.generate_syn_feature(
        config, netG, dataloader.unseenclasses, dataloader.attribute, config.syn_num,
        netF=netF if config.is_dec and config.is_feedback else None,
        netDec=netDec if config.is_dec and config.is_feedback else None)
    # Generalized zero-shot learning
    if config.gzsl:   
        # Concatenate real seen features with synthesized unseen features
        # (train_att_feature,
        #  test_seen_att_feature,
        #  test_unseen_att_feature) = utils.inference_atten_f(config, netA, dataloader)
            
        train_X = torch.cat((train_att_feature, syn_feature), 0)
        train_Y = torch.cat((dataloader.train_label, syn_label), 0)
        nclass = config.nclass_all
        # Train GZSL classifier
        gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, test_seen_att_feature,
                                        test_unseen_att_feature, dataloader, nclass,
                                        config.cuda, config.classifier_lr, 0.5, 25,
                                        config.syn_num, generalized=True, 
                                        netDec=netDec if config.is_dec else None,
                                        dec_size=config.attSize, dec_hidden_size=4096)
        if best_gzsl_acc < gzsl_cls.H:
            (best_acc_seen, 
            best_acc_unseen, 
            best_gzsl_acc) = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H

        print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen,
                                                        gzsl_cls.acc_unseen, 
                                                        gzsl_cls.H), 
                                                        end=" ")
    # Zero-shot learning # Train ZSL classifier
    zsl_cls = classifier.CLASSIFIER(syn_feature,
                                    util.map_label(syn_label, dataloader.unseenclasses.cpu()),
                                    test_seen_att_feature, test_unseen_att_feature, 
                                    dataloader, dataloader.unseenclasses.size(0),
                                    config.cuda, config.classifier_lr, 0.5, 25,
                                    config.syn_num, generalized=False, 
                                    netDec=netDec if config.is_dec else None,
                                    dec_size=config.attSize, dec_hidden_size=4096)
    acc = zsl_cls.acc
    if best_zsl_acc < acc:
        best_zsl_acc = acc
    print('ZSL: unseen accuracy=%.4f' % (acc))

    # reset G to training mode
    netG.train()
    netA.train()
    if config.is_dec:
        netDec.train()
        if config.is_feedback:
            netF.train()

    wandb.log({'epoch': epoch,
               'acc_unseen': gzsl_cls.acc_unseen,
               'acc_seen': gzsl_cls.acc_seen,
               'H': gzsl_cls.H,
               'acc_zs': zsl_cls.acc,
               'best_acc_unseen': best_acc_unseen,
               'best_acc_seen': best_acc_seen,
               'best_H': best_gzsl_acc,
               'best_acc_zs': best_zsl_acc})


print('Dataset', config.dataset)
print('the best ZSL unseen accuracy is', best_zsl_acc)
if config.gzsl:
    print('Dataset', config.dataset)
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)

exit(0)
