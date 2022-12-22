import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


def loss_fn(recon_x, x, mean, log_var):
    # BCE = torch.nn.functional.binary_cross_entropy(
    #     recon_x+1e-12, x.detach(), size_average=False)
    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x+1e-12, x.detach(), reduction='sum')
    BCE = BCE.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) -
                           log_var.exp()) / x.size(0)
    return (BCE + KLD)


def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)


@torch.no_grad()
def generate_syn_feature(opt, generator, classes, attribute, num, netF=None, netDec=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.visual_dim)
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise)
        syn_attv = Variable(syn_att)
        fake = generator(syn_noisev, c=syn_attv)
        if netF is not None:
            # only to call the forward function of decoder
            dec_out = netDec(fake)
            dec_hidden_feat = netDec.getLayersOutDet()  # no detach layers
            feedback_out = netF(dec_hidden_feat)
            fake = generator(syn_noisev, a1=opt.a2, c=syn_attv,
                             feedback_layers=feedback_out)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


@torch.no_grad()
def inference_atten_f(config, netA, dataloader):
    train_att_feature = []
    for features in dataloader.train_torch_loader:
        features = features[0].to(config.device)
        atten_f = netA.extract_pooling_attributes_features(
            features, config.atten_compose_type_runtime).cpu()
        train_att_feature.append(atten_f)
    train_att_feature = torch.cat(train_att_feature)

    test_seen_att_feature = []
    for features in dataloader.test_seen_torch_loader:
        features = features[0].to(config.device)
        atten_f = netA.extract_pooling_attributes_features(
            features, config.atten_compose_type_runtime).cpu()
        test_seen_att_feature.append(atten_f)
    test_seen_att_feature = torch.cat(test_seen_att_feature)

    test_unseen_att_feature = []
    for features in dataloader.test_unseen_torch_loader:
        features = features[0].to(config.device)
        atten_f = netA.extract_pooling_attributes_features(
            features, config.atten_compose_type_runtime).cpu()
        test_unseen_att_feature.append(atten_f)
    test_unseen_att_feature = torch.cat(test_unseen_att_feature)

    return train_att_feature, test_seen_att_feature, test_unseen_att_feature


def calc_gradient_penalty(opt, netD, real_data, fake_data, input_att, lambda1):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
    return gradient_penalty
