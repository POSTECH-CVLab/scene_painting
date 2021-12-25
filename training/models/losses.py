import torch
import torch.nn.functional as F
import torch.nn as nn
from models.vggloss import VGG19


class losses_computer():
    def __init__(self, opt):
        self.opt = opt
        if not opt.no_labelmix:
            self.labelmix_function = torch.nn.MSELoss()

        self.loss_type_binary = opt.loss_binary

        self.margin = opt.margin
        self.gamma = opt.gamma

    def omni_loss(self, pred, one_hot_label, loss_type):
        one_hot_label = F.interpolate(one_hot_label, scale_factor=0.5, mode='nearest') 

        b, nc_plus_2, h, w = pred.shape
        if loss_type == 'g_loss_fake':
            dummy = torch.zeros((b, 2, h, w), dtype=torch.int64, device=pred.device)
            dummy[:,0,:,:] = 1
            label = torch.cat([one_hot_label, dummy], dim=1)
        elif loss_type == 'd_loss_fake':
            dummy = torch.zeros((b, nc_plus_2, h, w), dtype=torch.int64, device=pred.device)
            dummy[:,-1,:,:] = 1
            label = dummy
        elif loss_type == 'd_loss_real':
            dummy = torch.zeros((b, 2, h, w), dtype=torch.int64, device=pred.device)
            dummy[:,0,:,:] = 1
            label = torch.cat([one_hot_label, dummy], dim=1)

        label[:,0,:,:] = -1 
        """
        label: positive=1, negative=0, ignore=-1
        """

        pred = pred + self.margin
        pred = pred * self.gamma

        pred[label == 1] = -1 * pred[label == 1]
        pred[label == -1] = -1e12

        pred_neg = pred.clone()
        pred_neg[label == 1] = -1e12

        pred_pos = pred.clone()
        pred_pos[label == 0] = -1e12

        zeros = torch.zeros_like(pred[:,:1,:,:])
        pred_neg = torch.cat([pred_neg, zeros], dim=1)
        pred_pos = torch.cat([pred_pos, zeros], dim=1)
        neg_loss = torch.logsumexp(pred_neg, dim=1, keepdim=True)
        pos_loss = torch.logsumexp(pred_pos, dim=1, keepdim=True)
        loss = torch.mean(neg_loss + pos_loss)

        return loss

    def seg_loss(self, input, label):
        weight_map = get_class_balancing(input, label, False, False)
        label_idx_map = torch.argmax(label, dim=1)
        loss = F.cross_entropy(input, label_idx_map, reduction='none')
        loss = torch.mean(loss * weight_map[:, 0, :, :])
        return loss

    def whole_hinge_loss(self, fake, loss_type):

        if loss_type == 'g_loss_fake':
            loss = -torch.mean(fake)
        elif loss_type == 'd_loss_fake':
            loss = torch.mean(F.relu(1.0 + fake))
        elif loss_type == 'd_loss_real':
            loss = torch.mean(F.relu(1.0 - fake))

        return loss

    def proj_loss(self, fake, label, loss_type):

        label = F.interpolate(label, scale_factor=0.5, mode='nearest')
        fake = torch.einsum('bchw,bchw->bhw', fake, label)

        label_float = (label[:,0] != 1).float()
        if loss_type == 'g_loss_fake':
            loss = (-fake * label_float).mean()
        elif loss_type == 'd_loss_fake':
            loss = (torch.nn.ReLU()(1.0 + fake) * label_float).mean()
        elif loss_type == 'd_loss_real':
            loss = (torch.nn.ReLU()(1.0 - fake) * label_float).mean()

        return loss
        

    def loss_multi(self, input, label, for_real, for_D=False):
        # --- balancing classes ---
        weight_map = get_class_balancing(input, label, self.opt.no_balancing_inloss, self.opt.contain_dontcare_label)
        # --- n+1 loss ---
        # Fake label: 0
        # Real label: 1 ~ 12
        target = get_n1_target(input, label, for_real)

        loss = F.cross_entropy(input, target, reduction='none')
        if for_real:
            loss = torch.mean(loss * weight_map[:, 0, :, :])
        else:
            loss = torch.mean(loss)
        return loss

    def loss_binary(self, valid=None, fake=None, for_D=False):

        if for_D:
            loss = compute_d_loss(valid, fake, self.loss_type_binary)
        else:
            loss = compute_g_loss(fake, self.loss_type_binary)

        return loss

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask * output_D_real + (1 - mask) * output_D_fake
        return self.labelmix_function(mixed_D_output, output_D_mixed)


def get_class_balancing(input, label, no_balancing_inloss=True, contain_dontcare_label=False):
    if not no_balancing_inloss:
        class_occurence = torch.sum(label, dim=(0, 2, 3))
        if contain_dontcare_label:
            class_occurence[0] = 0
        num_of_classes = (class_occurence > 0).sum()
        coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1])
        integers = torch.argmax(label, dim=1, keepdim=True)
        if contain_dontcare_label:
            coefficients[0] = 0
        weight_map = coefficients[integers]
    else:
        weight_map = torch.ones_like(input[:, :, :, :])
    return weight_map


def get_n1_target(input, label, target_is_real):
    # class index label map for real sample
    # tensor of 0 for fake sample (0th class is fake/real class)
    targets = get_target_tensor(input, target_is_real)
    num_of_classes = label.shape[1]
    integers = torch.argmax(label, dim=1)
    targets = targets[:, 0, :, :] * num_of_classes
    integers += targets.long()
    integers = torch.clamp(integers, min=num_of_classes - 1) - num_of_classes + 1
    return integers


def get_target_tensor(input, target_is_real):
    if target_is_real:
        return torch.ones_like(input, requires_grad=False)
    else:
        return torch.zeros_like(input, requires_grad=False)

def compute_g_loss(fake, loss_type):

    if loss_type == 'gan':
        valid_target = torch.ones_like(fake)
        g_loss = F.binary_cross_entropy_with_logits(fake, valid_target)
    elif loss_type == 'hinge':
        g_loss = -torch.mean(fake)
    else:
        raise ValueError()

    return g_loss

def compute_d_loss(valid, fake, loss_type):

    if loss_type == 'gan':
        valid_target = torch.ones_like(valid)
        real_loss = F.binary_cross_entropy_with_logits(valid, valid_target)

        fake_target = torch.zeros_like(fake)
        fake_loss = F.binary_cross_entropy_with_logits(fake, fake_target)

        d_loss = real_loss + fake_loss
    elif loss_type == 'hinge':
        d_loss = torch.mean(F.relu(1.0 - valid)) + torch.mean(F.relu(1.0 + fake))
    else:
        raise ValueError()

    return d_loss


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
