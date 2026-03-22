import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft



class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6): 
        super().__init__(); 
        self.eps=eps

    def forward(self, logits, target):
        if logits.shape[1] == 1:
            p = torch.sigmoid(logits)
            t = target.float().unsqueeze(1)
            inter = (p * t).sum(dim=(2,3))
            den = (p + t).sum(dim=(2,3))
            dice = (2 * inter + self.eps) / (den + self.eps)
            return 1 - dice.mean()
        else:
            p = F.softmax(logits,dim=1)
            t = F.one_hot(target,logits.shape[1]).permute(0,3,1,2).float()
            inter = (p * t).sum(dim=(2,3))
            den = (p + t).sum(dim=(2,3))
            dice = (2 * inter + self.eps) / (den + self.eps)
            return 1 - dice.mean()



class FocalFrequencyLoss(nn.Module):
    """
    Optimizes the frequency domain to recover sharp edges (high frequencies).
    """
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # 2D RFFT (Real Fast Fourier Transform)
        freq = torch.fft.rfft2(x, norm='ortho')
        return torch.stack([freq.real, freq.imag], -1)

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            # (B, C, H, W, 2)
            matrix_tmp = (recon_freq - real_freq) ** 2
            
            # Collapse complex dim -> (B, C, H, W)
            # This is the squared magnitude |diff|^2
            matrix_tmp = matrix_tmp[..., 0] + matrix_tmp[..., 1]

            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # Focal formulation: weight = |diff|^alpha
            # We currently have |diff|^2, so we raise to (alpha/2)
            weight_matrix = torch.clamp(matrix_tmp, min=1e-8, max=float('inf'))
            weight_matrix = weight_matrix ** (self.alpha / 2.0)

        weight_matrix = weight_matrix.unsqueeze(-1)

        loss = torch.mean(weight_matrix * (recon_freq - real_freq) ** 2)
        return loss

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        return self.loss_weight * self.loss_formulation(pred_freq, target_freq)



class MorphologicalLoss(nn.Module):
    """
    Optimizes the structural topology using differentiable erosion/dilation.
    """
    def __init__(self, kernel_size=5):
        super(MorphologicalLoss, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def erosion(self, x):
        return -F.max_pool2d(-x, kernel_size=self.kernel_size, stride=1, padding=self.padding)

    def dilation(self, x):
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)

    def forward(self, pred, target):
        pred_eroded = self.erosion(pred)
        target_eroded = self.erosion(target)
        loss_erosion = F.mse_loss(pred_eroded, target_eroded)
        
        pred_dilated = self.dilation(pred)
        target_dilated = self.dilation(target)
        loss_dilation = F.mse_loss(pred_dilated, target_dilated)
        
        return loss_erosion + loss_dilation

        

class TDFLoss(nn.Module):
    def __init__(self, lambda_base=1.0, lambda_focal=0.1, lambda_morph=0.5, num_classes=1):
        super(TDFLoss, self).__init__()
        self.lambda_base = lambda_base
        self.lambda_focal = lambda_focal
        self.lambda_morph = lambda_morph
        self.num_classes = num_classes

        self.dice = DiceLoss() 
        self.focal_freq = FocalFrequencyLoss()
        self.morph = MorphologicalLoss(kernel_size=5)
        self.ce = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        # 1. Base Loss (Dice + CE)
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
            target_float = target.float().unsqueeze(1) if target.dim() == 3 else target.float()
            loss_base = self.lambda_base * (self.dice(probs, target) + self.ce(logits, target_float))

        else:
            probs = F.softmax(logits, dim=1)
            target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
            loss_base = self.lambda_base * (self.dice(probs, target) + self.ce(logits, target.long()))
            target_float = target_one_hot

        # 2. Spectral Loss (on Probabilities)
        if self.num_classes == 1:
            loss_spec = self.focal_freq(probs, target_float)
        else:
            loss_spec = 0
            for c in range(self.num_classes):
                loss_spec += self.focal_freq(probs[:, c:c+1], target_float[:, c:c+1])
            loss_spec /= self.num_classes

        # 3. Morphological Loss (on Probabilities)
        if self.num_classes == 1:
            loss_morph = self.morph(probs, target_float)
        else:
            loss_morph = 0
            for c in range(self.num_classes):
                loss_morph += self.morph(probs[:, c:c+1], target_float[:, c:c+1])
            loss_morph /= self.num_classes

        return loss_base + (self.lambda_focal * loss_spec) + (self.lambda_morph * loss_morph)