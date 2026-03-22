import torch
import numpy as np



def iou_score(logits, target, eps=1e-6):
    if logits.shape[1] == 1:
        p = (torch.sigmoid(logits) > 0.5).long()
        t = target.unsqueeze(1)
        inter = ((p == 1) & (t == 1)).sum(dim=(2,3)).float()
        union = ((p == 1) | (t == 1)).sum(dim=(2,3)).float()
        iou = ((inter + eps) / (union + eps)).mean().item()
        return iou
    else:
        p = logits.argmax(dim=1)
        ious = []
        C = logits.shape[1]
        for c in range(C):
            pc = (p == c)
            tc = (target == c)
            inter = (pc & tc).sum().float()
            union = (pc | tc).sum().float()
            if union == 0: 
                continue
            ious.append(((inter + eps) / (union + eps)).item())
        return float(np.mean(ious)) if ious else 0.0



def dice_score(logits, target, eps=1e-6):
    if logits.shape[1] == 1:
        p = (torch.sigmoid(logits) > 0.5).long()
        t = target.unsqueeze(1)
        inter = ((p == 1) & (t == 1)).sum(dim=(2,3)).float()
        den = (p.sum(dim=(2,3)).float() + t.sum(dim=(2,3)).float())
        return ((2 * inter + eps) / (den + eps)).mean().item()
    else:
        p = logits.argmax(dim=1)
        dices = []
        C = logits.shape[1]
        for c in range(C):
            pc = (p == c)
            tc = (target == c)
            inter = (pc & tc).sum().float()
            den = pc.sum().float() + tc.sum().float()
            if den == 0: 
                continue
            dices.append(((2 * inter + eps) / (den + eps)).item())
        return float(np.mean(dices)) if dices else 0.0
    
    

def calculate_metrics(logits, target, eps=1e-6):
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        t = target.unsqueeze(1)
    else:
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).unsqueeze(1)
        t = target.unsqueeze(1)

    tp = ((preds == 1) & (t == 1)).sum().float()
    fp = ((preds == 1) & (t == 0)).sum().float()
    fn = ((preds == 0) & (t == 1)).sum().float()
    tn = ((preds == 0) & (t == 0)).sum().float()

    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    return accuracy.item(), precision.item(), recall.item()