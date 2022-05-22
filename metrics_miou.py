# pred_3d and orig_3d are of size (batch_size, 16, 128, 128) if not - do unsqueeze
def calc_val_data(pred_3d, orig_3d, n_cls=3):
    from torch import logical_and as LAND
    from torch import logical_or as LOR
    
    bat_size = preds.shape[0]

    intersection = torch.tensor([[torch.sum(LAND((pred_3d[b]==i),(orig_3d[b]==i))) for i in range(n_cls)] for b in range(bat_size)])
    union =        torch.tensor([[torch.sum(LOR( (pred_3d[b]==i),(orig_3d[b]==i))) for i in range(n_cls)] for b in range(bat_size)])
    target =       torch.tensor([[torch.sum(orig_3d[b]==i) for i in range(n_cls)] for b in range(bat_size)])
    
    # Output shapes: batch_size x num_classes
    return intersection, union, target

def calc_val_loss(intersection, union, target, eps = 1e-7):

    mean_iou = torch.mean((intersection+eps)/(union+eps))
    mean_class_rec = torch.mean((intersection+eps)/(target+eps))
    mean_acc = torch.nansum(intersection)/torch.nansum(target)

    return mean_iou, mean_class_rec, mean_acc
