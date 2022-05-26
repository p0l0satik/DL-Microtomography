import os

val_pics = sorted(os.listdir('/trinity/home/a.razorenova/lab/gpfs/a.razorenova/Microtomo/compare_data/test4k/scans')) # путь до сканов
val_strs = sorted(os.listdir('/trinity/home/a.razorenova/lab/gpfs/a.razorenova/Microtomo/compare_data/test4k/struct')) # путь до структур

def pic_saver(pred, gt, name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0][0].imshow(pred[0].detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=15)
    axes[0][0].set_title('Prediction for Au layer')
    axes[0][1].imshow(gt[0].detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=15)
    axes[0][1].set_title('Ground truth for Au layer')

    axes[1][0].imshow(pred[1].detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=150)
    axes[1][0].set_title('Prediction for Al layer')
    axes[1][1].imshow(gt[1].detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=150)
    axes[1][1].set_title('Ground truth for Al layer')

    fig.colorbar(axes[0][1].imshow(gt[0].T.detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=15), ax=axes[0], fraction=0.0213)
    fig.colorbar(axes[1][1].imshow(gt[1].T.detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=150), ax=axes[1], fraction=0.0213)
    
    fig.savefig('val_pics/'+name+'.png')
    plt.close(fig)
    
 ! mkdir val_pics
    
 for pic, struct in zip(val_pics, val_strs):
    scan = np.load('/trinity/home/a.razorenova/lab/gpfs/a.razorenova/Microtomo/compare_data/test4k/scans/' + pic)
    scan = torch.unsqueeze(torch.permute(torch.from_numpy(scan), (2, 0, 1)), dim=0).float()
    pred = torch.squeeze(unet(scan.cuda()))
    gt = torch.from_numpy(np.load('/trinity/home/a.razorenova/lab/gpfs/a.razorenova/Microtomo/compare_data/test4k/struct/' + struct))
    pic_saver(pred, gt, pic[5:-4])
    np.save('val_pics/'+pic[5:-4]+'.npy', pred.detach().cpu())
    
 ! zip val_pics val_pics/*
    
 
