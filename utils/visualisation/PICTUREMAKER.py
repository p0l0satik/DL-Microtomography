# preds[0][0] - предсказание для золота, preds[0][1] - предсказание для алюминия, с таргетом то же самое

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0][0].imshow(preds[0][0].detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=15)
axes[0][0].set_title('Prediction for Au layer')
axes[0][1].imshow(target[0][0].detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=15)
axes[0][1].set_title('Ground truth for Au layer')

axes[1][0].imshow(preds[0][1].detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=150)
axes[1][0].set_title('Prediction for Al layer')
axes[1][1].imshow(target[0][1].detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=150)
axes[1][1].set_title('Ground truth for Al layer')

fig.colorbar(axes[0][1].imshow(preds[0][0].detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=15), ax=axes[0], fraction=0.0213)
fig.colorbar(axes[1][1].imshow(target[0][1].detach().cpu().numpy(), interpolation='nearest', origin='lower', vmin=0, vmax=150), ax=axes[1], fraction=0.0213)
