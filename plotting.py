import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch


def make_video_from_np_array(arr):
    pass


def cutoff_image(im, cutoff, minmax=[-0.5, 0.5]):
    min_val, max_val = minmax
    im_co = torch.zeros_like(im)
    im_co[im > cutoff] = max_val
    im_co[im < cutoff] = min_val
    return im_co


def ml_to_sl(x):
    # converts multi lane image to single lane image (used by FC Classification model).
    return x[:, 0, -1]


def sl_to_ml(x, shape=(1, 4)):
    # converts single lane image (used by FC Classification model) to multi lane image for plotting or comparing.
    dst = torch.zeros((x.size(0),) + shape + (x.size(-1),))
    dst[:, 0, -1] = x
    return dst


def plt_imshow(im):
    plt.imshow(im, interpolation='nearest', aspect='auto', cmap='gray')


def visualize_recons(sx, sy, sphases, y_pred, t, cutoff=0):
    count = 5
    obs = sx[:, 0].cpu()
    phase = sphases.unsqueeze(1).cpu()
    y = sy[:, 0].cpu()
    y_pred_plot = y_pred[:, 0].detach().cpu()
    
    y_pred_plot2 = torch.zeros_like(y_pred_plot)
    y_pred_plot2[y_pred_plot > cutoff] = 0.5
    y_pred_plot2[y_pred_plot < cutoff] = -0.5
    
    fig = plt.figure(figsize=(15, 10))

    fig.add_subplot(count, 1, 1)
    plt.imshow(obs[t], interpolation='nearest', aspect='auto', cmap='gray')
    
    fig.add_subplot(count, 1, 2)
    plt.imshow(phase[t], interpolation='nearest', aspect='auto', cmap='gray')
    
    fig.add_subplot(count, 1, 3)
    plt.imshow(y[t], interpolation='nearest', aspect='auto', cmap='gray')
    
    fig.add_subplot(count, 1, 4)
    plt.imshow(y_pred_plot[t], interpolation='nearest', aspect='auto', cmap='gray')
    
    fig.add_subplot(count, 1, 5)
    plt.imshow(y_pred_plot2[t], interpolation='nearest', aspect='auto', cmap='gray')
    
    plt.show()


def visualize_recons_multi(sx, sy, sphases, y_pred, y_pred2, t, cutoff=0, only_cutoff=False):
    count = 5 if only_cutoff else 7
    obs = sx[:, 0].cpu()
    phase = sphases.unsqueeze(1).cpu()
    y = sy[:, 0].cpu()
    cutoff1, cutoff2 = cutoff if isinstance(cutoff, (list, tuple)) else (cutoff, cutoff)

    y_pred_pl = y_pred[:, 0].detach().cpu()
    y_pred_pl_co = cutoff_image(y_pred_pl, cutoff1)

    y_pred_pl2 = y_pred2[:, 0].detach().cpu()
    y_pred_pl2_co = cutoff_image(y_pred_pl2, cutoff2)

    fig = plt.figure(figsize=(15, count * 2))
    plt_idx = 1

    fig.add_subplot(count, 1, plt_idx)
    plt.imshow(obs[t], interpolation='nearest', aspect='auto', cmap='gray')
    plt_idx += 1

    fig.add_subplot(count, 1, plt_idx)
    plt.imshow(phase[t], interpolation='nearest', aspect='auto', cmap='gray')
    plt_idx += 1

    fig.add_subplot(count, 1, plt_idx)
    plt.imshow(y[t], interpolation='nearest', aspect='auto', cmap='gray')
    plt_idx += 1

    if not only_cutoff:
        fig.add_subplot(count, 1, plt_idx)
        plt.imshow(y_pred_pl[t], interpolation='nearest', aspect='auto', cmap='gray')
        plt_idx += 1

    fig.add_subplot(count, 1, plt_idx)
    plt.imshow(y_pred_pl_co[t], interpolation='nearest', aspect='auto', cmap='gray')
    plt_idx += 1

    if not only_cutoff:
        fig.add_subplot(count, 1, plt_idx)
        plt.imshow(y_pred_pl2[t], interpolation='nearest', aspect='auto', cmap='gray')
        plt_idx += 1

    fig.add_subplot(count, 1, plt_idx)
    plt.imshow(y_pred_pl2_co[t], interpolation='nearest', aspect='auto', cmap='gray')
    plt_idx += 1

    plt.show()


class Index():
    def __init__(self, data, plot):
        self.idx = 0
        self.data = data
        self.plot = plot

    def next(self, event):
        if self.idx + 1 >= len(self.data): return
        self.idx += 1
        self.plot.set_data(self.data[self.idx])
        plt.draw()

    def prev(self, event):
        if self.idx - 1 < 0: return
        self.idx -= 1
        self.plot.set_data(self.data[self.idx])
        plt.draw()
    
    def store_buttons(self, *buttons):
        self.buttons = []
        self.buttons.extend(buttons)


def show_interactive_rollout(rollout_list):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    image = plt.imshow(rollout_list[0], interpolation='nearest', aspect='auto', cmap='gray', vmin=0, vmax=1)

    callback = Index(rollout_list, image)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Previous')
    bnext.on_clicked(callback.next)
    bprev.on_clicked(callback.prev)
    callback.store_buttons(bnext, bprev)