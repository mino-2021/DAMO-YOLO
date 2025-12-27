"""
Plot training loss curves from DAMO-YOLO log files
"""

import argparse
import re
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log_file(log_path):
    """Parse training log and extract loss values"""

    pattern = r'epoch: (\d+)/\d+.*total_loss: ([\d.]+), loss_cls: ([\d.]+), loss_bbox: ([\d.]+), loss_dfl: ([\d.]+)'

    data = {
        'epoch': [],
        'total_loss': [],
        'loss_cls': [],
        'loss_bbox': [],
        'loss_dfl': []
    }

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                total_loss = float(match.group(2))
                loss_cls = float(match.group(3))
                loss_bbox = float(match.group(4))
                loss_dfl = float(match.group(5))

                data['epoch'].append(epoch)
                data['total_loss'].append(total_loss)
                data['loss_cls'].append(loss_cls)
                data['loss_bbox'].append(loss_bbox)
                data['loss_dfl'].append(loss_dfl)

    return data


def average_by_epoch(data):
    """Average losses per epoch"""
    from collections import defaultdict

    epoch_data = defaultdict(lambda: {'total_loss': [], 'loss_cls': [], 'loss_bbox': [], 'loss_dfl': []})

    for i, epoch in enumerate(data['epoch']):
        epoch_data[epoch]['total_loss'].append(data['total_loss'][i])
        epoch_data[epoch]['loss_cls'].append(data['loss_cls'][i])
        epoch_data[epoch]['loss_bbox'].append(data['loss_bbox'][i])
        epoch_data[epoch]['loss_dfl'].append(data['loss_dfl'][i])

    result = {'epoch': [], 'total_loss': [], 'loss_cls': [], 'loss_bbox': [], 'loss_dfl': []}

    for epoch in sorted(epoch_data.keys()):
        result['epoch'].append(epoch)
        result['total_loss'].append(sum(epoch_data[epoch]['total_loss']) / len(epoch_data[epoch]['total_loss']))
        result['loss_cls'].append(sum(epoch_data[epoch]['loss_cls']) / len(epoch_data[epoch]['loss_cls']))
        result['loss_bbox'].append(sum(epoch_data[epoch]['loss_bbox']) / len(epoch_data[epoch]['loss_bbox']))
        result['loss_dfl'].append(sum(epoch_data[epoch]['loss_dfl']) / len(epoch_data[epoch]['loss_dfl']))

    return result


def plot_loss_curves(data, output_path='training_loss.png', title='Training Loss'):
    """Plot loss curves"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Total Loss
    axes[0, 0].plot(data['epoch'], data['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Classification Loss
    axes[0, 1].plot(data['epoch'], data['loss_cls'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # BBox Loss
    axes[1, 0].plot(data['epoch'], data['loss_bbox'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('BBox Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # DFL Loss
    axes[1, 1].plot(data['epoch'], data['loss_dfl'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('DFL Loss')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved loss plot to: {output_path}")
    plt.close()


def plot_combined_loss(data, output_path='training_loss_combined.png', title='Training Loss'):
    """Plot all losses in one graph"""

    plt.figure(figsize=(10, 6))

    plt.plot(data['epoch'], data['total_loss'], 'b-', linewidth=2, label='Total Loss')
    plt.plot(data['epoch'], data['loss_cls'], 'g-', linewidth=2, label='Cls Loss')
    plt.plot(data['epoch'], data['loss_bbox'], 'r-', linewidth=2, label='BBox Loss')
    plt.plot(data['epoch'], data['loss_dfl'], 'm-', linewidth=2, label='DFL Loss')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved combined loss plot to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss from log file")
    parser.add_argument('--log', '-l', type=str, required=True, help='Log file path')
    parser.add_argument('--output', '-o', type=str, default='training_loss.png', help='Output image path')
    parser.add_argument('--title', '-t', type=str, default='Training Loss', help='Plot title')
    parser.add_argument('--combined', action='store_true', help='Plot all losses in one graph')

    args = parser.parse_args()

    print(f"Parsing log file: {args.log}")
    data = parse_log_file(args.log)
    print(f"Found {len(data['epoch'])} log entries")

    # Average by epoch
    data = average_by_epoch(data)
    print(f"Averaged to {len(data['epoch'])} epochs")

    if args.combined:
        plot_combined_loss(data, args.output, args.title)
    else:
        plot_loss_curves(data, args.output, args.title)
