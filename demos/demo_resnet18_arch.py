# -*- coding: utf-8 -*-
"""
ResNet18 结构示意图生成脚本
使用 matplotlib 绘制 ResNet18 网络架构图
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os
from font_setup import setup_cjk_font

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_resnet18_arch():
    font_cjk = setup_cjk_font()
    fig, ax = plt.subplots(1, 1, figsize=(22, 12))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_facecolor('#FAFBFC')

    # 颜色方案
    COLOR_STEM = '#FF6B6B'
    COLOR_STEM_EDGE = '#E05555'
    COLOR_LAYER1 = '#4ECDC4'
    COLOR_LAYER2 = '#45B7D1'
    COLOR_LAYER3 = '#96CEB4'
    COLOR_LAYER4 = '#FFEAA7'
    COLOR_FC = '#DDA0DD'
    COLOR_ARROW = '#888888'
    COLOR_SKIP = '#FF6347'

    y_center = 6.0
    bar_height = 1.0

    def add_rect(x, w, y, h, color, edge_color, label, fontsize=8, text_color='white'):
        rect = FancyBboxPatch((x, y - h/2), w, h,
                              boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor=edge_color,
                              linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color,
                fontfamily='monospace')

    def add_output_arrow(x, w, y, label, color=COLOR_ARROW):
        ax.annotate('', xy=(x + w + 0.15, y), xytext=(x + w - 0.05, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        ax.text(x + w + 0.35, y, label, va='center', fontsize=7,
                color='#444444', fontfamily='monospace')

    # ============ 第一行：数据流 + Stem ============
    row1_y = 10.0

    # Input
    add_rect(0.3, 1.2, row1_y, 0.7, '#E8E8E8', '#CCCCCC',
             'Input\n3x160x120', fontsize=7, text_color='#333')
    add_output_arrow(0.3, 1.2, row1_y, '(3,160,120)')

    # Conv1 7x7
    add_rect(2.0, 2.2, row1_y, 0.8, COLOR_STEM, COLOR_STEM_EDGE,
             'Conv1 7x7,64 /2\nBN + ReLU', fontsize=8)
    add_output_arrow(2.0, 2.2, row1_y, '(64,80,60)')

    # MaxPool (DonkeyCar uses maxpool implicitly in stem)
    add_rect(4.6, 1.6, row1_y, 0.7, '#FF9F43', '#E08830',
             'MaxPool 3x3 /2', fontsize=8)
    add_output_arrow(4.6, 1.6, row1_y, '(64,40,30)')

    # ============ 第二行：Layer1 ============
    row2_y = 8.0

    # Layer1 标题
    ax.text(0.3, row2_y + 0.5, 'Layer1', fontsize=10, fontweight='bold',
            color=COLOR_LAYER1, va='center', fontfamily='monospace')

    # Block1
    add_rect(1.2, 2.0, row2_y, 0.7, COLOR_LAYER1, '#3AAFA9',
             'Conv 3x3,64\nConv 3x3,64', fontsize=7)

    # skip
    ax.annotate('', xy=(2.2, row2_y + 0.6), xytext=(2.2, row1_y - 0.4),
                arrowprops=dict(arrowstyle='->', color=COLOR_SKIP, lw=1.2,
                               connectionstyle="arc3,rad=0.3", linestyle='dashed'))
    add_output_arrow(1.2, 2.0, row2_y, '(64,40,30)', color=COLOR_LAYER1)

    # Block2
    add_rect(3.6, 2.0, row2_y, 0.7, COLOR_LAYER1, '#3AAFA9',
             'Conv 3x3,64\nConv 3x3,64', fontsize=7)
    add_output_arrow(3.6, 2.0, row2_y, '(64,40,30)', color=COLOR_LAYER1)

    ax.text(5.8, row2_y, '+', fontsize=20, color=COLOR_ARROW, va='center', ha='center')

    # ============ 第三行：Layer2 ============
    row3_y = 6.0

    ax.text(0.3, row3_y + 0.5, 'Layer2', fontsize=10, fontweight='bold',
            color=COLOR_LAYER2, va='center', fontfamily='monospace')

    # downsample arrow
    ax.annotate('', xy=(1.2 + 1.0, row3_y + 0.6), xytext=(3.6 + 1.0, row2_y - 0.4),
                arrowprops=dict(arrowstyle='->', color=COLOR_SKIP, lw=1.2,
                               connectionstyle="arc3,rad=-0.3", linestyle='dashed'))
    ax.text(3.2, row2_y - 0.8, '/2', fontsize=7, color=COLOR_SKIP, ha='center')

    add_rect(1.2, 2.0, row3_y, 0.7, COLOR_LAYER2, '#3896A8',
             'Conv 3x3,128 /2\nConv 3x3,128', fontsize=7)
    add_output_arrow(1.2, 2.0, row3_y, '(128,20,15)', color=COLOR_LAYER2)

    add_rect(3.6, 2.0, row3_y, 0.7, COLOR_LAYER2, '#3896A8',
             'Conv 3x3,128\nConv 3x3,128', fontsize=7)
    add_output_arrow(3.6, 2.0, row3_y, '(128,20,15)', color=COLOR_LAYER2)

    ax.text(5.8, row3_y, '+', fontsize=20, color=COLOR_ARROW, va='center', ha='center')

    # ============ 第四行：Layer3 ============
    row4_y = 4.0

    ax.text(0.3, row4_y + 0.5, 'Layer3', fontsize=10, fontweight='bold',
            color=COLOR_LAYER3, va='center', fontfamily='monospace')

    ax.annotate('', xy=(1.2 + 1.0, row4_y + 0.6), xytext=(3.6 + 1.0, row3_y - 0.4),
                arrowprops=dict(arrowstyle='->', color=COLOR_SKIP, lw=1.2,
                               connectionstyle="arc3,rad=-0.3", linestyle='dashed'))

    add_rect(1.2, 2.0, row4_y, 0.7, COLOR_LAYER3, '#7AAE96',
             'Conv 3x3,256 /2\nConv 3x3,256', fontsize=7)
    add_output_arrow(1.2, 2.0, row4_y, '(256,10,8)', color=COLOR_LAYER3)

    add_rect(3.6, 2.0, row4_y, 0.7, COLOR_LAYER3, '#7AAE96',
             'Conv 3x3,256\nConv 3x3,256', fontsize=7)
    add_output_arrow(3.6, 2.0, row4_y, '(256,10,8)', color=COLOR_LAYER3)

    ax.text(5.8, row4_y, '+', fontsize=20, color=COLOR_ARROW, va='center', ha='center')

    # ============ 第五行：Layer4 ============
    row5_y = 2.0

    ax.text(0.3, row5_y + 0.5, 'Layer4', fontsize=10, fontweight='bold',
            color=COLOR_LAYER4, va='center', fontfamily='monospace')

    ax.annotate('', xy=(1.2 + 1.0, row5_y + 0.6), xytext=(3.6 + 1.0, row4_y - 0.4),
                arrowprops=dict(arrowstyle='->', color=COLOR_SKIP, lw=1.2,
                               connectionstyle="arc3,rad=-0.3", linestyle='dashed'))

    add_rect(1.2, 2.0, row5_y, 0.7, COLOR_LAYER4, '#DDC056',
             'Conv 3x3,512 /2\nConv 3x3,512', fontsize=7)
    add_output_arrow(1.2, 2.0, row5_y, '(512,5,4)', color='#B8A040')

    add_rect(3.6, 2.0, row5_y, 0.7, COLOR_LAYER4, '#DDC056',
             'Conv 3x3,512\nConv 3x3,512', fontsize=7)
    add_output_arrow(3.6, 2.0, row5_y, '(512,5,4)', color='#B8A040')

    ax.text(5.8, row5_y, '+', fontsize=20, color=COLOR_ARROW, va='center', ha='center')

    # ============ 第六行：分类头 ============
    row6_y = 0.6

    # GAP
    add_rect(0.3, 1.5, row6_y, 0.6, '#95E1D3', '#6EB5A8',
             'GAP', fontsize=8, text_color='#333')
    add_output_arrow(0.3, 1.5, row6_y, '(512)', color=COLOR_ARROW)

    # FC
    add_rect(2.2, 1.5, row6_y, 0.6, COLOR_FC, '#B070B0',
             'FC (512x1)', fontsize=8)
    add_output_arrow(2.2, 1.5, row6_y, 'Output\n(1)', color=COLOR_ARROW)

    # ============ 图例 ============
    ax.text(7.5, 10.5, '图例', fontsize=10, fontweight='bold', color='#333')

    legends = [
        (COLOR_STEM, 'Stem 层'),
        (COLOR_LAYER1, 'Layer1 (64ch)'),
        (COLOR_LAYER2, 'Layer2 (128ch)'),
        (COLOR_LAYER3, 'Layer3 (256ch)'),
        (COLOR_LAYER4, 'Layer4 (512ch)'),
        (COLOR_FC, '全连接层'),
    ]
    for i, (c, label) in enumerate(legends):
        ly = 9.8 - i * 0.45
        rect = FancyBboxPatch((7.5, ly - 0.15), 0.6, 0.3,
                              boxstyle="round,pad=0.05",
                              facecolor=c, edgecolor='gray', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(8.4, ly, label, fontsize=8, va='center', color='#444')

    # 跳跃连接说明
    ax.annotate('', xy=(7.5, 7.2), xytext=(8.1, 7.2),
                arrowprops=dict(arrowstyle='->', color=COLOR_SKIP, lw=1.2,
                               linestyle='dashed'))
    ax.text(8.4, 7.2, '残差跳跃连接', fontsize=8, va='center', color='#444')

    # 右侧统计信息
    ax.text(7.5, 6.5, '参数统计', fontsize=10, fontweight='bold', color='#333')
    stats = [
        '总层数: 18 Conv + 1 FC',
        '总参数量: ~11.2M (标准)',
        '输入: 3 x 160 x 120',
        '输出: 1 (转向角)',
        '每个 Block: 2x Conv3x3',
        '短接方式: 逐元素相加',
        '下采样: stride=2',
    ]
    for i, s in enumerate(stats):
        ax.text(7.5, 5.9 - i * 0.35, '• ' + s, fontsize=7.5, va='center',
                color='#555555', fontfamily='monospace')

    # 标题
    ax.text(10.5, 11.3, 'ResNet18 网络架构 (DonkeyCar 自动驾驶)',
            fontsize=16, fontweight='bold', color='#222222', ha='center',
            fontfamily='monospace')
    ax.text(10.5, 10.9, 'ResNet18AutoDrive — 回归任务，输出单个转向角',
            fontsize=9, color='#888888', ha='center',
            fontfamily='monospace')

    plt.tight_layout(pad=0.5)
    save_path = os.path.join(OUTPUT_DIR, 'resnet18_architecture.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] ResNet18 结构示意图已保存: {save_path}")


if __name__ == '__main__':
    draw_resnet18_arch()
