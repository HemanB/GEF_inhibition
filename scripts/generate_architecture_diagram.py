#!/usr/bin/env python3
"""Publication-quality pipeline diagram with embedded structure rendering."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
})

struct_img = mpimg.imread('/cwork/hsb26/GEF_inhibition/results/figures/structure_render.png')

# ── Colors ───────────────────────────────────────────────────────────
C = {
    'design': '#2E6FAC', 'expt': '#D9756C', 'ml': '#A0527A',
    'rfd': '#E67E50', 'mpnn': '#8B7AB8', 'af3': '#2E6FAC', 'qc': '#70AD87',
    'score': '#E07020', 'value': '#2D9B4E', 'attn': '#D63030',
    'loop': '#C0392B', 'text': '#1A1A1A', 'sub': '#555555',
}

fig = plt.figure(figsize=(18, 10))

# ── Three horizontal strips (tighter spacing) ────────────────────────
ax_top = fig.add_axes([0.02, 0.84, 0.96, 0.13])   # design pipeline
ax_mid = fig.add_axes([0.02, 0.20, 0.96, 0.62])    # ML analysis
ax_bot = fig.add_axes([0.02, 0.04, 0.96, 0.13])    # experimental

for a in [ax_top, ax_mid, ax_bot]:
    a.axis('off')

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def pill(ax, x, y, w, h, title, sub, color, num=None, fs_title=12, fs_sub=9.5):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.10", linewidth=1.5,
                         facecolor=color, edgecolor='white', alpha=0.92, zorder=3)
    ax.add_patch(box)
    if num is not None:
        # Badge positioned just outside top-left corner of box
        bx = x - w/2 - 0.05
        by = y + h/2 + 0.05
        badge = plt.Circle((bx, by), 0.18,
                            facecolor=color, edgecolor='white', linewidth=1.0, zorder=5)
        ax.add_patch(badge)
        ax.text(bx, by, str(num),
                ha='center', va='center', fontsize=8.5, fontweight='bold',
                color='white', zorder=6)
    ty = y + 0.08 if sub else y
    ax.text(x, ty, title, ha='center', va='center',
            fontsize=fs_title, fontweight='bold', color='white', zorder=4)
    if sub:
        ax.text(x, y - 0.17, sub, ha='center', va='center',
                fontsize=fs_sub, color='white', alpha=0.85, zorder=4)

def harrow(ax, x1, y, x2, color='#444', lw=1.5):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->,head_width=0.18,head_length=0.12',
                                color=color, lw=lw), zorder=2)

# ═══════════════════════════════════════════════════════════════════════
# TOP: Computational Design (1-4) — compact boxes
# ═══════════════════════════════════════════════════════════════════════
ax_top.set_xlim(0, 20)
ax_top.set_ylim(0, 1.8)

ax_top.text(0.15, 1.6, 'COMPUTATIONAL DESIGN', fontsize=14, fontweight='bold',
            color=C['design'], va='center')

xpos = [2.5, 7.0, 11.5, 16.0]
top_data = [('RFDiffusion', 'scaffold generation', 1, C['rfd']),
            ('ProteinMPNN', 'sequence design', 2, C['mpnn']),
            ('AlphaFold 3', 'structure prediction', 3, C['af3']),
            ('QC Filter', 'iPTM / pLDDT / PAE', 4, C['qc'])]

for (t, s, n, col), xc in zip(top_data, xpos):
    pill(ax_top, xc, 0.7, 3.2, 0.95, t, s, col, num=n)
top_colors = [d[3] for d in top_data]
for i in range(3):
    harrow(ax_top, xpos[i] + 1.6, 0.7, xpos[i+1] - 1.6, color='#888')

# ═══════════════════════════════════════════════════════════════════════
# BOTTOM: Experimental (5-8) — compact boxes
# ═══════════════════════════════════════════════════════════════════════
ax_bot.set_xlim(0, 20)
ax_bot.set_ylim(0, 1.8)

ax_bot.text(0.15, 1.6, 'EXPERIMENTAL VALIDATION', fontsize=14, fontweight='bold',
            color=C['expt'], va='center')

bpos = [16.0, 11.5, 7.0, 2.5]
bot_data = [('Expression', 'E. coli', 5),
            ('Purification', 'affinity chrom.', 6),
            ('ELISA', 'binding assay', 7),
            ('Binding Data', 'n = 15 designs', 8)]

for (t, s, n), xc in zip(bot_data, bpos):
    pill(ax_bot, xc, 0.7, 3.2, 0.95, t, s, C['expt'], num=n)
for i in range(3):
    harrow(ax_bot, bpos[i] - 1.6, 0.7, bpos[i+1] + 1.6, color=C['expt'])

# ═══════════════════════════════════════════════════════════════════════
# MIDDLE: ML Analysis (tighter layout)
# ═══════════════════════════════════════════════════════════════════════
ax = ax_mid
ax.set_xlim(0, 20)
ax.set_ylim(0, 8)

ax.text(0.15, 6.85, 'ML ANALYSIS', fontsize=14, fontweight='bold', color=C['ml'])

# ── A: Structure render (smaller) ────────────────────────────────────
im = OffsetImage(struct_img, zoom=0.14)
ab = AnnotationBbox(im, (2.0, 4.6), frameon=False,
                    zorder=3)
ax.add_artist(ab)

ax.text(2.0, 6.2, 'AF3 Predicted Complex', fontsize=12, fontweight='bold',
        ha='center', color=C['text'])

# Number badge — top-left of structure image
badge9 = plt.Circle((0.61, 6.25), 0.22, facecolor=C['ml'], edgecolor='white',
                     linewidth=1.0, zorder=5, transform=ax.transData)
ax.add_patch(badge9)
ax.text(0.61, 6.25, '9', fontsize=9, fontweight='bold',
        ha='center', va='center', color='white', zorder=6)

# Color legend (compact, single line)
ly = 1.6
for label, col, dx in [('Inhibitor', '#3A7CC8', 0),
                        ('GTPase', '#999999', 1.8),
                        ('Switch I/II', '#CCCC00', 3.6)]:
    ax.add_patch(Rectangle((0.3 + dx, ly - 0.06), 0.25, 0.18,
                            facecolor=col, edgecolor='none', zorder=3))
    ax.text(0.65 + dx, ly + 0.03, label, fontsize=8.5, va='center', color=C['sub'])

# ── B: Feature matrix ────────────────────────────────────────────────
mat_cx, mat_cy = 5.8, 4.6
mat_w, mat_h = 1.5, 2.8
n_rows, n_cols = 16, 6

np.random.seed(42)
feat_data = np.random.rand(n_rows, n_cols)
feat_data[:, :2] = np.round(feat_data[:, :2])
feat_data[:, 2:4] *= 0.3
feat_data[4:7, 2:4] = 0.9
feat_data[11:13, 2:4] = 0.7

cw = mat_w / n_cols
ch = mat_h / n_rows

for i in range(n_rows):
    for j in range(n_cols):
        color = plt.cm.YlOrRd(feat_data[i, j] * 0.85)
        rect = Rectangle((mat_cx - mat_w/2 + j * cw,
                           mat_cy + mat_h/2 - (i + 1) * ch),
                          cw, ch, facecolor=color, edgecolor='white',
                          linewidth=0.3, zorder=3)
        ax.add_patch(rect)

border = Rectangle((mat_cx - mat_w/2, mat_cy - mat_h/2), mat_w, mat_h,
                    facecolor='none', edgecolor='#888', linewidth=0.8, zorder=4)
ax.add_patch(border)

ax.text(mat_cx, mat_cy + mat_h/2 + 0.55, 'Feature Matrix', fontsize=12,
        fontweight='bold', ha='center', color=C['text'])
ax.text(mat_cx, mat_cy + mat_h/2 + 0.18, 'n residues × 35 features',
        fontsize=9, ha='center', color=C['sub'], fontstyle='italic')

ax.text(mat_cx - mat_w/2 - 0.1, mat_cy, 'residues',
        fontsize=8.5, ha='right', va='center', color=C['sub'],
        rotation=90, fontstyle='italic')

badge10 = plt.Circle((mat_cx - mat_w/2 - 0.19, mat_cy + mat_h/2 + 0.6), 0.22,
                      facecolor=C['ml'], edgecolor='white', linewidth=1.0, zorder=5)
ax.add_patch(badge10)
ax.text(mat_cx - mat_w/2 - 0.19, mat_cy + mat_h/2 + 0.6, '10',
        fontsize=9, fontweight='bold',
        ha='center', va='center', color='white', zorder=6)

# Arrow: structure → features
harrow(ax, 0, 4.6, mat_cx - mat_w/2 - 0.1, color=C['ml'], lw=1.8)
ax.text(4.3, 5.3, 'extract per-residue\nfeatures', fontsize=9,
        ha='center', color=C['ml'], fontstyle='italic', linespacing=1.3)

# ── C: Score-Value Model ─────────────────────────────────────────────

# Step 11 label
ax.text(9.0, 7.2, 'Score-Value Attention Model', fontsize=12,
        fontweight='bold', ha='left', va='center', color=C['text'])
badge11 = plt.Circle((8.74, 7.2), 0.22, facecolor=C['ml'], edgecolor='white',
                      linewidth=1.0, zorder=5)
ax.add_patch(badge11)
ax.text(8.74, 7.2, '11', fontsize=9, fontweight='bold',
        ha='center', va='center', color='white', zorder=6)

# Score network (top branch)
sx, sy = 10.1, 6.0
sw, sh = 2.3, 0.75
pill(ax, sx, sy, sw, sh, 'Score Network', 'MLP: 35 → 8 → 1',
     C['score'], fs_title=11, fs_sub=8.5)

# Softmax (top branch)
fx, fy = 13.1, 6.0
fw, fh = 1.8, 0.75
pill(ax, fx, fy, fw, fh, 'Softmax / τ', '+ entropy loss',
     C['attn'], fs_title=11, fs_sub=8.5)

# Value network (bottom branch)
vx, vy = 10.1, 3.0
vw, vh = 2.3, 0.75
pill(ax, vx, vy, vw, vh, 'Value Network', 'Linear: 35 → 1',
     C['value'], fs_title=11, fs_sub=8.5)

# Arrow: features → split point
split_x = 8.0
harrow(ax, mat_cx + mat_w/2, 4.6, split_x - 0.25, color=C['ml'], lw=1.8)

# Split point
ax.text(split_x, 4.6, 'xᵢ', fontsize=14, fontweight='bold',
        ha='center', va='center', color=C['ml'],
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor=C['ml'], linewidth=1.2))

# Fork: split → score, split → value
ax.annotate('', xy=(sx - sw/2, sy), xytext=(split_x + 0.2, 4.85),
            arrowprops=dict(arrowstyle='->,head_width=0.18,head_length=0.12',
                            color=C['score'], lw=1.5), zorder=2)
ax.annotate('', xy=(vx - vw/2, vy), xytext=(split_x + 0.2, 4.35),
            arrowprops=dict(arrowstyle='->,head_width=0.18,head_length=0.12',
                            color=C['value'], lw=1.5), zorder=2)

# Score → Softmax
harrow(ax, sx + sw/2, sy, fx - fw/2, color=C['score'], lw=1.5)

# Branch labels
ax.text(fx, fy + fh/2 + 0.25, 'αᵢ  (attention weights)',
        fontsize=10, ha='center', color=C['attn'], fontstyle='italic')
ax.text(vx, vy - vh/2 - 0.25, 'vᵢ  (scalar contributions)',
        fontsize=10, ha='center', color=C['value'], fontstyle='italic')

# Multiply node
mx, my = 15.2, 4.6
circle_m = plt.Circle((mx, my), 0.28,
                       facecolor='white', edgecolor=C['text'],
                       linewidth=1.5, zorder=3)
ax.add_patch(circle_m)
ax.text(mx, my, '×', ha='center', va='center',
        fontsize=20, fontweight='bold', color=C['text'], zorder=4)

# Softmax → multiply
ax.annotate('', xy=(mx - 0.03, my + 0.24), xytext=(fx + fw/2, fy),
            arrowprops=dict(arrowstyle='->,head_width=0.15,head_length=0.1',
                            color=C['attn'], lw=1.3,
                            connectionstyle='arc3,rad=-0.1'), zorder=2)
# Value → multiply
ax.annotate('', xy=(mx - 0.03, my - 0.24), xytext=(vx + vw/2, vy),
            arrowprops=dict(arrowstyle='->,head_width=0.15,head_length=0.1',
                            color=C['value'], lw=1.3,
                            connectionstyle='arc3,rad=0.1'), zorder=2)

# Sum node
sum_x = 17.0
circle_s = plt.Circle((sum_x, my), 0.25,
                       facecolor=C['ml'], edgecolor='white',
                       linewidth=1.2, zorder=3)
ax.add_patch(circle_s)
ax.text(sum_x, my, 'Σ', ha='center', va='center',
        fontsize=15, fontweight='bold', color='white', zorder=4)

harrow(ax, mx + 0.28, my, sum_x - 0.25, color=C['text'], lw=1.3)
ax.text(mx + 0.8, my + 0.08, 'αᵢ·vᵢ', fontsize=14, ha='center',
        color=C['text'], fontweight='bold')

# Output: ŷ
harrow(ax, sum_x + 0.25, my, 18.2, color=C['ml'], lw=1.5)
ax.text(18.5, my, 'ŷ', fontsize=18, fontweight='bold',
        ha='center', va='center', color=C['ml'])
ax.text(18.5, my - 0.45, 'binding\nprediction', fontsize=9,
        ha='center', va='top', color=C['sub'])

# ── Attention profile inset (step 12) ────────────────────────────────
bar_x0, bar_y0 = 15.0, 1.0
bar_w_total, bar_h_total = 3.5, 1.1
np.random.seed(7)
n_bars = 30
attn_vals = np.random.exponential(0.25, n_bars)
attn_vals[6] = 1.4; attn_vals[10] = 1.1; attn_vals[18] = 0.8
attn_vals = attn_vals / attn_vals.max()

bw = bar_w_total / n_bars
for i in range(n_bars):
    h = attn_vals[i] * bar_h_total * 0.85
    col = plt.cm.Reds(0.25 + 0.6 * attn_vals[i])
    rect = Rectangle((bar_x0 + i * bw, bar_y0), bw * 0.85, h,
                      facecolor=col, edgecolor='none', zorder=3)
    ax.add_patch(rect)

ax.text(bar_x0 + bar_w_total/2, bar_y0 - 0.2, 'scaffold position',
        fontsize=8, ha='center', color=C['sub'])
ax.text(bar_x0 - 0.1, bar_y0 + bar_h_total/2, 'αᵢ',
        fontsize=10, ha='right', va='center', color=C['attn'], fontstyle='italic')

badge12 = plt.Circle((bar_x0 - 0.45, bar_y0 + bar_h_total + 0.3), 0.22,
                      facecolor=C['ml'], edgecolor='white', linewidth=1.0, zorder=5)
ax.add_patch(badge12)
ax.text(bar_x0 - 0.45, bar_y0 + bar_h_total + 0.3, '12', fontsize=9,
        fontweight='bold', ha='center', va='center', color='white', zorder=6)
ax.text(bar_x0 - 0.1, bar_y0 + bar_h_total + 0.3,
        'Interpretation', fontsize=11, fontweight='bold',
        ha='left', va='center', color=C['text'])

# ── Key insight (bottom of ML panel) ─────────────────────────────────
box_ins = FancyBboxPatch((2.5, 0.15), 10.0, 0.6,
                          boxstyle="round,pad=0.10", linewidth=1.0,
                          facecolor='#FFF8E8', edgecolor='#D4A020',
                          alpha=0.9, zorder=3)
ax.add_patch(box_ins)
ax.text(7.5, 0.45,
        'Scalar values force information through the attention bottleneck → interpretable importance',
        fontsize=12, ha='center', va='center', color='#6B4F10', fontstyle='italic', zorder=4)

# ═══════════════════════════════════════════════════════════════════════
# CLOSED LOOP ARROW (fig-level, tighter)
# ═══════════════════════════════════════════════════════════════════════

fig.patches.append(FancyArrowPatch(
    (0.92, 0.56), (0.96, 0.56),
    arrowstyle='-', color=C['loop'], lw=2.5,
    transform=fig.transFigure, mutation_scale=1, zorder=8))
# Right edge: up from ML to Design row
fig.patches.append(FancyArrowPatch(
    (0.96, 0.56), (0.96, 0.83),
    arrowstyle='-', color=C['loop'], lw=2.5,
    transform=fig.transFigure, mutation_scale=1, zorder=8))
# Top edge: across to left
fig.patches.append(FancyArrowPatch(
    (0.96, 0.83), (0, 0.83),
    arrowstyle='-', color=C['loop'], lw=2.5,
    transform=fig.transFigure, mutation_scale=1, zorder=8))
# Left edge: up to comp design row
fig.patches.append(FancyArrowPatch(
    (0, 0.83), (0, 0.89),
    arrowstyle='-', color=C['loop'], lw=2.5,
    transform=fig.transFigure, mutation_scale=1, zorder=8))
# Right edge num 2: comp design row
fig.patches.append(FancyArrowPatch(
    (0, 0.89), (0.05, 0.89),
    arrowstyle='->,head_width=6,head_length=5',
    color=C['loop'], lw=2.5,
    transform=fig.transFigure, mutation_scale=1, zorder=8))

fig.text(0.50, 0.81, 'Attention-guided redesign',
         fontsize=12, fontweight='bold', color=C['loop'], ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                   edgecolor=C['loop'], linewidth=1.5), zorder=11)

# Vertical connector: QC → Expression (right side, dashed)
fig.patches.append(FancyArrowPatch(
    (0.783, 0.84), (0.783, 0.17),
    arrowstyle='->,head_width=4,head_length=3',
    color=C['expt'], lw=2.4, linestyle='dashed', alpha=0.35,
    transform=fig.transFigure, mutation_scale=1, zorder=1))
fig.text(0.769, 0.50, 'designs', fontsize=8.5, color=C['expt'],
         fontstyle='italic', alpha=0.55, rotation=90, va='center',
         transform=fig.transFigure)

# AF3 → ML structures (dashed)
fig.patches.append(FancyArrowPatch(
    (0.58, 0.84), (0.13, 0.74),
    arrowstyle='->,head_width=4,head_length=3',
    color=C['design'], lw=2.4, linestyle='dashed', alpha=0.45,
    transform=fig.transFigure, mutation_scale=1, zorder=1))
fig.text(0.36, 0.76, 'structures', fontsize=8.5, color=C['design'],
         fontstyle='italic', alpha=0.55, ha='center',
         transform=fig.transFigure)

# Binding data → ML labels (dashed)
fig.patches.append(FancyArrowPatch(
    (0.025, 0.16), (0, 0.16),
    arrowstyle='-', 
    color=C['expt'], lw=2.4, linestyle='dashed', alpha=0.45,
    transform=fig.transFigure, mutation_scale=1, zorder=8))
fig.patches.append(FancyArrowPatch(
    (0, 0.16), (0, 0.74),
    arrowstyle='-', 
    color=C['expt'], lw=2.4, linestyle='dashed', alpha=0.45,
    transform=fig.transFigure, mutation_scale=1, zorder=8))
fig.patches.append(FancyArrowPatch(
    (0, 0.74), (0.025, 0.74),
    arrowstyle='->,head_width=4,head_length=3',
    color=C['expt'], lw=2.4, linestyle='dashed', alpha=0.45,
    transform=fig.transFigure, mutation_scale=1, zorder=1))

# ── Save ─────────────────────────────────────────────────────────────
for fmt in ['svg', 'png', 'pdf']:
    kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
    if fmt == 'png':
        kw['dpi'] = 300
    path = f'/cwork/hsb26/GEF_inhibition/results/figures/architecture_diagram.{fmt}'
    plt.savefig(path, **kw)
    print(f'Saved {path}')
