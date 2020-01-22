import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numpy as np


def barplot_with_folds(labels, names, data, colors, height=1.8, width=4, filename=None):

    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
        
    fig, ax = plt.subplots(figsize=(width, height))

    barWidth = 0.5
    positions = np.arange(len(labels)) * barWidth * 1.1

    THRESHOLD = 20
    BOTTOM_PADDING = 2
    TOP_PADDING = 4
    LEFT_PADDING = 0.22
    PATCH_LEFT_PADDING = 0.2465
    POLYGON_TOP = 18
    POLYGON_TOP_LEFT = 0.1
    POLYGON_TOP_RIGHT = 0.05
    POLYGON_BOTTOM_PADDING = - 1

    for i, label in enumerate(labels):
        values = data[label]
        percentages = np.array(values)
        percentages = percentages / percentages.sum() * 100
        # Top position of last bar
        bottom = 0
        align_left = not bool(len(names) % 2)  # So that totals always are on the right
        prev_percentage = None
        # Buffer used to detect if last legend label (text on the right of all bars) 
        # would overlap with current one. If so current one is pushed upward.
        label_bottom = 0
        # Buffer used to detect if a fold must be pushed upward because last label
        # on the same side was also pushed upward and would overlap.
        last_left = 0
        last_right = 0
        for j, (name, value, percentage, color) in enumerate(zip(names, values, percentages, colors)):
            ax.bar([positions[i]], [percentage], bottom=bottom, edgecolor='white', width=barWidth,
                   color=color, align='center')

            config = {'s': value, 'fontsize': 11, 'zorder': len(names) - j + 1}

            x = positions[i]
            if align_left:
                fold_bottom = max(last_left, bottom)
                last_left = fold_bottom
                config['x'] = x - LEFT_PADDING
                config['horizontalalignment'] = 'left'
            else:
                fold_bottom = max(last_right, bottom)
                last_right = fold_bottom
                config['x'] = x + LEFT_PADDING
                config['horizontalalignment'] = 'right'

            if percentage - (fold_bottom - bottom) > THRESHOLD:
                config['verticalalignment'] = 'top'
                config['y'] = bottom + percentage - TOP_PADDING
                if align_left:
                    last_left = config['y']
                else:
                    last_right = config['y']
            else:
                config['verticalalignment'] = 'bottom'
                config['y'] = fold_bottom + BOTTOM_PADDING
                if align_left:
                    last_left = config['y'] + POLYGON_TOP * .9 - BOTTOM_PADDING
                else:
                    last_right = config['y'] + POLYGON_TOP * .9 - BOTTOM_PADDING

            ax.text(color='white', **config)

            if j == 0 and percentage < 1:
                polygon_bottom = 0
            else:
                polygon_bottom = bottom + BOTTOM_PADDING + POLYGON_BOTTOM_PADDING

            if percentage - (fold_bottom - bottom) <= THRESHOLD and align_left:
                pts = np.array([[x - PATCH_LEFT_PADDING, polygon_bottom],
                                [x - PATCH_LEFT_PADDING, fold_bottom + POLYGON_TOP],
                                [x - POLYGON_TOP_LEFT, fold_bottom + POLYGON_TOP],
                                [x - POLYGON_TOP_RIGHT, polygon_bottom]])
                p = Polygon(pts, closed=True, color=color, linewidth=0, alpha=1,
                            zorder=len(names) - j)
                ax.add_patch(p)
            elif percentage - (fold_bottom - bottom) <= THRESHOLD and not align_left:
                pts = np.array([[x + PATCH_LEFT_PADDING, polygon_bottom],
                                [x + PATCH_LEFT_PADDING, fold_bottom + POLYGON_TOP],
                                [x + POLYGON_TOP_LEFT, fold_bottom + POLYGON_TOP],
                                [x + POLYGON_TOP_RIGHT, polygon_bottom]])
                p = Polygon(pts, closed=False, color=color, linewidth=0, alpha=1,
                            zorder=len(names) - j)
                ax.add_patch(p)

            outside = 0.12
            if i == len(labels) - 1:
                nudge = 0
                if label_bottom < bottom + percentage / 2:
                    label_bottom = bottom + percentage / 2
                ax.text(x=x + LEFT_PADDING + outside - 0.05,
                        y=label_bottom,
                        s=name,
                        color=color,
                        fontsize=12,
                        horizontalalignment='left',
                        verticalalignment='center')
                label_bottom += 15

            bottom += max(percentage, 1)
            align_left = not align_left

            prev_percentage = percentage 

        config = {'s': 'total', 'fontsize': 11, 'zorder': len(names) - j + 1}
        x = positions[i]

        if align_left:
            fold_bottom = min(max(last_left, bottom), bottom + percentage - 2)
            last_left = fold_bottom
            config['x'] = x - LEFT_PADDING
            config['horizontalalignment'] = 'left'
        else:
            fold_bottom = min(max(last_right, bottom), bottom + percentage)
            last_right = fold_bottom
            config['x'] = x + LEFT_PADDING
            config['horizontalalignment'] = 'right'

        if percentage - (fold_bottom - bottom) > THRESHOLD:
            config['verticalalignment'] = 'top'
            config['y'] = bottom + percentage - TOP_PADDING
            if align_left:
                last_left = config['y']
            else:
                last_right = config['y']
        else:
            config['verticalalignment'] = 'bottom'
            config['y'] = fold_bottom + BOTTOM_PADDING
            if align_left:
                last_left = config['y'] + POLYGON_TOP * .9 - BOTTOM_PADDING
            else:
                last_right = config['y'] + POLYGON_TOP * .9 - BOTTOM_PADDING

        config['y'] = 100
        config['verticalalignment'] = 'bottom'
        config['s'] = str(sum(values))
        config['alpha'] = 0.6
        ax.text(color='black', **config)


    ax.set_ylabel('%', fontsize=12,
                  color='#333F4B', rotation=0)
    ax.set_xlabel('')

    ax.tick_params(axis='y', which='major', labelsize=8)
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='x', which=u'both', length=0)
     
    # change the style of the axis spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)

    # Custom x axis
    ax.set_yticks(np.arange(0, 125, 25))
    ax.spines['left'].set_bounds(0, 100)
    ax.set_xlim(-0.3, positions[-1] + 0.3 + 0.8)
    # ax.set_ylim(0, 120)

    # plt.subplots_adjust(top=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    # Show graphic
    print(filename)
    plt.savefig(filename, dpi=300)


def barplot_vertical(labels, values, filename, vmax, width, height, xlabel, ylabel=None,
                     threshold=None):

    fig, ax = plt.subplots(figsize=(width, height))
    ax.barh(np.arange(len(values)), values, color='#737373')
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(labels)

    ax.set_xticks(np.arange(0, 70, 10))

    red = '#cb181d'

    if threshold:
        ax.hlines(threshold, 0, vmax, linestyles='dashed', color=red)
        ax.text(vmax, threshold + 0.5, '50%', horizontalalignment='right',
                verticalalignment='center', color=red)

    total = sum(values)

    for i, value in enumerate(values):
        ax.text(value + 1, i, '{}%'.format(int(value / total * 100 + 0.5)),
                horizontalalignment='left',
                verticalalignment='center')

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_smart_bounds(True)

    ax.tick_params(axis='y', which=u'both',length=0)

    print(filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    colors = """\
#46a285
#dc6d42
#6d80ab
#c76aa3
#86a824""".split('\n')

    barplot_with_folds(
        labels='A',
        names=['One', 'Two', 'Three', 'Four', 'Five'],
        data={
            'A': [10, 20, 50, 30, 30]},
        colors=colors,
        filename='test1.png')

    barplot_with_folds(
        labels='AB',  # DE',
        names=['One', 'Two', 'Three', 'Four', 'Five'],
        data={
            'A': [2, 0, 3, 4, 80],
            'B': [20, 10, 30, 50, 30]},
        colors=colors,
        width=4,
        height=1.8,
        filename='test2.png')

    barplot_with_folds(
        labels='ABC',
        names=['One', 'Two', 'Three', 'Four', 'Five'],
        data={
            'A': [10, 20, 50, 30, 30],
            'B': [20, 10, 30, 50, 30],
            'C': [2, 0, 3, 4, 80]},
        colors=colors,
        width=6,
        height=1.8,
        filename='test3.png')

    barplot_with_folds(
        labels='ABCDE',
        names=['One', 'Two', 'Three', 'Four', 'Five'],
        data={
            'A': [10, 20, 50, 30, 30],
            'B': [20, 10, 30, 50, 30],
            'C': [2, 0, 3, 4, 80],
            'D': [0, 40, 5, 30, 15],
            'E': [25, 10, 0, 50, 0]},
        colors=colors,
        width=8,
        height=2,
        filename='test5.png')
