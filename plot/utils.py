
def add_better(ax, x, y, direc, fontsize=16):
    '''add a better arrow in the given axes

    Args:
        ax: the given axes to draw arrow on
        x: x coord in axes percentage, should be in (0, 1)
        y: y coord in axes percentage, should be in (0, 1)
        direc: direction of the arrow, can be one of down, up, left, right,
            tr (top right), tl (top left), br (bottom right), bl (bottom left)
        fontsize: font size of the text

    Usage:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
        for i in range(5):
            add_better(ax, np.random.random(), np.random.random(), 'up', \
                np.random.randint(5, 30))
        fig.savefig('./test.pdf', format='pdf', bbox_inches='tight')
    '''
    assert 0 < x < 1 and 0 < y < 1, f'invalid x ({x}) or y ({y}) value'
    assert direc in ['up', 'down', 'left', 'right', 'tr', 'tl', 'br',
                     'bl'], ('invalid direc value ({direc})')

    if direc == 'down':
        ax.annotate('',
                    xy=(x, y - 0.05 / 16 * fontsize),
                    xytext=(0, 51 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x - 0.009 / 16 * fontsize,
                y,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=270,
                transform=ax.transAxes)
    elif direc == 'up':
        ax.annotate('',
                    xy=(x, y + 0.30 / 16 * fontsize),
                    xytext=(0, -51 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x - 0.005 / 16 * fontsize,
                y,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=270,
                transform=ax.transAxes)
    elif direc == 'left':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(51 / 16 * fontsize, 0),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x + 0.105 / 16 * fontsize,
                y - 0.05 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=0,
                transform=ax.transAxes)
    elif direc == 'right':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(-51 / 16 * fontsize, 0),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x - 0.105 / 16 * fontsize,
                y - 0.05 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=0,
                transform=ax.transAxes)
    elif direc == 'bl':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(36 / 16 * fontsize, 36 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x + 0.075 / 16 * fontsize,
                y + 0.012 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=45,
                transform=ax.transAxes)
    elif direc == 'tl':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(36 / 16 * fontsize, -36 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x + 0.07 / 16 * fontsize,
                y - 0.25 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=315,
                transform=ax.transAxes)
    elif direc == 'tr':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(-36 / 16 * fontsize, -36 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x - 0.07 / 16 * fontsize,
                y - 0.24 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=45,
                transform=ax.transAxes)
    elif direc == 'br':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(-36 / 16 * fontsize, 36 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x - 0.075 / 16 * fontsize,
                y - 0.012 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=315,
                transform=ax.transAxes)
