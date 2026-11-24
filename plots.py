PLOTS_VERSION = 6

# Colour cycle and interaction helpers for the assessment notebooks.
# Importing this module sets the notebook-wide colour cycle (60 dark, then lights).

import matplotlib.pyplot as plt
from matplotlib import cycler

plt.rcParams['legend.labelcolor'] = '0.35'
plt.rcParams['legend.edgecolor'] = '0.35'


def _lum(c):
    r, g, b = c[:3]
    return 0.299*r + 0.587*g + 0.114*b


def _dark_only(cmap, n):
    return [cmap(i / max(n - 1, 1)) for i in range(n) if _lum(cmap(i / max(n - 1, 1))) < 0.62]


_pool = ([plt.cm.tab20(2*i) for i in range(10)]
         + [plt.cm.tab20b(2*i) for i in range(10)]
         + [plt.cm.tab20c(2*i) for i in range(10)]
         + [plt.cm.Set3(i/11) for i in range(12)]
         + [plt.cm.Dark2(i/7) for i in range(8)]
         + [plt.cm.Paired(i/11) for i in range(12)]
         + [plt.cm.Accent(i/7) for i in range(8)]
         + [plt.cm.Set1(i/8) for i in range(9)]
         + [plt.cm.gist_ncar(i/23) for i in range(24)]
         + [plt.cm.gist_rainbow(i/23) for i in range(24)])
_cycle = [c for c in _pool if _lum(c) < 0.60]
_seen, _dedup = set(), []
for _c in _cycle:
    _k = tuple(round(v, 2) for v in _c[:3])
    if _k not in _seen:
        _seen.add(_k)
        _dedup.append(_c)
    if len(_dedup) >= 60:
        break
plt.rcParams['axes.prop_cycle'] = cycler(color=_dedup)


def unique_handles(fig):
    # One artist per label across all axes; first wins. fig.legend over
    # multi-axes repeats each label without this.
    out, seen = [], set()
    for ax in fig.axes:
        # mpl 3.11: errorbar labels live on the ErrorbarContainer, its Line2D
        # children are '_nolegend_'. Containers first, then stray labeled lines.
        for a in list(ax.containers) + list(ax.lines) + list(ax.patches):
            lab = a.get_label()
            if lab and not lab.startswith('_') and lab not in seen:
                seen.add(lab)
                out.append(a)
    return out


def _split_label(label):
    # The notebooks label a curve 'model [dataset]'. Anything else is model-only.
    if label.endswith(']') and ' [' in label:
        model, dataset = label.rsplit(' [', 1)
        return model, dataset[:-1]
    return label, None


def _set_visible(artist, visible):
    if hasattr(artist, 'set_visible'):
        artist.set_visible(visible)
        return
    # An ErrorbarContainer holds its parts in .lines and .bars (sometimes
    # nested in tuples) instead of being a single drawable artist.
    for attr in ('lines', 'bars', 'patches'):
        for part in getattr(artist, attr, None) or []:
            _set_visible(part, visible)


def make_interactive(fig):
    '''Adds show/hide controls to a finished figure.

    Clicking a legend entry toggles that one curve. The buttons above the
    figure toggle a whole model (one column) or a whole dataset (one row).
    Both layers apply at once, so a legend hide survives a button filter.'''
    handles = unique_handles(fig)
    if not handles:
        return fig

    labels = [h.get_label() for h in handles]
    shown = {lab: True for lab in labels}
    hidden = {'model': set(), 'dataset': set()}
    models, datasets = [], []
    for lab in labels:
        model, dataset = _split_label(lab)
        if model not in models:
            models.append(model)
        if dataset is not None and dataset not in datasets:
            datasets.append(dataset)

    buttons = {}

    def refresh():
        for h, lab in zip(handles, labels):
            model, dataset = _split_label(lab)
            _set_visible(h, shown[lab]
                         and model not in hidden['model']
                         and dataset not in hidden['dataset'])
        for (kind, name), button in buttons.items():
            button.button_style = 'warning' if name in hidden[kind] else ''
        fig.canvas.draw_idle()

    def toggle(kind, name):
        hidden[kind].symmetric_difference_update({name})
        refresh()

    # Legend clicks. Key off the text itself; the parallel handle list can carry
    # '_nolegend_' entries that do not correspond to a plotted curve.
    for legend in fig.legends + [ax.get_legend() for ax in fig.axes if ax.get_legend()]:
        for text in legend.get_texts():
            if text.get_text() in shown:
                text.set_picker(True)
                text._toggle_label = text.get_text()

    def on_pick(event):
        label = getattr(event.artist, '_toggle_label', None)
        if label is None:
            return
        shown[label] = not shown[label]
        event.artist.set_alpha(1.0 if shown[label] else 0.2)
        refresh()

    fig.canvas.mpl_connect('pick_event', on_pick)

    # Buttons live outside the canvas; constrained layout leaves no room inside.
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        fig._controls = None
        return fig

    def make_button(kind, name):
        button = widgets.Button(description=name,
                                layout=widgets.Layout(width='auto', min_width='6em'))
        button.on_click(lambda _: toggle(kind, name))
        buttons[(kind, name)] = button
        return button

    model_row = widgets.HBox([make_button('model', m) for m in models])
    dataset_row = widgets.HBox([make_button('dataset', d) for d in datasets])
    panel = widgets.VBox([model_row, dataset_row])
    fig._controls = panel
    display(panel)
    return fig
