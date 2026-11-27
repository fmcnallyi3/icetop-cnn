PLOTS_VERSION = 8
LEGEND_COLUMNS = 4

# Colour cycle and interaction helpers for the assessment notebooks.
# Importing this module sets the notebook-wide colour cycle (60 dark, then lights).

import matplotlib.pyplot as plt
from matplotlib import cycler

plt.rcParams['legend.labelcolor'] = '0.35'
plt.rcParams['legend.edgecolor'] = '0.35'
# 72 dpi keeps the plots and their control rows on one screen. Font sizes are in
# points, so proportions are unchanged; only the raster gets smaller.
plt.rcParams['figure.dpi'] = 72


def _lum(c):
    r, g, b = c[:3]
    return 0.299*r + 0.587*g + 0.114*b


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


def _labelled_artists(fig):
    # Every labelled artist, grouped by label, across all axes. Subplots that
    # sit side by side draw one copy of each curve per axes, so a toggle has to
    # reach all of them, not just the first one it finds.
    groups = {}
    for ax in fig.axes:
        # mpl 3.11: errorbar labels live on the ErrorbarContainer, its Line2D
        # children are '_nolegend_'. Containers first, then stray labeled lines.
        for a in list(ax.containers) + list(ax.lines) + list(ax.patches):
            lab = a.get_label()
            if lab and not lab.startswith('_'):
                groups.setdefault(lab, []).append(a)
    return groups


def unique_handles(fig):
    # One artist per label, first wins. fig.legend over multi-axes repeats each
    # label without this.
    return [group[0] for group in _labelled_artists(fig).values()]


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


def _detach_auto_display(fig):
    # ipympl queues every figure for display at the end of the cell. The control
    # panel already holds this canvas, so drop the queued copy or the cell
    # renders a second, static version of the same figure.
    try:
        from ipympl.backend_nbagg import _Backend_ipympl
        _Backend_ipympl._to_show.remove(fig)
    except (ImportError, AttributeError, ValueError):
        pass


# Panels built while deferring waits here until show_all() runs.
_defer = False
_pending = []


def defer(enable=True):
    '''Hold every interactive panel back until show_all().

    A full notebook run builds a dozen large canvases; rendering each one as its
    cell finishes floods the widget channel while the next cell is still
    working. Deferring shows them all at the end instead.'''
    global _defer
    _defer = enable


def show_all():
    '''Display every panel built since the last show_all().'''
    from IPython.display import display
    while _pending:
        display(_pending.pop(0))


def make_interactive(fig):
    '''Adds show/hide controls under a figure.

    One button per model (a column) and per dataset (a row), all on at the
    start. Clicking a button or a legend entry hides that model or dataset in
    every subplot at once.'''
    groups = _labelled_artists(fig)
    if not groups:
        return fig

    shown = {lab: True for lab in groups}
    hidden = {'model': set(), 'dataset': set()}
    models, datasets = [], []
    for lab in groups:
        model, dataset = _split_label(lab)
        if model not in models:
            models.append(model)
        if dataset is not None and dataset not in datasets:
            datasets.append(dataset)

    buttons = {}

    def is_on(kind, name):
        return name not in hidden[kind]

    def refresh():
        for lab, artists in groups.items():
            model, dataset = _split_label(lab)
            visible = shown[lab] and is_on('model', model) and is_on('dataset', dataset)
            for artist in artists:
                _set_visible(artist, visible)
        for (kind, name), button in buttons.items():
            button.button_style = 'success' if is_on(kind, name) else ''
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

    panel_children = []
    is_widget = isinstance(fig.canvas, widgets.Widget)
    if is_widget:
        # ipympl: the canvas is a widget, so it belongs inside the panel and the
        # buttons can sit directly under it.
        panel_children.append(fig.canvas)
    panel_children.append(widgets.HBox([make_button('model', m) for m in models]))
    if datasets:
        panel_children.append(widgets.HBox([make_button('dataset', d) for d in datasets]))
    panel = widgets.VBox(panel_children)
    fig._controls = panel
    refresh()
    if is_widget:
        _detach_auto_display(fig)
    if _defer:
        _pending.append(panel)
    else:
        display(panel)
    return fig
