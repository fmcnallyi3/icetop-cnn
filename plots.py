PLOTS_VERSION = 4
LEGEND_COLUMNS = 3

# Color cycle + legend tools for the assessment notebooks.
# Importing this module sets the notebook-wide color cycle (60 dark, then lights).
# make_interactive is a no-op: the hub UI strips widget mimes, so click-to-toggle
# cannot render. Filter via the pairs cell and re-run plot cells.

import matplotlib.pyplot as plt
from matplotlib import cycler


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
        for a in list(ax.lines) + list(ax.patches) + list(ax.collections):
            lab = a.get_label()
            if lab and not lab.startswith('_') and lab not in seen:
                seen.add(lab)
                out.append(a)
    return out


def make_interactive(fig):
    # No-op: hub UI strips widget mimes (buttons render as raw repr), so
    # click-to-toggle cannot render. Filter via the pairs cell instead.
    return
