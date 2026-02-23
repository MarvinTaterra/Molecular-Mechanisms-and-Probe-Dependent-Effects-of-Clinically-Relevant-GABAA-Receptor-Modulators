"""
Create figure comparing 3 free energy landscapes.
Auto-detects previously computed bootstrap .npz files
Reads COLVAR files from 3 separate input directories (if no cache)
Overlays all 3 free energy profiles on a single clean plot
Saves all outputs in the configured output directory

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
from pathlib import Path
from scipy import stats
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 17,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'legend.framealpha': 1.0,
    'legend.edgecolor': '#333333',
    'legend.fancybox': False,
    'legend.shadow': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 1.8,
    'xtick.major.width': 1.8,
    'ytick.major.width': 1.8,
    'xtick.minor.width': 1.2,
    'ytick.minor.width': 1.2,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3.5,
    'ytick.minor.size': 3.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 3.5,
    'lines.markersize': 8,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.grid': False,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.6,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
})


CONFIG = {

    'inputs': [
        {
            'name': 'GABA',                            
            'colvar_directory': './GABA_open',
            'colvar_pattern': 'COLVAR.*',
            'color': "#648FFF",
            'linestyle': '-',
            'barrier_energy': 50,
            'stuck': False,
        },
        {
            'name': 'GABA + Diazepam',
            'colvar_directory': './Diazepam',
            'colvar_pattern': 'COLVAR.*',
            'color': '#DC267F',
            'linestyle': '-',
            'barrier_energy': 100,
            'stuck': True,
        },
        {
            'name': 'GABA + Phenobarbital',
            'colvar_directory': './Phenobarbital',
            'colvar_pattern': 'COLVAR.*',
            'color': '#785EF0',
            'linestyle': '-',
            'barrier_energy': 100,
            'stuck': True,
        },
    ],

    'output_dir': './publication_figures',
    'figure_name': 'free_energy_3inputs',

    'n_bootstrap': 100,
    'n_cores': None,
    'temperature': 303.15,
    'bandwidth_factor': 0.005,
    'grid_points': 10000,
    'grid_min': None,
    'grid_max': None,
    'figure_width': 7.5,
    'figure_height': 5.0,
    'error_alpha': 0.15,            
    'error_alpha_stuck': 0.25,     
    'plateau_threshold_frac': 0.05,
    'fade_length': 150,
    'y_min': 0,
    'y_max': 70,
    'x_min': None,
    'x_max': 4.5,

    'annotate_states': True,
    'states': {
        'Resting': {'position': -4, 'color': '#0173B2'},
        'Open': {'position': 0, 'color': '#029E73'},
        'Desensitized': {'position': 4, 'color': '#D55E00', 'label_offset': -0.5},
    },
    'state_sigma': 0.3,             

    'show_grid': True,
}




def collect_walker_data(colvar_dir, pattern='COLVAR.*'):
    """Collect data from all COLVAR files in a directory."""
    print(f"\n  Directory: {os.path.abspath(colvar_dir)}")
    print(f"  Pattern: {pattern}")

    colvar_path = Path(colvar_dir)
    colvar_files = sorted(colvar_path.glob(pattern))

    if not colvar_files:
        raise FileNotFoundError(
            f"No COLVAR files found matching '{pattern}' in {colvar_dir}"
        )

    print(f"  Found {len(colvar_files)} COLVAR files")

    all_data = []
    for filepath in colvar_files:
        try:
            with open(filepath, 'r') as f:
                header = ''
                for line in f:
                    if line.startswith('#!'):
                        header = line.strip()
                        break

            if header:
                columns = header.replace('#! FIELDS', '').split()
            else:
                columns = ['time', 'cv.node-0', 'opes.bias', 'opes.rct', 'opes.zed']

            data = np.loadtxt(filepath, comments=['#', '@'])

            time_idx = columns.index('time') if 'time' in columns else 0
            cv_idx = columns.index('cv.node-0') if 'cv.node-0' in columns else 1
            bias_idx = columns.index('opes.bias') if 'opes.bias' in columns else 2
            rct_idx = columns.index('opes.rct') if 'opes.rct' in columns else 3
            zed_idx = columns.index('opes.zed') if 'opes.zed' in columns else 4

            extracted_data = np.column_stack((
                data[:, time_idx],
                data[:, cv_idx],
                data[:, bias_idx],
                data[:, rct_idx],
                data[:, zed_idx],
            ))

            all_data.append(extracted_data)
            print(f"   {filepath.name}: {len(extracted_data):,} frames")

        except Exception as e:
            print(f"  Warning: Could not load {filepath.name}: {e}")

    if not all_data:
        raise ValueError("No data could be loaded from COLVAR files")

    combined_data = np.vstack(all_data)
    print(f"  Total frames: {len(combined_data):,}")

    return combined_data


def reweight_data(data, temperature=303.15):
    """Calculate weights from bias using reweighting."""
    cv_values = data[:, 1]
    bias_values = data[:, 2]

    beta = 1 / (0.00831446 * temperature)
    weights = np.exp(beta * bias_values)
    weights = weights / np.sum(weights)

    n_eff = 1 / np.sum(weights**2)
    print(f"  Samples: {len(weights):,}  |  Effective: {n_eff:.1f}  |  "
          f"Efficiency: {n_eff/len(weights)*100:.2f}%")

    return cv_values, weights


def bootstrap_worker(seed, values, weights, grid, bandwidth, temp):
    np.random.seed(seed)
    n_samples = len(values)
    indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
    bootstrap_values = values[indices]
    bootstrap_weights = weights[indices]
    bootstrap_weights = bootstrap_weights / np.sum(bootstrap_weights)

    try:
        kde = stats.gaussian_kde(
            bootstrap_values, weights=bootstrap_weights, bw_method=bandwidth
        )
        pdf = kde(grid)
        kBT = 0.00831446 * temp
        free_energy = -kBT * np.log(pdf + 1e-10)
        free_energy = free_energy - np.min(free_energy)
        return free_energy
    except Exception as e:
        print(f"  Bootstrap sample failed: {e}")
        return None


def bootstrap_free_energy_multicore(values, weights, grid, config):
    n_bootstrap = config['n_bootstrap']
    n_cores = config['n_cores'] if config['n_cores'] else cpu_count()
    temp = config['temperature']

    bandwidth = config['bandwidth_factor'] * np.std(values)

    worker = partial(
        bootstrap_worker,
        values=values, weights=weights, grid=grid,
        bandwidth=bandwidth, temp=temp,
    )

    seeds = np.random.randint(0, 2**31, size=n_bootstrap)

    print(f"  Running {n_bootstrap} bootstrap samples on {n_cores} cores...")
    with Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.imap(worker, seeds),
            total=n_bootstrap, desc="  Bootstrap",
        ))

    bootstrap_results = np.array([r for r in results if r is not None])

    if len(bootstrap_results) < n_bootstrap:
        print(f"  {n_bootstrap - len(bootstrap_results)} samples failed")

    fe_mean = np.mean(bootstrap_results, axis=0)
    fe_std = np.std(bootstrap_results, axis=0)

    print(f"  FE range: {fe_mean.min():.2f} – {fe_mean.max():.2f} kJ/mol  |  "
          f"Mean SE: {fe_std.mean():.3f} kJ/mol")

    return fe_mean, fe_std, bootstrap_results


def auto_detect_bootstraps(config):
    output_dir = Path(config['output_dir'])
    cached = {}

    for i in range(len(config['inputs'])):
        tag = f"input{i+1}"
        cache_file = output_dir / f'{tag}_bootstrap.npz'
        if cache_file.exists():
            try:
                data = np.load(cache_file)
                cached[i] = {
                    'grid': data['grid'],
                    'fe_mean': data['fe_mean'],
                    'fe_std': data['fe_std'],
                    'bootstrap_samples': data['bootstrap_samples'],
                }
                print(f"  Found cached bootstrap: {cache_file}")
            except Exception as e:
                print(f"  Could not load {cache_file}: {e}")

    old_bootstrap = output_dir / 'cv_bootstrap_samples.npz'
    if old_bootstrap.exists():
        print(f"  Found legacy bootstrap file: {old_bootstrap}")

    return cached


def process_all_inputs(config):
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 70)
    print("AUTO-DETECTING CACHED BOOTSTRAP DATA")
    print("=" * 70)

    cached_bootstraps = auto_detect_bootstraps(config)
    all_cached = all(i in cached_bootstraps for i in range(len(config['inputs'])))

    if all_cached:
        print("\nAll bootstrap caches found — skipping COLVAR reading")
        grid = cached_bootstraps[0]['grid']

        results = []
        for i, inp in enumerate(config['inputs']):
            orig_file = output_dir / f'input{i+1}_original.npz'
            if orig_file.exists():
                orig = np.load(orig_file)
                cv = orig['cv']
                weights = orig['weights']
            else:
                print(f"\n  Loading COLVAR data for Input {i+1}...")
                data = collect_walker_data(inp['colvar_directory'], inp['colvar_pattern'])
                cv, weights = reweight_data(data, config['temperature'])
                np.savez(output_dir / f'input{i+1}_original.npz',
                         cv=cv, weights=weights)

            results.append({
                'name': inp['name'],
                'color': inp['color'],
                'linestyle': inp['linestyle'],
                'barrier_energy': inp['barrier_energy'],
                'stuck': inp.get('stuck', False),
                'cv': cv,
                'weights': weights,
                'fe_mean': cached_bootstraps[i]['fe_mean'],
                'fe_std': cached_bootstraps[i]['fe_std'],
            })

        return grid, results

    print("\n" + "=" * 70)
    print("COLLECTING DATA FROM ALL INPUTS")
    print("=" * 70)

    input_datasets = []
    all_cv_global = []

    for i, inp in enumerate(config['inputs']):
        print(f"\n--- Input {i+1}: {inp['name']} ---")
        data = collect_walker_data(inp['colvar_directory'], inp['colvar_pattern'])
        cv_vals, wts = reweight_data(data, config['temperature'])
        input_datasets.append({'cv': cv_vals, 'weights': wts})
        all_cv_global.append(cv_vals)

        np.savez(output_dir / f'input{i+1}_original.npz',
                 cv=cv_vals, weights=wts)

    all_cv_concat = np.concatenate(all_cv_global)
    g_min = config['grid_min'] if config['grid_min'] is not None else all_cv_concat.min()
    g_max = config['grid_max'] if config['grid_max'] is not None else all_cv_concat.max()
    margin = 0.05 * (g_max - g_min)
    shared_grid = np.linspace(g_min - margin, g_max + margin, config['grid_points'])

    print("\n" + "=" * 70)
    print("BOOTSTRAP RESAMPLING")
    print("=" * 70)

    results = []
    for i, (inp, ds) in enumerate(zip(config['inputs'], input_datasets)):
        tag = f"input{i+1}"
        cache_file = output_dir / f'{tag}_bootstrap.npz'

        if i in cached_bootstraps:
            print(f"\n--- Input {i+1}: Using cached bootstrap ---")
            fe_mean = cached_bootstraps[i]['fe_mean']
            fe_std = cached_bootstraps[i]['fe_std']
        else:
            print(f"\n--- Input {i+1}: {inp['name']} ---")
            fe_mean, fe_std, bootstrap_samples = bootstrap_free_energy_multicore(
                ds['cv'], ds['weights'], shared_grid, config
            )
            np.savez(cache_file,
                     grid=shared_grid, fe_mean=fe_mean, fe_std=fe_std,
                     bootstrap_samples=bootstrap_samples)
            print(f"  Cached to {cache_file}")

        results.append({
            'name': inp['name'],
            'color': inp['color'],
            'linestyle': inp['linestyle'],
            'barrier_energy': inp['barrier_energy'],
            'stuck': inp.get('stuck', False),
            'cv': ds['cv'],
            'weights': ds['weights'],
            'fe_mean': fe_mean,
            'fe_std': fe_std,
        })

    return shared_grid, results


def clip_plateau(grid, fe_mean, fe_std, threshold_frac=0.05):
    """
    Mask out flat plateau regions. Returns clipped arrays with NaN outside
    the sampled basin, plus the indices of the clip boundaries.
    """
    fe_range = fe_mean.max() - fe_mean.min()
    threshold = fe_mean.max() - threshold_frac * fe_range

    below = fe_mean < threshold

    if not np.any(below):
        return (np.full_like(fe_mean, np.nan),
                np.full_like(fe_std, np.nan), 0, 0)

    first_below = np.argmax(below)
    last_below = len(below) - 1 - np.argmax(below[::-1])

    margin = max(1, int(0.01 * len(grid)))
    clip_start = max(0, first_below - margin)
    clip_end = min(len(grid) - 1, last_below + margin)

    fe_clipped = np.full_like(fe_mean, np.nan)
    std_clipped = np.full_like(fe_std, np.nan)
    fe_clipped[clip_start:clip_end + 1] = fe_mean[clip_start:clip_end + 1]
    std_clipped[clip_start:clip_end + 1] = fe_std[clip_start:clip_end + 1]

    return fe_clipped, std_clipped, clip_start, clip_end


def plot_line_with_fade(ax, grid, fe, color, linestyle, linewidth, label,
                        clip_start, clip_end, fade_length, zorder=3):
    valid = ~np.isnan(fe)
    if not np.any(valid):
        return

    idx_start = np.argmax(valid)
    idx_end = len(valid) - 1 - np.argmax(valid[::-1])

    x = grid[idx_start:idx_end + 1]
    y = fe[idx_start:idx_end + 1]
    n_pts = len(x)

    alphas = np.ones(n_pts)
    fade = min(fade_length, n_pts // 3) 

    if fade > 1:
        alphas[:fade] = np.linspace(0, 1, fade)
        alphas[-fade:] = np.linspace(1, 0, fade)

    rgba = mcolors.to_rgba(color)
    segments = []
    colors = []
    for j in range(n_pts - 1):
        seg = [(x[j], y[j]), (x[j + 1], y[j + 1])]
        segments.append(seg)
        avg_alpha = (alphas[j] + alphas[j + 1]) / 2
        colors.append((*rgba[:3], avg_alpha))

    lc = mcoll.LineCollection(segments, colors=colors, linewidths=linewidth,
                              linestyle=linestyle, capstyle='round',
                              zorder=zorder, label=label)
    ax.add_collection(lc)


def fill_between_with_fade(ax, grid, fe, std, color, base_alpha,
                           clip_start, clip_end, fade_length):
    """
    Error band that fades at the clipped edges, matching the line fade.
    """
    valid = ~np.isnan(fe)
    if not np.any(valid):
        return

    idx_start = np.argmax(valid)
    idx_end = len(valid) - 1 - np.argmax(valid[::-1])

    x = grid[idx_start:idx_end + 1]
    y = fe[idx_start:idx_end + 1]
    s = std[idx_start:idx_end + 1]
    n_pts = len(x)

    alphas = np.ones(n_pts) * base_alpha
    fade = min(fade_length, n_pts // 3)

    if fade > 1:
        alphas[:fade] = np.linspace(0, base_alpha, fade)
        alphas[-fade:] = np.linspace(base_alpha, 0, fade)

    chunk_size = max(1, fade // 10)
    rgba = mcolors.to_rgba(color)

    for start in range(0, n_pts - 1, chunk_size):
        end = min(start + chunk_size + 1, n_pts)
        chunk_alpha = float(np.mean(alphas[start:end]))
        ax.fill_between(x[start:end], y[start:end] - s[start:end],
                         y[start:end] + s[start:end],
                         color=(*rgba[:3], chunk_alpha),
                         linewidth=0, zorder=2)




def create_overlay_figure(grid, results, config):
    print("\n" + "=" * 70)
    print("CREATING REFINED OVERLAY FIGURE")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(config['figure_width'], config['figure_height']))

    threshold_frac = config.get('plateau_threshold_frac', 0.05)
    fade_length = config.get('fade_length', 150)
    if config['annotate_states']:
        sigma = config.get('state_sigma', 0.3)
        for state_name, state_info in config['states'].items():
            pos = state_info['position']
            ax.axvline(pos, color='#AAAAAA', linestyle='--', linewidth=1.4,
                       alpha=0.6, zorder=1)

    for res in results:
        if res['stuck']:
            fe_clip, std_clip, cs, ce = clip_plateau(
                grid, res['fe_mean'], res['fe_std'], threshold_frac
            )
            plot_line_with_fade(
                ax, grid, fe_clip,
                color=res['color'], linestyle=res['linestyle'],
                linewidth=3.5, label=res['name'],
                clip_start=cs, clip_end=ce,
                fade_length=fade_length, zorder=3,
            )
            fill_between_with_fade(
                ax, grid, fe_clip, std_clip,
                color=res['color'],
                base_alpha=config.get('error_alpha_stuck', 0.25),
                clip_start=cs, clip_end=ce,
                fade_length=fade_length,
            )
            print(f"  Clipped + faded: {res['name']}")

        else:
            ax.plot(grid, res['fe_mean'],
                    color=res['color'], linestyle=res['linestyle'],
                    linewidth=3.5, label=res['name'], zorder=3,
                    solid_capstyle='round')
            ax.fill_between(grid,
                            res['fe_mean'] - res['fe_std'],
                            res['fe_mean'] + res['fe_std'],
                            color=res['color'],
                            alpha=config['error_alpha'],
                            zorder=2, linewidth=0)

    y_min = config.get('y_min', 0)
    y_max = config.get('y_max', None)
    if y_max is None:
        y_max = max(r['fe_mean'].max() for r in results) + 5
    ax.set_ylim(y_min, y_max)

    x_min = config.get('x_min', None)
    x_max = config.get('x_max', None)
    if x_min is not None or x_max is not None:
        cur_xmin, cur_xmax = ax.get_xlim()
        ax.set_xlim(x_min if x_min is not None else cur_xmin,
                    x_max if x_max is not None else cur_xmax)

    if config['annotate_states']:
        for state_name, state_info in config['states'].items():
            pos = state_info['position']
            label_x = pos + state_info.get('label_offset', 0)
            y_label = y_min + (y_max - y_min) * 0.96
            ax.text(label_x, y_label, state_name,
                    ha='center', va='top', fontsize=12,
                    color='#555555', style='italic')

    ax.set_xlabel('DeepTDA Collective Variable', fontsize=16)
    ax.set_ylabel('Free Energy (kJ/mol)', fontsize=16)

    import matplotlib.lines as mlines
    handles = []
    for res in results:
        h = mlines.Line2D([], [], color=res['color'],
                          linestyle=res['linestyle'], linewidth=3.5,
                          label=res['name'])
        handles.append(h)

    legend = ax.legend(
        handles=handles,
        loc='upper center', bbox_to_anchor=(0.5, -0.10),
        ncol=3, frameon=True, fancybox=False,
        edgecolor='#666666', framealpha=1.0, facecolor='white',
        fontsize=11, handlelength=2.5, columnspacing=1.5,
    )
    legend.get_frame().set_linewidth(1.0)

    ax.minorticks_on()

    if config['show_grid']:
        ax.grid(True, alpha=0.12, linestyle='-', linewidth=0.5,
                which='major', color='gray')
        ax.grid(True, alpha=0.06, linestyle='-', linewidth=0.3,
                which='minor', color='gray')
        ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color('#333333')

    plt.tight_layout()
    return fig



def save_figure(fig, config, suffix=''):
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    base_name = config['figure_name'] + suffix

    for fmt, ext_dpi in [('png', 300), ('pdf', None), ('svg', None)]:
        path = output_dir / f"{base_name}.{fmt}"
        kwargs = dict(format=fmt, bbox_inches='tight', facecolor='white')
        if ext_dpi:
            kwargs['dpi'] = ext_dpi
        fig.savefig(path, **kwargs)
        print(f"  {fmt.upper()} saved: {path}")

    plt.close(fig)


def save_combined_data(grid, results, config):
    output_dir = Path(config['output_dir'])
    header_parts = ['cv']
    columns = [grid]
    for i, res in enumerate(results):
        tag = f"input{i+1}"
        header_parts += [f'FE_{tag}(kJ/mol)', f'SE_{tag}(kJ/mol)']
        columns += [res['fe_mean'], res['fe_std']]

    out_array = np.column_stack(columns)
    out_file = output_dir / 'free_energy_3inputs.dat'
    np.savetxt(out_file, out_array, header=' '.join(header_parts), fmt='%.6f')
    print(f"  Combined data saved: {out_file}")


def create_statistics_summary(grid, results, config):
    output_dir = Path(config['output_dir'])
    summary_file = output_dir / 'free_energy_3inputs_statistics.txt'

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("FREE ENERGY LANDSCAPE COMPARISON — 3 INPUTS\n")
        f.write("=" * 70 + "\n\n")

        for i, res in enumerate(results):
            f.write(f"{'─' * 50}\n")
            f.write(f"Input {i+1}: {res['name']}\n")
            f.write(f"{'─' * 50}\n")
            f.write(f"  FE min:  {res['fe_mean'].min():.3f} kJ/mol\n")
            f.write(f"  FE max:  {res['fe_mean'].max():.3f} kJ/mol\n")
            f.write(f"  FE range: {res['fe_mean'].max() - res['fe_mean'].min():.3f} kJ/mol\n")
            f.write(f"  Mean SE: {res['fe_std'].mean():.3f} kJ/mol\n")
            f.write(f"  Max SE:  {res['fe_std'].max():.3f} kJ/mol\n")
            n_eff = 1 / np.sum(res['weights'] ** 2)
            f.write(f"  Samples: {len(res['cv']):,}\n")
            f.write(f"  Effective samples: {n_eff:.1f}\n")
            f.write(f"  CV range: [{res['cv'].min():.3f}, {res['cv'].max():.3f}]\n")
            f.write(f"  CV mean (weighted): "
                    f"{np.average(res['cv'], weights=res['weights']):.3f}\n\n")

        if config['annotate_states']:
            f.write("\n" + "=" * 70 + "\n")
            f.write("STATE-SPECIFIC FREE ENERGIES\n")
            f.write("=" * 70 + "\n\n")

            for state_name, state_info in config['states'].items():
                pos = state_info['position']
                idx = np.argmin(np.abs(grid - pos))
                f.write(f"{state_name} (CV ≈ {pos}):\n")
                for i, res in enumerate(results):
                    f.write(f"  Input {i+1}: {res['fe_mean'][idx]:.3f} ± "
                            f"{res['fe_std'][idx]:.3f} kJ/mol\n")
                f.write("\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("FREE ENERGY BARRIERS\n")
            f.write("=" * 70 + "\n\n")

            state_list = list(config['states'].keys())
            for si, s1 in enumerate(state_list):
                for s2 in state_list[si + 1:]:
                    pos1 = config['states'][s1]['position']
                    pos2 = config['states'][s2]['position']
                    idx1 = np.argmin(np.abs(grid - pos1))
                    idx2 = np.argmin(np.abs(grid - pos2))

                    f.write(f"{s1} ↔ {s2}:\n")
                    for i, res in enumerate(results):
                        lo, hi = min(idx1, idx2), max(idx1, idx2)
                        barrier_region = res['fe_mean'][lo:hi + 1]
                        barrier_idx = lo + np.argmax(barrier_region)
                        bh1 = res['fe_mean'][barrier_idx] - res['fe_mean'][idx1]
                        bh2 = res['fe_mean'][barrier_idx] - res['fe_mean'][idx2]
                        f.write(f"  Input {i+1}: "
                                f"from {s1} = {bh1:.3f} kJ/mol, "
                                f"from {s2} = {bh2:.3f} kJ/mol, "
                                f"barrier CV = {grid[barrier_idx]:.3f}\n")
                    f.write("\n")

    print(f"  Statistics saved: {summary_file}")




def main():
    config = CONFIG

    print(f"Working directory: {os.path.abspath('.')}")
    for i, inp in enumerate(config['inputs']):
        stuck_tag = " [STUCK]" if inp.get('stuck') else ""
        print(f"  Input {i+1}: {inp['name']}{stuck_tag}")

    grid, results = process_all_inputs(config)

    fig = create_overlay_figure(grid, results, config)
    print("\nSaving figure...")
    save_figure(fig, config, suffix='_overlay')

    print("\nSaving data...")
    save_combined_data(grid, results, config)
    create_statistics_summary(grid, results, config)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f" Output directory: {config['output_dir']}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
