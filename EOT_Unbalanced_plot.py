import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import ast

plt.rc('font', family='Times New Roman')
plt.rcParams.update({
    'figure.figsize': (8, 5.5),     
    'axes.titlesize': 28,         
    'axes.labelsize': 28,         
    'xtick.labelsize': 23,        
    'ytick.labelsize': 25,        
    'legend.fontsize': 19,        
    'lines.linewidth': 3,         
    'lines.markersize': 12,
    
    'figure.subplot.left': 0.15,
    'figure.subplot.right': 0.95,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.top': 0.95,
    
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',       
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',  
})

def parse_cg_iter(cg_iter_str):
    try:
        if pd.isna(cg_iter_str): return []
        if isinstance(cg_iter_str, list): return cg_iter_str
        if isinstance(cg_iter_str, str):
            s = cg_iter_str.strip()
            if not s or s.lower() == 'nan' or s.lower() == 'none':
                return []
            val = ast.literal_eval(s)
            if isinstance(val, list): return val
        return []
    except:
        return []

def calculate_stats_robust(df):
    stats = {
        'avg_time_per_step': np.nan,
        'avg_cg_iter': np.nan,
        'sem_cg_iter': np.nan,
        'total_time': df['time'].iloc[-1] if not df.empty else 0,
        'iter_count': df['iter'].iloc[-1] if not df.empty else 0,
        'final_grad_norm': np.nan 
    }
    
    if df.empty: return stats
    if 'grad_norm' in df.columns:
        stats['final_grad_norm'] = df['grad_norm'].iloc[-1]
    elif 'gnorm' in df.columns: 
        stats['final_grad_norm'] = df['gnorm'].iloc[-1]
    else:
        stats['final_grad_norm'] = 0.0 

    # --- 2. Calculate Avg CG Iterations & SEM ---
    if 'cg_iter' in df.columns and len(df) > 2:
        cg_iter_series = df['cg_iter'].iloc[2:] 
        
        try:
            cg_arr = pd.to_numeric(cg_iter_series, errors='coerce').to_numpy()
            valid_numeric = cg_arr[~np.isnan(cg_arr)]
            if len(valid_numeric) > 0:
                stats['avg_cg_iter'] = np.mean(valid_numeric)
                if len(valid_numeric) > 1:
                    stats['sem_cg_iter'] = np.std(valid_numeric, ddof=1) / np.sqrt(len(valid_numeric))
                else:
                    stats['sem_cg_iter'] = 0.0
        except:
            pass

        if np.isnan(stats['avg_cg_iter']):
            all_cg = []
            for val in cg_iter_series:
                parsed = parse_cg_iter(val)
                if parsed:
                    all_cg.extend(parsed)
                else:
                    try:
                        v = float(val)
                        if v > 0: all_cg.append(v)
                    except:
                        pass
            
            if len(all_cg) > 0:
                stats['avg_cg_iter'] = np.mean(all_cg)
                if len(all_cg) > 1:
                    stats['sem_cg_iter'] = np.std(all_cg, ddof=1) / np.sqrt(len(all_cg))
                else:
                    stats['sem_cg_iter'] = 0.0

    # --- 3. Calculate Avg Time per Step ---
    if 'newton_iter' in df.columns and df['newton_iter'].max() > 0:
        newton_phase = df[df['newton_iter'] > 0]
        if not newton_phase.empty:
            start_time = newton_phase['time'].iloc[0]
            end_time = df['time'].iloc[-1]
            steps = df['newton_iter'].max()
            if steps > 0:
                stats['avg_time_per_step'] = (end_time - start_time) / steps
    
    if np.isnan(stats['avg_time_per_step']) and stats['iter_count'] > 0:
        stats['avg_time_per_step'] = stats['total_time'] / stats['iter_count']

    return stats

def load_all_experiments(base_root, prefix="earth_"):
    summary_list = []
    sub_dirs = glob.glob(os.path.join(base_root, f'{prefix}*'))
    
    detected_algos = set()

    for d in sub_dirs:
        try:
            folder_name = os.path.basename(d)
            n_val = int(folder_name.split('_')[-1])
        except ValueError:
            continue

        csv_files = glob.glob(os.path.join(d, "*.csv"))
        for csv_path in csv_files:
            alg_name = os.path.splitext(os.path.basename(csv_path))[0]
            detected_algos.add(alg_name)
            
            try:
                df = pd.read_csv(csv_path)
                if df.empty: continue
                
                stats = calculate_stats_robust(df)
                
                summary_list.append({
                    'n_demand': n_val,
                    'algo': alg_name,
                    'total_time': stats['total_time'],
                    'avg_cg_iter': stats['avg_cg_iter'],
                    'sem_cg_iter': stats['sem_cg_iter'],
                    'final_grad_norm': stats['final_grad_norm'],
                    'avg_time_per_step': stats['avg_time_per_step'] 
                })
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

    df_summary = pd.DataFrame(summary_list)
    if not df_summary.empty:
        df_summary = df_summary.sort_values(by='n_demand')
    
    return df_summary, sorted(list(detected_algos))

def generate_style_map(all_algos):
    DESIRED_ORDER = ["Sinkhorn", "SSNS", "SPLR", "(E)PINS", "(E)IBSN"]
    
    algos_sorted = [alg for alg in DESIRED_ORDER if alg in all_algos]
    remaining = [alg for alg in all_algos if alg not in algos_sorted]
    algos_sorted.extend(remaining)

    cmap = plt.cm.get_cmap("tab10", len(algos_sorted))
    color_map = {name: cmap(i) for i, name in enumerate(algos_sorted)}
    
    markers_list = ['o', '^', 's', 'd', 'v', '<', '>']
    marker_map = {name: markers_list[i % len(markers_list)] for i, name in enumerate(algos_sorted)}

    style_map = {}
    for name in algos_sorted:
        style_map[name] = {
            'color': color_map[name],
            'ls': '-' if name == "(E)IBSN" else '--', 
            'marker': marker_map[name]
        }
        
    return style_map, algos_sorted

def generate_detailed_csv(df_summary, output_dir, baseline_alg='(E)IBSN'):
    if df_summary.empty: return

    p_total = df_summary.pivot(index='n_demand', columns='algo', values='total_time')
    p_step  = df_summary.pivot(index='n_demand', columns='algo', values='avg_time_per_step')
    p_cg    = df_summary.pivot(index='n_demand', columns='algo', values='avg_cg_iter')

    if baseline_alg not in p_total.columns:
        print(f"[Warning] Baseline algorithm '{baseline_alg}' not found. Skipping CSV generation.")
        return

    ratio_dfs = {}
    df_time_ratio = p_total.div(p_total[baseline_alg], axis=0)
    ratio_dfs['Total Time Ratio'] = df_time_ratio

    if not p_step.empty and baseline_alg in p_step.columns:
        df_step_ratio = p_step.div(p_step[baseline_alg], axis=0)
        ratio_dfs['Time/Step Ratio'] = df_step_ratio

    if not p_cg.empty and baseline_alg in p_cg.columns:
        df_cg_ratio = p_cg.div(p_cg[baseline_alg], axis=0)
        ratio_dfs['CG Iter Ratio'] = df_cg_ratio

    full_table = pd.concat(ratio_dfs.values(), axis=1, keys=ratio_dfs.keys())
    full_table = full_table.round(2)
    
    csv_path = os.path.join(output_dir, "detailed_ratios_table.csv")
    full_table.to_csv(csv_path)
    print(f"\n[CSV Generated] Detailed ratios saved to: {csv_path}")

def generate_latex_table_from_csv(result_root, csv_name="detailed_ratios_table.csv", specific_n_list=None):
    csv_path = os.path.join(result_root, csv_name)
    if not os.path.exists(csv_path):
        print(f"[Warning] Ratio CSV not found at: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        
        metric = 'Total Time Ratio'
        found_metric = None
        for col_level0 in df.columns.levels[0]:
            if 'Time Ratio' in col_level0:
                found_metric = col_level0
                break
        
        if not found_metric:
            print(f"[Error] Metric related to 'Time Ratio' not found in CSV.")
            return
            
        sub_df = df[found_metric]
        all_n_in_csv = sorted([int(x) for x in sub_df.index.unique()])
        
        if specific_n_list is not None and len(specific_n_list) > 0:
            selected_n = []
            for n in specific_n_list:
                if n in all_n_in_csv:
                    selected_n.append(n)
        else:
            selected_n = all_n_in_csv
            
        if not selected_n:
            return
            
        print("\n" + "="*60)
        print("GENERATED LATEX TABLE CODE")
        print("="*60 + "\n")
        
        print(r"\begin{table}[h]")
        print(r"\centering")
        print(r"\caption{Computation time ratios relative to (E)IBSN.}")
        print(r"\label{tab:nonsquare_ratios}")
        
        cols_format = "l" + "c" * len(selected_n)
        print(r"\begin{tabular}{" + cols_format + "}")
        print(r"\toprule")
        
        header_str = "Algorithm"
        for n_val in selected_n:
            header_str += f" & $n={n_val}$"
        print(header_str + r" \\")
        print(r"\midrule")
        
        desired_order = ["Sinkhorn", "SSNS", "SPLR", "(E)PINS", "(E)IBSN"]
        avail_algos = sub_df.columns.tolist()
        sorted_algos = [a for a in desired_order if a in avail_algos]
        for a in avail_algos:
            if a not in sorted_algos:
                sorted_algos.append(a)
        
        for alg in sorted_algos:
            row_str = f"{alg}"
            for n_val in selected_n:
                try:
                    val = sub_df.loc[n_val, alg]
                    if isinstance(val, pd.Series): val = val.iloc[0]
                    
                    if pd.isna(val):
                        row_str += " & -"
                    else:
                        if val >= 100:
                            row_str += f" & {val:.0f}"
                        elif val >= 10:
                            row_str += f" & {val:.1f}"
                        else:
                            row_str += f" & {val:.2f}"
                except KeyError:
                    row_str += " & -"
            print(row_str + r" \\")
            
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print(f"[Error generating table] {e}")

def plot_three_paper_figures(df, output_dir, m_fixed=50, all_detected_algos=None, tol=1e-9):
    os.makedirs(output_dir, exist_ok=True)
    
    style_map, sorted_algos = generate_style_map(all_detected_algos)
    
    df_time = df.pivot(index='n_demand', columns='algo', values='total_time')
    df_cg   = df.pivot(index='n_demand', columns='algo', values='avg_cg_iter')
    df_cg_sem = df.pivot(index='n_demand', columns='algo', values='sem_cg_iter')
    
    df_time.index = df_time.index.astype(int)
    df_cg.index = df_cg.index.astype(int)
    df_cg_sem.index = df_cg_sem.index.astype(int)

    # --------------------------------------------------
    # Plot 1: Time Scaling (Log-Log)
    # --------------------------------------------------
    plt.figure()
    for alg in sorted_algos:
        if alg not in df_time.columns: continue
        series = df_time[alg].dropna()
        if series.empty: continue
        
        s = style_map[alg]
        x_ratio = series.index / m_fixed 
        
        plt.plot(x_ratio, series.values, 
                 label=alg, 
                 color=s['color'], 
                 linestyle=s['ls'], 
                 marker=s['marker']) 
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Ratio $m/n$') 
    plt.ylabel('Time (s)')
    plt.title(f'Time Scaling ($n={m_fixed}$)')
    plt.grid(True, which="major", ls="-", alpha=0.3)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.legend(loc='upper left')
    
    save_path = os.path.join(output_dir, f"Fig_Time_Scaling_m{m_fixed}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"[Generated] {save_path}")
    plt.close()

    # --------------------------------------------------
    # Plot 2: Speedup Ratio vs IBSN
    # --------------------------------------------------
    plt.figure()
    base_alg = '(E)IBSN'
    
    if base_alg in df_time.columns:
        base_series = df_time[base_alg]
        
        for alg in sorted_algos:
            if alg not in df_time.columns: continue
            
            s = style_map[alg]
            
            if alg == base_alg:
                plt.axhline(1, color=s['color'], linestyle='-', linewidth=2, label=alg)
            else:
                series = df_time[alg] / base_series
                series = series.dropna()
                if series.empty: continue
                
                x_ratio = series.index / m_fixed
                
                plt.plot(x_ratio, series.values, 
                         label=alg,
                         color=s['color'], 
                         linestyle=s['ls'], 
                         marker=s['marker'])
        
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(top=1500) 
        
        plt.xlabel(r'Ratio $m/n$')
        # plt.ylabel(r'Speedup Ratio ($T_{\mathrm{Alg.}} / T_{\mathrm{IBSN}}$)')
        plt.ylabel(r'Speedup Ratio')
        plt.title(f'Computational Speedup ($n={m_fixed}$)')
        plt.grid(True, which="major", ls="-", alpha=0.3)
        plt.legend(loc='upper left')
        
        save_path = os.path.join(output_dir, f"Fig_Speedup_Ratio_m{m_fixed}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Generated] {save_path}")
    plt.close()

    # --------------------------------------------------
    # Plot 3: Average CG Iterations
    # --------------------------------------------------
    plt.figure()
    has_cg_data = False
    for alg in sorted_algos:
        if alg == 'Sinkhorn': continue 
        if alg not in df_cg.columns: continue
        
        series = df_cg[alg].dropna()
        if series.empty: continue
        
        has_cg_data = True
        s = style_map[alg]
        x_ratio = series.index / m_fixed
        
        plt.plot(x_ratio, series.values, 
                 label=alg,
                 color=s['color'], 
                 linestyle=s['ls'], 
                 marker=s['marker'])
        
        if alg in df_cg_sem.columns:
            sem_series = df_cg_sem[alg].reindex(series.index).fillna(0)
            
            plt.fill_between(x_ratio, 
                             series.values - sem_series.values, 
                             series.values + sem_series.values,
                             color=s['color'], 
                             alpha=0.2,       
                             edgecolor=None)

    if has_cg_data:
        plt.xscale('log')
        plt.xlabel(r'Ratio $m/n$')
        plt.ylabel('MeanCG')
        plt.title(f'MeanCG Iterations ($n={m_fixed}$)')
        plt.grid(True, which="major", ls="-", alpha=0.3)
        plt.legend(loc='upper left')
        
        save_path = os.path.join(output_dir, f"Fig_CG_Iter_m{m_fixed}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Generated] {save_path}")
    plt.close()


M_FIXED = 50
RESULT_ROOT = f"Results/earth_eot{M_FIXED}_0.001"
OUTPUT_PLOT_DIR = RESULT_ROOT

tol = 1e-9

# CUSTOM_N_LIST = [80,400,800,4000,8000,40000,80000,400000,800000]

print(f"Processing results from: {RESULT_ROOT}")

# 1. Load Data
df_summary, detected_algos = load_all_experiments(RESULT_ROOT)

if not df_summary.empty:
    generate_detailed_csv(df_summary, OUTPUT_PLOT_DIR, baseline_alg='(E)IBSN')

    plot_three_paper_figures(df_summary, OUTPUT_PLOT_DIR,
                             m_fixed=M_FIXED,
                             all_detected_algos=detected_algos,
                             tol=tol)
    print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_PLOT_DIR)}")

    # # 4. Generate LaTeX Table
    # generate_latex_table_from_csv(RESULT_ROOT,
    #                               csv_name="detailed_ratios_table.csv",
    #                               specific_n_list=CUSTOM_N_LIST)

else:
    print("No data found. Please check RESULT_ROOT path.")