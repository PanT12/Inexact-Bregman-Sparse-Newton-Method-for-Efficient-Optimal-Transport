import matplotlib.pyplot as plt
from Alg.Algorithm_Import import *
plt.rc('font', family='Times New Roman')
plt.rcParams.update({
    'figure.figsize': (8, 5.5),
    'figure.subplot.left': 0.15,
    'figure.subplot.right': 0.95,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.top': 0.95,
    # ===== Title =====
    'axes.titlesize': 28,

    # ===== Axis labels =====
    'axes.labelsize': 28,

    # ===== Tick labels =====
    'xtick.labelsize': 23,
    'ytick.labelsize': 25,

    # ===== Legend =====
    'legend.fontsize': 19,
    'legend.frameon': True,
    'legend.framealpha': 0.8,

    # ===== Lines =====
    'lines.linewidth': 3,
})


def plot_from_loaded(result_root, data, ylog=True, show_bands=True, title_prefix="", title_suffix="",
                     downsample_map=None, default_stride=1, solid_algos=None, figure_show=True):
    if not data:
        print("No data to plot.")
        return

    downsample_map = downsample_map or {}
    interest = 'grad_norm'
    def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["iter", "time", interest]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["iter", interest])
        if "time" in df.columns:
            t = df["time"].to_numpy(dtype=float)
            if t.size > 1 and not np.all(np.diff(t) >= 0):
                df["time"] = np.cumsum(t)
        return df.sort_values("iter")

    keys_to_skip = {"IBSN_EOT_skip1"}
    algos = [key for key in data.keys() if key not in keys_to_skip]
    algos.remove("(E)IBSN")
    algos.append("(E)IBSN")
    cmap = plt.cm.get_cmap("tab10", len(algos))
    color_map = {name: cmap(i) for i, name in enumerate(algos)}

    if isinstance(solid_algos, str):
        solid_set = {solid_algos}
    elif solid_algos is None:
        solid_set = set()
    else:
        solid_set = set(solid_algos)

    def ls(name: str) -> str:
        return "-" if name in solid_set else "--"

    # ========== Gradient Norm vs Iteration ==========
    for name in algos:
        print(name)
        stride = int(downsample_map.get(name, default_stride))
        # for d in data[name]:
        #     sanitize_df(d)
        dfs = [sanitize_df(d) for d in data[name]]

        curves = []
        for d in dfs:
            if stride > 1:
                d = d.iloc[::stride, :].reset_index(drop=True)
            arr = d[interest].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                curves.append(arr)

        if len(curves) == 0:
            continue

        Lmin = min(len(c) for c in curves)
        if Lmin == 0:
            continue
        M = np.vstack([c[:Lmin] for c in curves])

        mean_err = M.mean(0)
        std_err  = M.std(0)
        x = np.arange(Lmin)
        plt.plot(x, mean_err, color=color_map[name], linestyle=ls(name), label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    if ylog:
        plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title(f"{title_prefix} {title_suffix}")
    plt.tight_layout()
    plt.savefig(result_root + f"/alg_compare_iter_EOT_{title_prefix}_{title_suffix}.pdf")
    if figure_show:
        plt.show()

    # ========== Gradient Norm vs Time ==========
    for name in algos:
        dfs = [sanitize_df(d) for d in data[name]]

        times_list, errs_list = [], []
        for d in dfs:
            if "time" not in d.columns:
                continue
            t = d["time"].to_numpy(dtype=float)
            e = d[interest].to_numpy(dtype=float)
            finite = np.isfinite(t) & np.isfinite(e)
            t, e = t[finite], e[finite]
            if t.size == 0 or e.size == 0:
                continue
            L = min(len(t), len(e))
            if L > 0:
                times_list.append(t[:L])
                errs_list.append(e[:L])

        if len(times_list) == 0:
            continue

        Lmin = min(len(t) for t in times_list)
        T = np.vstack([t[:Lmin] for t in times_list])
        E = np.vstack([e[:Lmin] for e in errs_list])

        mean_t = T.mean(0)
        mean_e = E.mean(0)
        std_e  = E.std(0)

        plt.plot(mean_t, mean_e, color=color_map[name], linestyle=ls(name), label=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Gradient Norm")
    if ylog:
        plt.yscale("log")
    if exp == "synthetic":
        if m == 1000: right = 5
        elif m == 5000: right = 200
        elif m == 10000: right = 1000
        else: right = None
    elif exp == "real":
        if size == 32: right = 4
        elif size == 64: right = None
    plt.xlim(left=0, right=right)
    plt.ylim(top=1)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.title(f"{title_prefix} {title_suffix}")
    plt.tight_layout()
    plt.savefig(result_root + f"/alg_compare_time_EOT_{title_prefix}_{title_suffix}.pdf")
    if figure_show:
        plt.show()



exp = "synthetic"

# experiment settings
if exp == "synthetic":
    m = n = 1000
    title_prefix = cost_matrix_norm = "Square"  # "Square" or "Uniform"
    RESULT_ROOT = f"Results/cg/Synthetic/cg_compare_m{m}_n{n}"
    print(f"Result root is {RESULT_ROOT}")
    RESULT_ROOT = os.path.join(RESULT_ROOT, str(cost_matrix_norm))
    title_suffix = f"(m={m}, n={n})"
elif exp == "real":
    experiment_id = 1  # 1,2,3
    size = 32  # only for DOTmark
    ID = {"DOTmark": {1: "Shapes",
                      2: "ClassicImages",
                      3: "MicroscopyImages",
                      4: "WhiteNoise",
                      5: "GRFrough"}}
    category = ID["DOTmark"][experiment_id]
    RESULT_ROOT = f"Results/cg/Real/DOTmark_{category}_size{size}"
    print(f"Result root is {RESULT_ROOT}")
    title_prefix = f"{category}{size}"
    title_suffix = f"(ID1=2, ID2=4)"
else:
    raise ValueError(f"Unknown")
os.makedirs(RESULT_ROOT, exist_ok=True)

loaded = load_results(RESULT_ROOT)
print(f"Loaded results for algorithms: {list(loaded.keys())}")

downsample = {
    "ExtraGrad": 20,
}
plot_from_loaded(
    RESULT_ROOT,
    loaded,
    ylog=True,
    show_bands=True,
    title_prefix=title_prefix,
    title_suffix=title_suffix,
    downsample_map=downsample,
    default_stride=1,
    solid_algos="(E)IBSN",
    figure_show=True
)
