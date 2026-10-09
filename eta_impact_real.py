import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Alg.Algorithm_Import import *
from Data_Sample_Import import *
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
                     downsample_map=None, default_stride=1, solid_algos=None):
    if not data:
        print("No data to plot.")
        return

    downsample_map = downsample_map or {}

    interest = 'abs_err'

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

    def legend_show(eta_labels, alg_labels):
        eta_labels = sorted(list(set(eta_labels)))
        alg_labels = sorted(list(set(alg_labels)))
        eta_handles = [
            Line2D([0], [0], color=color_map[eta]) for eta in eta_labels
        ]
        alg_handles = [
            Line2D([0], [0], color="black", linestyle=ls(name))
            for name in alg_labels
        ]

        # 1) Add "Algorithm" title row (fake invisible line)
        title_eta = Line2D([], [], linestyle='none')
        # 2) Add "p" title row (fake invisible line)
        title_alg = Line2D([], [], linestyle='none')

        combined_handles = (
                [title_eta] + eta_handles +
                [title_alg] + alg_handles
        )

        combined_labels = (
                ["$\eta$"] + eta_labels +
                ["Algorithm"] + alg_labels
        )
        return combined_handles, combined_labels

    alg_eta = sorted(data.keys())
    cmap = plt.cm.get_cmap("tab10", len(eta_list))
    color_map = {str(eta): cmap(i) for i, eta in enumerate(eta_list)}

    if isinstance(solid_algos, str):
        solid_set = {solid_algos}
    elif solid_algos is None:
        solid_set = set()
    else:
        solid_set = set(solid_algos)

    def ls(name: str) -> str:
        if name == "IBSN": return "-"
        elif name == "PINS": return "--"
        elif name == "IBSink": return ":"
        else: return "-."
        # return "-" if name in solid_set else "--"

    # def ms(eta) -> str:
    #     if eta == "0.0001": return "s"
    #     elif eta == "0.001": return "o"
    #     elif eta == "0.01": return "^"
    #     else: return "d"

    def ms(name) -> str:
        if name == "IBSN": return "s"
        elif name == "PINS": return "o"
        elif name == "IBSink": return "^"
        else: return "d"

    alg_labels = []
    eta_labels = []
    # ========== Gap vs Iteration ==========
    plt.figure(figsize=(8, 5))
    for algname_eta in alg_eta:
        name, eta = algname_eta.split("_")
        alg_labels.append(name)
        eta_labels.append(eta)
        stride = int(downsample_map.get(name, default_stride))
        dfs = [sanitize_df(d) for d in data[algname_eta]]

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
        plt.plot(x, mean_err, color=color_map[eta], linestyle=ls(name), label=name + r'$(\eta=$' + eta + ")")
        # if show_bands:
        #     plt.fill_between(x, mean_err - std_err, mean_err + std_err,
        #                      color=color_map[name], alpha=0.15)

    plt.xlabel("Iteration")
    plt.ylabel("Gap")
    if ylog:
        plt.yscale("log")
    plt.grid(alpha=0.3)
    combined_handles, combined_labels = legend_show(eta_labels, alg_labels)
    plt.legend(
        handles=combined_handles,
        labels=combined_labels,
        loc="upper right",
        frameon=True,
        ncol=1,
        # handlelength=2.5,
        # handletextpad=0.8,
        # borderpad=0.7,
        # labelspacing=0.6,
    )
    plt.legend(loc="best", ncol=3)

    # plt.show()
    # plt.legend()
    plt.title(f"{title_prefix} {title_suffix}")
    plt.tight_layout()
    plt.savefig(result_root + f"/eta_compare_iter_{title_prefix}_{title_suffix}.pdf")
    plt.show()

    # ========== Gap vs Time ==========
    for algname_eta in alg_eta:
        name, eta = algname_eta.split("_")
        dfs = [sanitize_df(d) for d in data[algname_eta]]

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

        plt.plot(mean_t, mean_e, color=color_map[eta],
                 linestyle=ls(name),
                 marker=ms(name), markersize=8,
                 markevery=max(len(mean_t) // 5, 1),
                 label=name + r'$(\eta=$' + eta + ")")

    plt.xlabel("Time (s)")
    right=None
    plt.xlim(left=0, right=right)
    plt.ylabel("Gap")
    if ylog:
        plt.yscale("log")
    plt.grid(alpha=0.4)
    plt.legend(loc="best", ncol=1, frameon=True, fontsize=15)
    plt.title(f"{title_prefix} {title_suffix}")
    plt.tight_layout()
    plt.savefig(result_root + f"/eta_compare_time_{title_prefix}_{title_suffix}.pdf")
    plt.show()


eta_list = [1e-2, 1e-3]

ID = {
    "FashionMNIST": {1: [0, 1], 2: [2, 54698], 3: [3, 12580]},
    "MNIST": {1: [0, 1], 2: [22, 54698], 3: [333, 12580]},
    "DOTmark": {1: "Shapes", 2: "ClassicImages", 3: "MicroscopyImages", 4: "WhiteNoise"},
}

# choose data and experiment_id
experiment_id = 3
dataset_name = "FashionMNIST"  # MNIST, FashionMNIST, DOTmark
size = 64  # only for DOTmark

this_dir = os.path.dirname(__file__)
default_dir = os.path.join(this_dir, "data", dataset_name)

if dataset_name == "FashionMNIST" or dataset_name == "MNIST":
    pid, qid = ID[dataset_name][experiment_id]
    C, a, b = load_mnist(default_dir, p_id=pid, q_id=qid, norm_type='euclidean', eps=0.01)
    RESULT_ROOT = f"Results/eta_impact/eta_compare_{dataset_name}/pid{pid}_qid{qid}"
    print("maximum value in a:", np.max(a))
elif dataset_name == "DOTmark":
    category = ID[dataset_name][experiment_id]
    pid, qid = 2, 4
    C, a, b = load_dotmark(default_dir, category, size, p_id=pid, q_id=qid, norm_type='euclidean', eps=0.01)
    RESULT_ROOT = f"Results/eta_impact/eta_compare_{dataset_name}/{category}_size{size}"
else:
    raise ValueError(f"Unknown dataset name: {dataset_name}")

time_max = 10000.0
eta = 1e-3
tol = 1e-7
os.makedirs(RESULT_ROOT, exist_ok=True)


opt_cache_path = os.path.join(RESULT_ROOT, "opt_ab.npz")

if os.path.exists(opt_cache_path):
    cache = np.load(opt_cache_path, allow_pickle=False)
    opt = float(cache["opt"])
    print(f"  [cache] loaded a,b,opt from {opt_cache_path}")
else:
    guro = Gurobisolver(C, a, b)
    gt = guro.Gurobi_Solver_original()

    print("row difference is ", np.linalg.norm(a - gt.sum(axis=1)))
    print("col difference is ", np.linalg.norm(b - gt.sum(axis=0)))
    opt = float(np.sum(C * gt))
    print("  [Gurobi] computed opt =", opt)

    np.savez(opt_cache_path, opt=opt)
    print(f"  [cache] saved opt to {opt_cache_path}")

ALGORITHMS = {
    "IBSN": run_IBSN,
    "PINS": run_PINS,
    "IBSink": run_IBSink,
}

for eta in eta_list:
    print(f" Running eta={eta}...")
    for alg_name, fn in ALGORITHMS.items():
        DATA_path = os.path.join(RESULT_ROOT, f"{alg_name}_{eta}.csv")
        if not os.path.exists(DATA_path):
            print(f" Running {alg_name}...")
            X, history, eta_val = fn(C, a, b, eta=eta, tol=1e-11, time_max=time_max, opt=opt)
            df = pd.DataFrame({
                "algo": alg_name,
                "data_name": dataset_name,
                "iter": np.arange(len(history['time']), dtype=int),
                "opt": [opt] * len(history.get("time", [])),
                "eta": [eta_val] * len(history.get("time", [])),
            })
            for key in history:
                if not history[key]:
                    history[key] = [None] * len(history['time'])
            df = pd.concat([df, pd.DataFrame(history)], axis=1)

            out_path = os.path.join(RESULT_ROOT, f"{alg_name}_{eta}.csv")
            df.to_csv(out_path, index=False)
            print(f"  saved -> {out_path}")


loaded = load_results(RESULT_ROOT)
print(f"Loaded results for algorithms: {list(loaded.keys())}")

plot_from_loaded(
    RESULT_ROOT,
    loaded,
    ylog=True,
    show_bands=True,
    title_prefix=f"{dataset_name}" if dataset_name != "DOTmark" else f"{category}{size}",
    title_suffix=f"(ID1={pid}, ID2={qid})",
    default_stride=1,
    solid_algos="IBSN",
)
