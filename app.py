"""Fleet health dashboard with a dataset switcher (FD001-FD004), a results-summary section, and a separate,
non-technical "Understand this project" explainability page.
Run:  streamlit run app.py
Needs models/<FD>_best.joblib (made by run_experiments.py) and the raw NASA files in data/."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data import N_COND, load_fd

st.set_page_config(page_title="Engine RUL Dashboard", layout="wide", page_icon="\u2708\ufe0f")

# ----------------------------------------------------------------------------------------------------------------
# styling: a bit of custom CSS so the stock Streamlit look feels less like a default template
# ----------------------------------------------------------------------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.6rem; max-width: 1250px; }
h1 { font-weight: 800 !important; letter-spacing: -0.5px; }
h2, h3 { font-weight: 700 !important; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(120,140,255,0.08), rgba(120,140,255,0.02));
    border: 1px solid rgba(130,150,255,0.18);
    border-radius: 14px;
    padding: 14px 16px 10px 16px;
}
div[data-testid="stMetricValue"] { font-weight: 800; }
.badge {
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.3px; text-align: center; min-width: 78px;
}
.badge-critical { background: rgba(255, 75, 75, 0.18); color: #ff6b6b; border: 1px solid rgba(255,75,75,0.4); }
.badge-warning  { background: rgba(255, 180, 50, 0.18); color: #ffb432; border: 1px solid rgba(255,180,50,0.4); }
.badge-ok       { background: rgba(60, 200, 130, 0.18); color: #3cc882; border: 1px solid rgba(60,200,130,0.4); }
.hero {
    background: linear-gradient(135deg, rgba(120,140,255,0.14), rgba(120,140,255,0.02));
    border: 1px solid rgba(130,150,255,0.18); border-radius: 18px; padding: 22px 26px; margin-bottom: 18px;
}
.pill {
    display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    background: rgba(130,150,255,0.14); border: 1px solid rgba(130,150,255,0.3); margin: 2px 6px 2px 0;
}
.explain-card {
    background: rgba(130,150,255,0.05); border: 1px solid rgba(130,150,255,0.15);
    border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
}
.step-num {
    display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px;
    border-radius: 50%; background: rgba(130,150,255,0.25); font-weight: 800; margin-right: 10px;
}
/* the fleet table is injected as raw HTML (so the colour badges can render inside cells), and raw HTML tables
   get NO styling from Streamlit by default -- scoped to .fleet-table so it can't affect st.dataframe elsewhere */
/* overflow-x:auto + a min-width on the inner table means columns scroll horizontally in a narrow container
   instead of being crushed to the point where badge text like "CRITICAL" gets clipped or words run together */
.fleet-table { border: 1px solid rgba(130,150,255,0.18); border-radius: 12px; margin-bottom: 4px; overflow-x: auto; }
.fleet-table table { width: 100%; min-width: 480px; border-collapse: collapse; font-size: 0.9rem; table-layout: auto; }
.fleet-table thead th {
    background: rgba(130,150,255,0.16); text-align: left !important; white-space: nowrap;
    padding: 10px 16px !important; font-weight: 700; border-bottom: 1px solid rgba(130,150,255,0.25);
}
.fleet-table tbody td { padding: 10px 16px !important; border-bottom: 1px solid rgba(255,255,255,0.06); white-space: nowrap; }
.fleet-table tbody tr:last-child td { border-bottom: none; }
.fleet-table tbody tr:hover { background: rgba(130,150,255,0.06); }
.fleet-table tbody tr:nth-child(even) { background: rgba(255,255,255,0.015); }
.gallery-missing {
    background: rgba(255,255,255,0.03); border: 1px dashed rgba(255,255,255,0.15); border-radius: 12px;
    padding: 18px; text-align: center; opacity: 0.7; font-size: 0.85rem; height: 100%;
    display: flex; flex-direction: column; justify-content: center;
}
</style>
""", unsafe_allow_html=True)

ALL_FDS = ["FD001", "FD002", "FD003", "FD004"]
DESC = {"FD001": "1 operating condition, 1 fault mode", "FD002": "6 operating conditions, 1 fault mode",
        "FD003": "1 operating condition, 2 fault modes", "FD004": "6 operating conditions, 2 fault modes"}
trained = [f for f in ALL_FDS if Path(f"models/{f}_best.joblib").exists()]


def badge(status):
    cls = {"CRITICAL": "badge-critical", "WARNING": "badge-warning", "OK": "badge-ok"}[status]
    return f'<span class="badge {cls}">{status}</span>'


# ----------------------------------------------------------------------------------------------------------------
# sidebar: dataset controls + the one button that switches the whole page into the explainability view
# ----------------------------------------------------------------------------------------------------------------
st.sidebar.markdown("### \u2708\ufe0f Engine RUL")
page = st.sidebar.radio("View", ["\U0001F4CA Dashboard", "\U0001F4D6 Understand this project"], label_visibility="collapsed")
st.sidebar.divider()

DATA_DIR = st.sidebar.text_input("Data folder", "data")
fd = st.sidebar.selectbox("Dataset", ALL_FDS,
                          format_func=lambda f: f"{f}  \u2713 trained" if f in trained else f"{f}  (not trained)")
st.sidebar.caption(DESC[fd])
has_conservative = Path(f"models/{fd}_conservative.joblib").exists()
use_conservative = False
if has_conservative:
    use_conservative = st.sidebar.toggle(
        "\U0001F6E1\ufe0f Use risk-adjusted (conservative) model", value=False,
        help="Swaps in a model that deliberately predicts LOWER remaining life, to cut late predictions -- "
             "trades some accuracy for safety. See Results achieved for the exact trade-off.")
crit = st.sidebar.slider("Critical threshold (cycles)", 5, 60, 20)
warn = st.sidebar.slider("Warning threshold (cycles)", 20, 120, 50)


@st.cache_resource
def get_model(fd, conservative=False):
    suffix = "conservative" if conservative else "best"
    return joblib.load(f"models/{fd}_{suffix}.joblib")


@st.cache_data
def get_fleet(fd, data_dir, conservative=False):
    pred = get_model(fd, conservative)
    _, test, rul_true = load_fd(data_dir, fd)
    last = pred.predict_raw(test, last_only=True)
    last["true_rul"] = rul_true
    return test, last


# ==================================================================================================================
# PAGE 1: the "Understand this project" explainability page -- written for someone outside data science
# ==================================================================================================================
if page.startswith("\U0001F4D6"):
    st.markdown('<div class="hero"><h1>\u2708\ufe0f Understanding this project</h1>'
               '<p style="font-size:1.05rem; opacity:0.85;">A plain-language walkthrough: what the problem is, '
               'what data we used, how the model works, and how good it is -- no data-science background needed.</p>'
               '</div>', unsafe_allow_html=True)

    st.markdown("## 1. The problem, in one sentence")
    st.markdown('<div class="explain-card">Jet engines wear out gradually. If we can predict '
               '<b>how many flights (cycles) an engine has left before it needs maintenance</b>, we can service it '
               'just in time -- not so early that we waste a perfectly good engine, and not so late that it fails '
               'in the air. This is called <b>Remaining Useful Life (RUL)</b> prediction.</div>', unsafe_allow_html=True)

    st.markdown("## 2. The data")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="explain-card"><b>Where it comes from</b><br>NASA ran simulations of turbofan '
                   'jet engines until each one failed, recording <b>21 sensors</b> (temperature, pressure, fan '
                   'speed, etc.) every single flight cycle, from a healthy start to failure.<br><br>'
                   '<b>Four versions of the problem</b>, from easiest to hardest:'
                   '<ul><li><b>FD001</b> -- one flying condition, one type of fault</li>'
                   '<li><b>FD003</b> -- one flying condition, two types of fault</li>'
                   '<li><b>FD002</b> -- six flying conditions, one fault</li>'
                   '<li><b>FD004</b> -- six flying conditions, two faults (the hardest)</li></ul></div>',
                   unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="explain-card"><b>Why "6 flying conditions" makes it harder</b><br>'
                   'An engine cruising at high altitude naturally reads differently on its sensors than one '
                   'climbing near sea level -- even if both are perfectly healthy. Without correcting for this, '
                   'the model would confuse "a different flying condition" with "the engine is failing".<br><br>'
                   '<b>The fix:</b> we grouped flight data into its 6 underlying conditions and compared each '
                   'sensor only against <i>its own condition\u2019s</i> normal range, so a real decline stands out '
                   'from a condition change.</div>', unsafe_allow_html=True)

    st.markdown("## 3. From raw sensors to a prediction (the pipeline)")
    steps = [
        ("Clean & group", "Checked for missing/duplicate data (none found) and grouped readings by their flying condition."),
        ("Normalise", "Compared each sensor to the normal range for its own flying condition, so condition changes don't look like failures."),
        ("Pick useful sensors", "Dropped sensors that never change, or that don't relate to engine age at all -- they add noise, not signal."),
        ("Build a history window", "For each prediction, the model looks at the last 60 flights of sensor data, not just the current one, to see the trend."),
        ("Summarise the window", "Turned those 60 flights into simple numbers: the latest reading, the average, how much it's changed, the trend direction."),
        ("Predict with several models", "Several different models (decision-tree-based, and neural networks that read the raw trend directly) each make a prediction."),
        ("Combine them", "Averaged the models together -- different models make different mistakes, so averaging cancels some of them out."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f'<div class="explain-card"><span class="step-num">{i}</span><b>{title}</b><br>'
                   f'<span style="opacity:0.85">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("## 4. How the model was tested fairly")
    st.markdown('<div class="explain-card">The model is only ever judged on engines it has <b>never seen during '
               'training</b>. We also made sure no single engine\'s data appeared on both sides of that split '
               '(it\'s all-or-nothing, per engine) -- otherwise the test would be too easy and the results '
               'misleadingly good, a mistake that is surprisingly common in this kind of project.</div>',
               unsafe_allow_html=True)

    st.markdown("## 5. What the sensors actually told us")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="explain-card"><b>Which sensors mattered most</b><br>'
                   'Not all 21 sensors are useful. A handful never change at all across an engine\'s life -- those '
                   'were dropped. Of the rest, a small group (sensors tracking fan speed, pressure ratio, and '
                   'temperature) consistently moved together with engine age and dominated every prediction the '
                   'model made, across all four versions of the problem. <b>How many flights the engine has '
                   'already flown</b> turned out to be one of the single most useful pieces of information -- '
                   'which makes intuitive sense: it\'s the flight equivalent of an odometer reading.</div>',
                   unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="explain-card"><b>Checking for bad sensor readings</b><br>'
                   'We checked for unusually extreme sensor values two ways, then looked at whether those '
                   'unusual points actually hurt the model\'s accuracy. On most datasets they didn\'t -- in fact '
                   'accuracy was often <i>better</i> on those points, because extreme sensor readings are what '
                   'degradation near failure actually looks like, not noise to be removed. <b>We deliberately did '
                   'not delete anything</b>: doing so would risk throwing away the exact pattern the model needs '
                   'to learn. One exception, flagged honestly rather than hidden, is noted in the limitations '
                   'below.</div>', unsafe_allow_html=True)

    st.markdown("## 6. What shape does the data have? (and why it matters)")
    st.markdown('<div class="explain-card">Before trusting a model\'s accuracy number, it helps to know what kind '
               'of mistake it tends to make. We checked two things: how long engines tend to last, and how the '
               'model\'s errors are spread out.<br><br>'
               '<b>Engine lifetimes aren\'t evenly spread around an average</b> -- most fail somewhat close to a '
               'typical age, but a few last much longer, creating a lopsided (skewed) pattern rather than a '
               'symmetric bell curve. This matches how engineers normally expect mechanical failures to behave.'
               '<br><br><b>The model\'s prediction errors also aren\'t perfectly symmetric</b> -- on some versions '
               'of the problem the model leans slightly toward one type of mistake over the other, and on all of '
               'them, a few larger-than-usual errors are more common than a perfectly smooth, symmetric spread '
               'would suggest. This is exactly why we didn\'t stop at one accuracy number (see section 8).</div>',
               unsafe_allow_html=True)

    st.markdown("## 7. Which models we tried, and what won")
    st.markdown('<div class="explain-card"><b>Two families of model competed</b>, fairly, on data they\'d never '
               'seen: <b>decision-tree models</b> (which look at summary statistics of the last 60 flights -- the '
               'latest reading, the average, the trend) and <b>neural networks</b> (which read the raw sequence of '
               'the last 60 flights directly, including a version that pays extra attention to the most relevant '
               'flights). We also tried simply averaging several models together.<br><br>'
               '<b>The tree-based models won, consistently</b>, on every version of the problem -- the averaged '
               'combination only helped on the easiest one. This doesn\'t mean neural networks are worse in '
               'general; with only 100-260 training engines, there usually isn\'t enough data for a neural network '
               'to show its usual advantage. We also compared the final model against two honest baselines -- '
               '"always predict the typical engine\'s lifetime" and a simple formula assuming engines wear out at '
               'a constant rate -- and it beat both by a wide margin on every dataset, which is the real test of '
               'whether the sensors are adding genuine information.</div>', unsafe_allow_html=True)

    st.markdown("## 8. How good is it, in plain terms?")
    summ_path = Path("results/summary.csv")
    if summ_path.exists():
        try:
            s = pd.read_csv(summ_path, index_col=0)
            rows_html = "".join(
                f'<tr><td>{d}</td><td>{r.get("test_rmse", float("nan")):.1f}</td>'
                f'<td>{r.get("test_rmse_last30", float("nan")):.1f}</td></tr>'
                for d, r in s.iterrows())
            st.markdown(f'<div class="explain-card">On engines that are <b>close to failing (within about 30 '
                       f'flights)</b> -- the range that actually matters for a maintenance decision -- the model '
                       f'is typically off by only a few flights. Further from failure, where almost nothing '
                       f'visibly changes yet, predictions are naturally less precise; this is expected and shared '
                       f'by every model tested on this problem.<br><br>'
                       f'<table style="width:100%; text-align:left;"><tr><th>Dataset</th>'
                       f'<th>Typical error (flights)</th><th>Error near failure (flights)</th></tr>'
                       f'{rows_html}</table></div>', unsafe_allow_html=True)
        except Exception:
            st.info("Run the dashboard once on real data to see the actual numbers here.")
    else:
        st.markdown('<div class="explain-card">On engines close to failing, the model is typically off by only a '
                   'few flights; further out, predictions are naturally less precise. See the '
                   '<b>\U0001F4CA Dashboard \u2192 Results achieved</b> tab once models are trained.</div>',
                   unsafe_allow_html=True)

    st.markdown("## 9. Going beyond a single number: how confident is the model?")
    st.markdown('<div class="explain-card">A single predicted number hides something important: <i>how sure</i> '
               'the model is. We built a second kind of model that predicts a <b>range</b> instead of one number '
               '(e.g. "between 35 and 60 flights left"), and checked whether that range was honest -- if we say '
               '"80% confident", roughly 80% of the time the true answer really should fall inside it. The first '
               'version wasn\'t; we fixed it with a statistical correction so the stated confidence can actually '
               'be trusted, not just assumed.</div>', unsafe_allow_html=True)

    st.markdown("## 10. A safer, risk-adjusted option")
    cp_path = Path("results/conservative_comparison.csv")
    extra = ""
    if cp_path.exists():
        try:
            comp = pd.read_csv(cp_path)
            extra = (f'<br><br>In testing, this cut the share of "too optimistic" predictions from roughly '
                    f'{comp["main_pct_late"].mean():.0f}% to {comp["conservative_pct_late"].mean():.0f}% on '
                    f'average across all four datasets, at the cost of being somewhat less precise overall -- a '
                    f'deliberate, documented trade-off, not a free upgrade.')
        except Exception:
            pass
    st.markdown(f'<div class="explain-card">Being <i>wrong in the optimistic direction</i> (predicting more life '
               f'than an engine really has) is the riskier mistake for maintenance planning -- it\'s the one that '
               f'can lead to a missed service window. So alongside the main, most-accurate model, we built a '
               f'<b>second, deliberately more cautious model</b> that leans toward under-estimating remaining '
               f'life rather than over-estimating it, sized using how much costlier a late call is assumed to be '
               f'than an early one.{extra}<br><br>Try the <b>\U0001F6E1\ufe0f Use risk-adjusted model</b> toggle on '
               f'the main dashboard to see this in action.</div>', unsafe_allow_html=True)

    st.markdown("## 11. Honest limitations")
    st.markdown('<div class="explain-card"><ul>'
               '<li>This is simulated data, not a real fleet -- real sensors drift, go missing, and get repaired '
               'in ways this clean simulation doesn\'t capture.</li>'
               '<li>On the two hardest datasets, the main model is sometimes slightly <b>too optimistic</b> about '
               'how much life is left for engines in the middle of their life. The risk-adjusted model (section '
               '10) largely fixes this, but introduces its own small trade-off with one particular scoring metric '
               '-- a nuance we investigated rather than glossed over.</li>'
               '<li>The model can\'t predict beyond a set ceiling (it will just say "125+ (or 150+) flights left"), '
               'since very early in an engine\'s life there\'s nothing yet to tell it apart from a brand-new one.</li>'
               '<li>On the hardest dataset (FD004, with two overlapping fault types), one of our data-quality '
               'checks behaved differently than on the other three, and we don\'t yet have a confirmed '
               'explanation -- flagged as an open question rather than papered over.</li>'
               '</ul></div>', unsafe_allow_html=True)
    st.stop()


# ==================================================================================================================
# PAGE 2: the main dashboard
# ==================================================================================================================
if fd not in trained:
    st.warning(f"No trained model for {fd} yet. Train it, then refresh this page:")
    st.code(f"python run_experiments.py --data_dir {DATA_DIR} --fds {fd} --windows 40 60 --caps 125 "
            f"--models hgb xgb lstm cnn_lstm cnn_lstm_attn", language="powershell")
    st.stop()

pred = get_model(fd, use_conservative)
try:
    test, fleet = get_fleet(fd, DATA_DIR, use_conservative)
except FileNotFoundError:
    st.error(f"Could not find the {fd} files in '{DATA_DIR}'.")
    st.stop()
fleet = fleet.copy()
fleet["status"] = np.where(fleet.pred_rul < crit, "CRITICAL", np.where(fleet.pred_rul < warn, "WARNING", "OK"))

if use_conservative:
    st.warning("\U0001F6E1\ufe0f Showing the **risk-adjusted (conservative)** model: predictions are "
              "deliberately biased toward 'less time left' to reduce late predictions. Expect lower accuracy "
              "than the main model -- see the Results achieved tab for the exact trade-off.", icon="\U0001F6E1\ufe0f")

st.markdown(f'<div class="hero"><h1>\u2708\ufe0f Turbofan Engine Health &amp; Remaining Useful Life</h1>'
           f'<span class="pill">{fd}</span><span class="pill">{DESC[fd]}</span>'
           f'<span class="pill">Model: {" + ".join(pred.names)}</span>'
           f'<span class="pill">window={pred.window}</span><span class="pill">RUL cap={pred.cap}</span>'
           f'<span class="pill">{len(pred.pre.cols_)} sensors used</span></div>', unsafe_allow_html=True)

tab_fleet, tab_results, tab_bench = st.tabs(["\U0001F6E9\ufe0f Fleet & engine view", "\U0001F3C6 Results achieved",
                                             "\U0001F4C8 Visual gallery"])

# every plot any of this project's scripts might have produced for the CURRENTLY SELECTED dataset, in one place.
# (fd) is already selected in the sidebar, so this whole gallery updates automatically when the dataset changes.
CLUSTER_NOTE = ("Not applicable to this dataset: it has only 1 operating condition, so there is nothing to "
               "cluster. This plot only exists for FD002/FD004 (6 operating conditions each).")
# each entry: (folder, filename, title, caption, applicable_to_this_dataset) -- "applicable" lets the gallery
# tell "not generated yet" (run the script) apart from "doesn't apply here" (nothing to run, by design)
GALLERY = [
    ("results", f"{fd}_test_predictions.png", "Predicted vs true RUL on the test engines",
     "Scatter of predicted vs true remaining life (left) and both sorted by true RUL (right) -- shows where the "
     "model over- or under-predicts across the full range of engine lives.", True),
    ("phm_results", f"{fd}_feature_importance.png", "Which features drove the predictions",
     "Permutation importance: how much the model's accuracy drops when each feature is shuffled. Higher = more "
     "important.", True),
    ("phm_results", f"{fd}_health_indicator.png", "Composite health indicator (PCA)",
     "A single degradation index built from all sensors with PCA, plotted over several engines' lives -- the "
     "standard PHM alternative to using raw multi-sensor input directly.", True),
    ("distribution_results", f"{fd}_lifetime.png", "Engine lifetime distribution",
     "Histogram of engine lifetimes against the best-fitting statistical distribution (left), and a Q-Q plot "
     "checking that fit (right).", True),
    ("distribution_results", f"{fd}_residuals.png", "Residual distribution of the final model",
     "Histogram of prediction errors against a Normal curve (left), and a Q-Q plot vs Normal (right) -- checks "
     "whether RMSE is really the theoretically ideal loss for this model's errors.", True),
    ("distribution_results", f"{fd}_cluster_k.png", "Validating the operating-condition count",
     "Silhouette score for different numbers of operating-condition clusters -- confirms (or not) the assumed "
     "number of regimes, instead of only trusting documentation.", N_COND[fd] > 1),
    ("eda_results", f"{fd}_eda.png", "Exploratory data analysis",
     "Top-left: a top sensor's trend across several engines. Top-right: engine lifetime histogram. Bottom-left: "
     "sensor correlation with cycle. Bottom-right: any points flagged as possible outliers.", True),
    ("quantile_results", f"{fd}_quantile_study.png", "Uncertainty calibration",
     "Left: calibration curve (target quantile vs how often it's actually exceeded -- the dashed line is perfect "
     "calibration). Right: the predicted 10-90% interval against true RUL on the test engines.", True),
]
GALLERY_APPLICABLE = [g for g in GALLERY if g[4]]
GALLERY_NA = [g for g in GALLERY if not g[4]]


def gallery_items(fd):
    return [(folder, title, caption, p) for folder, fname, title, caption, ok in GALLERY_APPLICABLE
            if (p := Path(folder) / fname).exists()]

# ---- Fleet & engine view -----------------------------------------------------------------------------------------
with tab_fleet:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engines", len(fleet))
    c2.metric("\U0001F534 Critical", int((fleet.status == "CRITICAL").sum()))
    c3.metric("\U0001F7E0 Warning", int((fleet.status == "WARNING").sum()))
    c4.metric("Test RMSE", f"{np.sqrt(np.mean((fleet.pred_rul - fleet.true_rul) ** 2)):.1f} cycles")

    def render_fleet_table(rows):
        rows = rows.round(1).copy()
        rows["status"] = rows["status"].apply(badge)
        html = rows.rename(columns={"unit": "Unit", "cycle": "Cycle", "pred_rul": "Predicted RUL",
                                    "true_rul": "True RUL", "status": "Status"}).to_html(escape=False, index=False)
        st.markdown(f'<div class="fleet-table">{html}</div>', unsafe_allow_html=True)

    sorted_fleet = fleet.sort_values("pred_rul")[["unit", "cycle", "pred_rul", "true_rul", "status"]]
    TOP_N = 15

    # fleet table and engine inspector are now STACKED, each at FULL page width -- no more squeezing a 5-column
    # table or a chart into a narrow side-by-side column.
    st.subheader("Fleet (most urgent first)")
    render_fleet_table(sorted_fleet.head(TOP_N))
    st.download_button("\u2b07\ufe0f Download fleet report (CSV)", fleet.to_csv(index=False), f"{fd}_fleet_report.csv")

    st.divider()
    unit = st.selectbox("Inspect engine", fleet.sort_values("pred_rul").unit.tolist())
    eng = test[test.unit == unit]
    hist = pred.predict_raw(eng, last_only=False)
    last_cycle = hist.cycle.max()
    true_last = float(fleet.loc[fleet.unit == unit, "true_rul"].iloc[0])
    hist["true_rul"] = np.minimum(true_last + (last_cycle - hist.cycle), pred.cap)
    status_now = fleet.loc[fleet.unit == unit, "status"].iloc[0]
    st.markdown(f"#### Engine {unit}: predicted vs true RUL over its life &nbsp; {badge(status_now)}",
               unsafe_allow_html=True)
    # both charts stacked FULL page width, one after another -- maximum room for each, no side-by-side squeeze
    st.line_chart(hist.set_index("cycle")[["pred_rul", "true_rul"]], color=["#ff4b4b", "#1f77b4"], height=420)
    st.caption("True RUL before the last cycle is reconstructed by counting back from the official final RUL value.")

    st.markdown("##### Sensor trends")
    sc1, sc2 = st.columns([3, 1])
    sensors = sc1.multiselect("Sensors", pred.pre.cols_, default=pred.pre.cols_[:2])
    smooth = sc2.toggle("Smooth", value=True, help="5-cycle rolling average, for display only -- the model "
                        "itself uses the raw, unsmoothed values.")
    if sensors:
        sdf = pred.pre.transform(eng).set_index("cycle")[sensors]
        if smooth:
            sdf = sdf.rolling(5, min_periods=1, center=True).mean()
        st.line_chart(sdf, height=420)
        st.caption("Sensors shown after per-operating-condition normalisation (z-scores), so trends are "
                  "comparable across sensors. Raw sensor noise is real -- smoothing is for readability only.")

    if len(sorted_fleet) > TOP_N:
        st.divider()
        with st.expander(f"\U0001F4CB Show all {len(sorted_fleet)} engines (full width)"):
            render_fleet_table(sorted_fleet)  # rendered at FULL page width, not squeezed into the narrow left column

    items = gallery_items(fd)
    if items:
        st.divider()
        st.markdown(f"#### \U0001F4C8 Key visuals for {fd}")
        preview = items[:4]  # a quick-look strip; the full set lives in the Visual gallery tab
        cols = st.columns(len(preview), gap="medium")
        for col, (folder, title, caption, p) in zip(cols, preview):
            with col:
                with st.container(border=True):
                    st.image(str(p), use_container_width=True)
                    st.caption(title)
        if len(items) > len(preview):
            st.caption(f"+ {len(items) - len(preview)} more in the \U0001F4C8 Visual gallery tab \u2192")

# ---- Results achieved (project-wide, across all trained datasets) -------------------------------------------------
with tab_results:
    st.subheader("\U0001F3C6 Model results achieved so far")
    sp = Path("results/summary.csv")
    if sp.exists():
        summ = pd.read_csv(sp, index_col=0)
        cols = st.columns(len(summ))
        for col, (dset, row) in zip(cols, summ.iterrows()):
            with col:
                tag = "\u2713 current" if dset == fd else ""
                st.markdown(f'<div class="explain-card" style="text-align:center;"><b>{dset}</b> {tag}<br>'
                           f'<span style="font-size:1.8rem; font-weight:800;">{row.get("test_rmse", float("nan")):.2f}</span>'
                           f'<br><span style="opacity:0.7; font-size:0.8rem;">official test RMSE (cycles)</span>'
                           f'<br><span style="opacity:0.7; font-size:0.8rem;">NASA score: '
                           f'{row.get("test_score", float("nan")):.0f}</span></div>', unsafe_allow_html=True)
        st.markdown("##### Full numbers")
        show_cols = [c for c in ["best_model", "best_cap", "best_window", "val_rmse", "test_rmse",
                                 "test_rmse_capped", "test_rmse_last30", "test_score"] if c in summ.columns]
        st.dataframe(summ[show_cols].round(2), use_container_width=True)
        st.caption("test_rmse_capped compares predictions against the true RUL capped at 125, so FD002/FD004 "
                  "aren't penalised for engines with more life left than the model can ever predict. "
                  "test_rmse_last30 is the error only for engines within 30 cycles of failure -- the range that "
                  "matters most for a maintenance decision.")
    else:
        st.info("No results/summary.csv yet. Train at least one dataset with run_experiments.py.")

    cp = Path("results/conservative_comparison.csv")
    if cp.exists():
        st.divider()
        st.markdown("##### \U0001F6E1\ufe0f Accuracy-optimal vs risk-adjusted (conservative) model")
        comp = pd.read_csv(cp)
        st.caption(f"Conservative model predicts the tau={comp['tau'].iloc[0]:.2f} quantile of RUL "
                  f"(cost ratio: a late prediction assumed {comp['cost_ratio'].iloc[0]:.0f}x worse than an early "
                  f"one). Trained separately -- it does not replace or change the main deployed models.")
        for _, r in comp.iterrows():
            st.markdown(f"**{r['fd']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{r['conservative_rmse']:.2f}", delta=f"{r['conservative_rmse'] - r['main_rmse']:+.2f}",
                      delta_color="inverse", help="vs main model (higher = worse, expected trade-off)")
            c2.metric("Mean error (bias)", f"{r['conservative_mean_error']:+.2f}",
                      delta=f"{r['conservative_mean_error'] - r['main_mean_error']:+.2f}", delta_color="inverse",
                      help="More negative = more conservative (predicts less remaining life than truth)")
            c3.metric("% predictions late", f"{r['conservative_pct_late']:.1f}%",
                      delta=f"{r['conservative_pct_late'] - r['main_pct_late']:+.1f}%", delta_color="inverse",
                      help="The risky direction -- this is what the conservative model aims to reduce")
            c4.metric("Mid-life bias", f"{r['conservative_mean_error_midlife']:+.2f}",
                      delta=f"{r['conservative_mean_error_midlife'] - r['main_mean_error_midlife']:+.2f}",
                      delta_color="inverse", help="Engines with 60-125 cycles left -- where the main model's "
                                                 "optimism was documented as a known limitation")
        st.caption("Deltas are vs the main model; a drop in '% predictions late' and 'mid-life bias' at the cost "
                  "of a higher RMSE is the expected, deliberate trade-off of this model.")
    elif Path("models").exists() and any(Path("models").glob("*_best.joblib")):
        st.caption("\U0001F6E1\ufe0f No risk-adjusted model trained yet. Run: `python train_conservative_model.py "
                  "--data_dir data --fds FD001 FD002 FD003 FD004`")

    dp = Path("distribution_results/residual_stats.csv")
    if dp.exists():
        st.markdown("##### Residual distribution (is the error symmetric, and is RMSE the right loss?)")
        st.dataframe(pd.read_csv(dp).round(3), use_container_width=True, hide_index=True)

    qp = Path("quantile_results/conformal_calibration.csv")
    if qp.exists():
        st.markdown("##### Uncertainty calibration (conformalized quantile regression)")
        st.dataframe(pd.read_csv(qp).round(2), use_container_width=True, hide_index=True)
        st.caption("picp = fraction of true values actually inside the predicted 10-90% interval (target 0.80).")

# ---- Visual gallery: every plot this project's scripts have produced for the selected dataset ------------------
with tab_bench:
    items = gallery_items(fd)
    st.subheader(f"\U0001F4C8 Visual gallery: {fd}")
    st.caption(f"Model: {' + '.join(pred.names)} | window={pred.window} | RUL cap={pred.cap} | "
              f"{len(pred.pre.cols_)} sensors used")

    if not items:
        st.info(f"No plots found yet for {fd}. Generate them with any of: run_experiments.py, "
               "eda.py, distribution_analysis.py, phm_additions.py, quantile_study.py.")
    else:
        st.progress(len(items) / max(1, len(GALLERY_APPLICABLE)),
                   text=f"{len(items)} / {len(GALLERY_APPLICABLE)} applicable analyses available for {fd}")
        n_cols = 2
        for i in range(0, len(items), n_cols):
            cols = st.columns(n_cols, gap="large")
            for col, (folder, title, caption, p) in zip(cols, items[i:i + n_cols]):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.image(str(p), use_container_width=True)
                        st.caption(caption)

    missing_scripts = [fname.rsplit(".", 1)[0].replace(f"{fd}_", "") for folder, fname, title, caption, ok
                       in GALLERY_APPLICABLE if not (Path(folder) / fname).exists()]
    if missing_scripts:
        st.divider()
        st.markdown("##### Not generated yet for this dataset")
        st.caption("Run the corresponding script to add these: " + ", ".join(missing_scripts))

    if GALLERY_NA:
        st.divider()
        st.markdown("##### Not applicable to this dataset")
        cols = st.columns(len(GALLERY_NA))
        for col, (folder, fname, title, caption, ok) in zip(cols, GALLERY_NA):
            with col:
                st.markdown(f'<div class="gallery-missing"><b>{title}</b><br><br>{CLUSTER_NOTE}</div>',
                           unsafe_allow_html=True)
