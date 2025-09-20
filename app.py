import time
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date

# ————————————————————————————————
# 1. Page Config
# ————————————————————————————————
st.set_page_config(page_title="Grain Distribution Dashboard", layout="wide")
st.title("🚛 Grain Distribution Dashboard")

# ————————————————————————————————
# Helpers
# ————————————————————————————————
def to_excel(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    (df if df is not None else pd.DataFrame()).to_excel(buf, index=False)
    buf.seek(0)
    return buf.getvalue()

def pack_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for name, df in sheets.items():
            (df if df is not None else pd.DataFrame()).to_excel(w, sheet_name=name, index=False)
    buf.seek(0)
    return buf.getvalue()

def _get_setting(settings: pd.DataFrame, name: str, default=None, cast=float):
    try:
        v = settings.loc[settings["Parameter"] == name, "Value"].iloc[0]
        return cast(v)
    except Exception:
        return cast(default) if default is not None else None

# Try both sheet name variants your projects have used
SHEET_ALIASES = {
    "Settings": ["Settings"],
    "LGs": ["LGs"],
    "FPS": ["FPS"],
    "Vehicles": ["Vehicles"],
    "CG_to_LG": ["CG_to_LG", "CG_to_LG_Dispatch"],
    "LG_to_FPS": ["LG_to_FPS", "LG_to_FPS_Dispatch"],
    "Stock_Levels": ["Stock_Levels"],
}

REQUIRED_COLS = {
    "Settings": {"Parameter", "Value"},
    "LGs": {"LG_ID", "LG_Name"},
    "FPS": {"FPS_ID", "Monthly_Demand_tons", "Max_Capacity_tons"},
    "Vehicles": {"Vehicle_ID"},
    "CG_to_LG": {"Day", "Vehicle_ID", "LG_ID", "Quantity_tons"},
    "LG_to_FPS": {"Day", "Vehicle_ID", "LG_ID", "FPS_ID", "Quantity_tons"},
    "Stock_Levels": {"Day", "Entity_Type", "Entity_ID", "Stock_Level_tons"},
}

def _need_cols(df: pd.DataFrame, needed: set, label: str):
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Sheet '{label}' missing columns: {sorted(miss)}")

@st.cache_data(show_spinner=False)
def load_from_bytes(xls_bytes: bytes):
    bio = BytesIO(xls_bytes)
    xfile = pd.ExcelFile(bio)
    names = set(xfile.sheet_names)

    def read_one(tag: str) -> pd.DataFrame:
        for nm in SHEET_ALIASES[tag]:
            if nm in names:
                df = pd.read_excel(xfile, sheet_name=nm)
                # normalize old column variants
                if tag == "CG_to_LG" and "Dispatch_Day" in df.columns and "Day" not in df.columns:
                    df = df.rename(columns={"Dispatch_Day": "Day"})
                if tag == "LG_to_FPS" and "Dispatch_Day" in df.columns and "Day" not in df.columns:
                    df = df.rename(columns={"Dispatch_Day": "Day"})
                return df
        raise ValueError(f"Workbook is missing sheet for '{tag}'. Tried: {SHEET_ALIASES[tag]}")

    settings     = read_one("Settings")
    lgs          = read_one("LGs")
    fps          = read_one("FPS")
    vehicles     = read_one("Vehicles")
    dispatch_cg  = read_one("CG_to_LG")
    dispatch_lg  = read_one("LG_to_FPS")
    stock_levels = read_one("Stock_Levels")

    # explicit mapping (avoid locals[] pitfalls)
    dfs = {
        "Settings": settings,
        "LGs": lgs,
        "FPS": fps,
        "Vehicles": vehicles,
        "CG_to_LG": dispatch_cg,
        "LG_to_FPS": dispatch_lg,
        "Stock_Levels": stock_levels,
    }

    # validate minimal columns
    for tag, need in REQUIRED_COLS.items():
        _need_cols(dfs[tag], need, tag)

    # ———— FIX 1: keep Vehicle_ID as string; numeric-coerce only numeric fields ————
    for c in ("Day", "LG_ID", "Quantity_tons"):
        if c in dispatch_cg.columns:
            dispatch_cg[c] = pd.to_numeric(dispatch_cg[c], errors="coerce")
    if "Vehicle_ID" in dispatch_cg.columns:
        dispatch_cg["Vehicle_ID"] = dispatch_cg["Vehicle_ID"].astype(str).str.strip()

    for c in ("Day", "LG_ID", "FPS_ID", "Quantity_tons"):
        if c in dispatch_lg.columns:
            dispatch_lg[c] = pd.to_numeric(dispatch_lg[c], errors="coerce")
    if "Vehicle_ID" in dispatch_lg.columns:
        dispatch_lg["Vehicle_ID"] = dispatch_lg["Vehicle_ID"].astype(str).str.strip()

    for c in ("Day", "Entity_ID", "Stock_Level_tons"):
        if c in stock_levels.columns:
            stock_levels[c] = pd.to_numeric(stock_levels[c], errors="coerce")
    # ———— END FIX 1 ————

    # settings params
    DAYS       = _get_setting(settings, "Distribution_Days", 30, int)
    TRUCK_CAP  = _get_setting(settings, "Vehicle_Capacity_tons", 11.5, float)
    VEH_TOTAL  = _get_setting(settings, "Vehicles_Total", 30, int)
    MAX_TRIPS  = _get_setting(settings, "Max_Trips_Per_Vehicle_Per_Day", 3, int)
    DEFAULT_LT = _get_setting(settings, "Default_Lead_Time_days", 3, float)

    # FPS thresholds (compute if missing)
    fps = fps.copy()
    if "Lead_Time_days" not in fps.columns:
        fps["Lead_Time_days"] = DEFAULT_LT
    else:
        fps["Lead_Time_days"] = fps["Lead_Time_days"].fillna(DEFAULT_LT)
    fps["Daily_Demand_tons"] = pd.to_numeric(fps["Monthly_Demand_tons"], errors="coerce")/30.0
    if "Reorder_Threshold_tons" not in fps.columns:
        fps["Reorder_Threshold_tons"] = fps["Daily_Demand_tons"] * fps["Lead_Time_days"]

    # aggregates (align with your original code)
    day_totals_cg = (dispatch_cg.groupby("Day", as_index=False)["Quantity_tons"].sum()
                     if not dispatch_cg.empty else pd.DataFrame(columns=["Day","Quantity_tons"]))
    day_totals_lg = (dispatch_lg.groupby("Day", as_index=False)["Quantity_tons"].sum()
                     if not dispatch_lg.empty else pd.DataFrame(columns=["Day","Quantity_tons"]))

    # ✅ trips/day = number of rows (each row is one trip)
    veh_usage = (
        dispatch_lg.groupby("Day").size().reset_index(name="Trips_Used")
        if not dispatch_lg.empty else pd.DataFrame(columns=["Day","Trips_Used"])
    )
    veh_usage["Max_Trips"] = VEH_TOTAL * MAX_TRIPS  # vehicles * trips/vehicle/day

    # LG stock pivot
    lg_stock = (stock_levels[stock_levels["Entity_Type"]=="LG"]
                .pivot(index="Day", columns="Entity_ID", values="Stock_Level_tons")
                .sort_index().ffill())

    # FPS stock w/ thresholds & risk
    fps_stock = (stock_levels[stock_levels["Entity_Type"]=="FPS"]
                 .merge(fps[["FPS_ID","Reorder_Threshold_tons"]],
                        left_on="Entity_ID", right_on="FPS_ID", how="left"))
    fps_stock["At_Risk"] = fps_stock["Stock_Level_tons"] <= fps_stock["Reorder_Threshold_tons"]

    return {
        "settings": settings, "lgs": lgs, "fps": fps, "vehicles": vehicles,
        "dispatch_cg": dispatch_cg, "dispatch_lg": dispatch_lg,
        "stock_levels": stock_levels, "lg_stock": lg_stock, "fps_stock": fps_stock,
        "day_totals_cg": day_totals_cg, "day_totals_lg": day_totals_lg,
        "veh_usage": veh_usage,
        "params": dict(DAYS=DAYS, TRUCK_CAP=TRUCK_CAP, VEH_TOTAL=VEH_TOTAL, MAX_TRIPS=MAX_TRIPS)
    }


# ————————————————————————————————
# Sidebar: upload & publish to session history
# ————————————————————————————————
with st.sidebar:
    st.header("📤 Load Simulation Output")
    upl = st.file_uploader("Upload Excel (simulation output)", type="xlsx")

    if "runs" not in st.session_state:
        st.session_state.runs = []  # [{name, ts, bytes}]

    name = st.text_input("Run name", value="My Run")
    pub = st.button("📌 Publish to history", disabled=(upl is None), use_container_width=True)
    if pub and upl is not None:
        data = upl.read()
        st.session_state.runs.append({"name": name.strip() or f"Run {len(st.session_state.runs)+1}",
                                      "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "bytes": data})
        st.success(f"Published “{st.session_state.runs[-1]['name']}”")
        st.stop()

    st.markdown("---")
    st.subheader("🕘 Session History")
    if st.session_state.runs:
        choices = [f"{i+1}. {r['name']} ({r['ts']})" for i,r in enumerate(st.session_state.runs)]
        sel = st.selectbox("Choose a run", options=["(none)"]+choices, index=0)
    else:
        sel = "(none)"

# Source of active workbook
active_bytes = None
if upl is not None and not pub:
    active_bytes = upl.read()
elif sel != "(none)":
    idx = int(sel.split(".")[0]) - 1
    active_bytes = st.session_state.runs[idx]["bytes"]

if active_bytes is None:
    st.info("Upload a simulation output Excel or pick a published run from the sidebar.")
    st.stop()

# ————————————————————————————————
# Load data
# ————————————————————————————————
try:
    D = load_from_bytes(active_bytes)
except Exception as e:
    st.error("Could not parse workbook.")
    st.exception(e)
    st.stop()

settings     = D["settings"]
lgs          = D["lgs"]
fps          = D["fps"]
vehicles     = D["vehicles"]
dispatch_cg  = D["dispatch_cg"]
dispatch_lg  = D["dispatch_lg"]
stock_levels = D["stock_levels"]
lg_stock     = D["lg_stock"]
fps_stock    = D["fps_stock"]
day_totals_cg= D["day_totals_cg"]
day_totals_lg= D["day_totals_lg"]
veh_usage    = D["veh_usage"]
DAYS         = D["params"]["DAYS"]
TRUCK_CAP    = D["params"]["TRUCK_CAP"]
MAX_TRIPS    = D["params"]["MAX_TRIPS"]  # per-vehicle/day
VEH_TOTAL    = D["params"]["VEH_TOTAL"]

# ✅ TOTAL daily capacity (trips * vehicles * tons)
DAILY_CAP = VEH_TOTAL * MAX_TRIPS * TRUCK_CAP

# -------------------------
# Convert/ensure Date columns and prepare date-based pivots
# -------------------------
def _ensure_date_col(df: pd.DataFrame, col: str = "Date"):
    if col in df.columns:
        # Convert to datetime.date where possible
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    else:
        # try to build from Day (int) as a fallback (keep as None if cannot)
        if "Day" in df.columns:
            try:
                # assume Day integers correspond to some epoch? we cannot assume mapping here —
                # safer to leave None and let UI handle if Date missing
                df[col] = pd.NaT
            except Exception:
                df[col] = pd.NaT
    return df

dispatch_cg = _ensure_date_col(dispatch_cg.copy(), "Date")
dispatch_lg = _ensure_date_col(dispatch_lg.copy(), "Date")
stock_levels = _ensure_date_col(stock_levels.copy(), "Date")

# Build date-based totals (used for slider bounds & quick KPIs)
date_totals_cg = (dispatch_cg.groupby("Date", as_index=False)["Quantity_tons"].sum()
                  if not dispatch_cg.empty and "Date" in dispatch_cg.columns else pd.DataFrame(columns=["Date","Quantity_tons"]))
date_totals_lg = (dispatch_lg.groupby("Date", as_index=False)["Quantity_tons"].sum()
                  if not dispatch_lg.empty and "Date" in dispatch_lg.columns else pd.DataFrame(columns=["Date","Quantity_tons"]))

# Build lg_stock by Date axis (pivot) for metrics
lg_stock_date = pd.DataFrame()
if not stock_levels.empty and "Date" in stock_levels.columns:
    lg_stock_date = (stock_levels[stock_levels["Entity_Type"] == "LG"]
                     .pivot(index="Date", columns="Entity_ID", values="Stock_Level_tons")
                     .sort_index().ffill())

# Build fps_stock_date (with thresholds & risk)
fps_stock_date = pd.DataFrame()
if not stock_levels.empty and "Date" in stock_levels.columns:
    fps_stock_date = (stock_levels[stock_levels["Entity_Type"] == "FPS"]
                      .merge(fps[["FPS_ID", "Reorder_Threshold_tons"]], left_on="Entity_ID", right_on="FPS_ID", how="left"))
    fps_stock_date["At_Risk"] = fps_stock_date["Stock_Level_tons"] <= fps_stock_date["Reorder_Threshold_tons"]

# Determine slider bounds from data (fallback to today +/- DAYS)
all_dates = []
for df in (date_totals_cg, date_totals_lg, stock_levels):
    if not df.empty and "Date" in df.columns:
        all_dates.extend([d for d in pd.to_datetime(df["Date"], errors="coerce").dropna().dt.date.unique()])

if all_dates:
    min_date = min(all_dates)
    max_date = max(all_dates)
else:
    # fallback to a sensible default: today .. today + DAYS-1
    min_date = date.today()
    max_date = date.today()

# ————————————————————————————————
# 5. Layout & Filters (DATE slider)
# ————————————————————————————————
with st.sidebar:
    st.header("Filters")

    # Date range slider (inclusive)
    date_range = st.slider(
        "Dispatch Window (dates)",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
    start_date, end_date = date_range[0], date_range[1]

    st.subheader("Select LGs")

    # 🔁 Map LG_ID → LG_Name for checkbox labels (keep returning LG_IDs)
    try:
        lg_id_to_name = {int(i): str(n) for i, n in zip(pd.to_numeric(lgs["LG_ID"], errors="coerce"), lgs["LG_Name"]) if pd.notna(i)}
    except Exception:
        lg_id_to_name = {}

    cols = st.columns(4)
    selected_lgs = []
    # iterate over LG IDs from lgs to preserve ordering rather than lg_stock_date.columns which may be missing
    try:
        lg_iter = [int(x) for x in lgs["LG_ID"].dropna().astype(int).unique()]
    except Exception:
        lg_iter = list(lg_stock_date.columns) if not lg_stock_date.empty else []

    for i, lg_id in enumerate(lg_iter):
        label = lg_id_to_name.get(int(lg_id) if pd.notna(lg_id) else lg_id, str(lg_id))
        if cols[i % 4].checkbox(label, value=True, key=f"lg_{lg_id}"):
            selected_lgs.append(lg_id)

    # 👇 normalize selected_lgs once for reuse in tabs
    selected_lg_ids = pd.to_numeric(pd.Series(selected_lgs), errors="coerce").dropna().astype(int).tolist()

    st.markdown("---")
    st.header("Quick KPIs")
    # compute CG/LG totals over selected date range
    if not dispatch_cg.empty and "Date" in dispatch_cg.columns:
        cg_sel = dispatch_cg[(dispatch_cg["Date"] >= start_date) & (dispatch_cg["Date"] <= end_date)]["Quantity_tons"].sum()
    else:
        cg_sel = 0.0
    if not dispatch_lg.empty and "Date" in dispatch_lg.columns:
        lg_sel = dispatch_lg[(dispatch_lg["Date"] >= start_date) & (dispatch_lg["Date"] <= end_date)]["Quantity_tons"].sum()
    else:
        lg_sel = 0.0

    st.metric("CG→LG Total (t)", f"{cg_sel:,.1f}")
    st.metric("LG→FPS Total (t)", f"{lg_sel:,.1f}")
    # show capacity figures that match the utilization math
    st.metric("Max Trips/Day", VEH_TOTAL * MAX_TRIPS)
    st.metric("Vehicles Available", VEH_TOTAL)
    st.metric("Truck Capacity (t)", TRUCK_CAP)

# Create tabs (added a new "CG→LG Report" tab)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "CG→LG Overview", "LG→FPS Overview",
    "CG→LG Report", "FPS Report",
    "FPS At-Risk", "FPS Data",
    "Downloads", "Metrics"
])

# ————————————————————————————————
# 6. CG→LG Overview
# ————————————————————————————————
with tab1:
    st.subheader("CG → LG Dispatch")
    if not dispatch_cg.empty and "Date" in dispatch_cg.columns:
        base = dispatch_cg[(dispatch_cg["Date"] >= start_date) & (dispatch_cg["Date"] <= end_date)].copy()
    else:
        base = pd.DataFrame(columns=dispatch_cg.columns) if dispatch_cg is not None else pd.DataFrame()

    if not base.empty and selected_lg_ids:
        base = base[base["LG_ID"].isin(selected_lg_ids)]
    df1 = base.groupby("Date", as_index=False)["Quantity_tons"].sum() if not base.empty else pd.DataFrame(columns=["Date","Quantity_tons"])
    if not df1.empty:
        fig1 = px.bar(df1, x="Date", y="Quantity_tons", text="Quantity_tons")
        fig1.update_traces(texttemplate="%{text:.1f}t", textposition="outside")
        st.plotly_chart(fig1, use_container_width=True, key="cg_lg_overview")
    else:
        st.info("No CG→LG dispatch in selected date window.")

# ————————————————————————————————
# 7. LG→FPS Overview
# ————————————————————————————————
with tab2:
    st.subheader("LG → FPS Dispatch")
    if not dispatch_lg.empty and "Date" in dispatch_lg.columns:
        base = dispatch_lg[(dispatch_lg["Date"] >= start_date) & (dispatch_lg["Date"] <= end_date)].copy()
    else:
        base = pd.DataFrame(columns=dispatch_lg.columns) if dispatch_lg is not None else pd.DataFrame()

    if not base.empty and selected_lg_ids:
        base = base[base["LG_ID"].isin(selected_lg_ids)]
    df2 = base.groupby("Date", as_index=False)["Quantity_tons"].sum() if not base.empty else pd.DataFrame(columns=["Date","Quantity_tons"])
    if not df2.empty:
        fig2 = px.bar(df2, x="Date", y="Quantity_tons", text="Quantity_tons")
        fig2.update_traces(texttemplate="%{text:.1f}t", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True, key="lg_fps_overview")
    else:
        st.info("No LG→FPS dispatch in selected date window.")

# ————————————————————————————————
# 8. CG→LG Report (NEW)
# ————————————————————————————————
with tab3:
    st.subheader("CG → LG Dispatch Details")
    if not dispatch_cg.empty and "Date" in dispatch_cg.columns:
        cg_df = dispatch_cg[(dispatch_cg["Date"] >= start_date) & (dispatch_cg["Date"] <= end_date)].copy()
    else:
        cg_df = pd.DataFrame(columns=dispatch_cg.columns) if dispatch_cg is not None else pd.DataFrame()

    if not cg_df.empty and selected_lg_ids:
        cg_df = cg_df[cg_df["LG_ID"].isin(selected_lg_ids)]

    # Aggregate by LG & Date; include trip count and LG Name
    if not cg_df.empty:
        cg_report = (
            cg_df.groupby(["LG_ID", "Date"], as_index=False)
                 .agg(Total_Dispatched_tons=("Quantity_tons", "sum"),
                      Trips_Count=("Vehicle_ID", "count"))
                 .merge(lgs[["LG_ID", "LG_Name"]], on="LG_ID", how="left")
                 .sort_values(["Date", "LG_Name", "LG_ID"])
        )
    else:
        cg_report = pd.DataFrame(columns=["LG_ID","Date","Total_Dispatched_tons","Trips_Count","LG_Name"])

    st.dataframe(cg_report, use_container_width=True)

    st.download_button(
        "Download CG→LG Report (Excel)",
        to_excel(cg_report),
        f"CG_to_LG_Report_{start_date}_to_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ————————————————————————————————
# 9. FPS Report
# ————————————————————————————————
with tab4:
    st.subheader("FPS-wise Dispatch Details")
    if not dispatch_lg.empty and "Date" in dispatch_lg.columns:
        fps_df = dispatch_lg[(dispatch_lg["Date"] >= start_date) & (dispatch_lg["Date"] <= end_date)].copy()
    else:
        fps_df = pd.DataFrame(columns=dispatch_lg.columns) if dispatch_lg is not None else pd.DataFrame()

    if not fps_df.empty and selected_lg_ids:
        fps_df = fps_df[fps_df["LG_ID"].isin(selected_lg_ids)]

    if fps_df.empty:
        report = pd.DataFrame(columns=["FPS_ID", "FPS_Name", "Total_Dispatched_tons", "Trips_Count", "Vehicle_IDs"])
    else:
        # Total tons per FPS
        report = (
            fps_df.groupby("FPS_ID", as_index=False)["Quantity_tons"]
                  .sum()
                  .rename(columns={"Quantity_tons": "Total_Dispatched_tons"})
        )

        # Trips per FPS = number of rows (robust even if Vehicle_ID has NA)
        trips = fps_df.groupby("FPS_ID").size().reset_index(name="Trips_Count")

        # Vehicle IDs per FPS = unique string IDs, drop NA, sorted
        veh_ids = (
            fps_df.dropna(subset=["Vehicle_ID"])
                  .assign(Vehicle_ID=fps_df["Vehicle_ID"].astype(str).str.strip())
                  .groupby("FPS_ID")["Vehicle_ID"]
                  .apply(lambda s: ", ".join(sorted(pd.unique(s))))
                  .reset_index(name="Vehicle_IDs")
        )

        # Merge parts + FPS name
        report = (report
                  .merge(trips, on="FPS_ID", how="left")
                  .merge(veh_ids, on="FPS_ID", how="left"))

        if "FPS_Name" in fps.columns:
            report = report.merge(fps[["FPS_ID", "FPS_Name"]], on="FPS_ID", how="left")
        else:
            report["FPS_Name"] = ""

        report["Trips_Count"] = report["Trips_Count"].fillna(0).astype(int)
        report["Vehicle_IDs"] = report["Vehicle_IDs"].fillna("")
        report = report[["FPS_ID", "FPS_Name", "Total_Dispatched_tons", "Trips_Count", "Vehicle_IDs"]]
        report = report.sort_values("Total_Dispatched_tons", ascending=False)

    st.dataframe(report, use_container_width=True)

# ————————————————————————————————
# 10. FPS At-Risk
# ————————————————————————————————
with tab5:
    st.subheader("FPS At-Risk List")
    if not fps_stock_date.empty:
        arf = fps_stock_date[(fps_stock_date["Date"] >= start_date) & (fps_stock_date["Date"] <= end_date) & (fps_stock_date["At_Risk"])]
        arf = arf[["Date","FPS_ID","Stock_Level_tons","Reorder_Threshold_tons"]]
    else:
        arf = pd.DataFrame(columns=["Date","FPS_ID","Stock_Level_tons","Reorder_Threshold_tons"])
    st.dataframe(arf, use_container_width=True)
    st.download_button("Download At-Risk (Excel)", to_excel(arf), "fps_at_risk.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ————————————————————————————————
# 11. FPS Data
# ————————————————————————————————
with tab6:
    st.subheader("FPS Stock & Upcoming Receipts")
    # safe end_date fallback using stock_levels date range
    if not stock_levels.empty and "Date" in stock_levels.columns:
        max_stock_date = stock_levels["Date"].dropna().max()
        actual_end_date = min(end_date, max_stock_date) if pd.notna(max_stock_date) else end_date
    else:
        actual_end_date = end_date

    fps_data = []
    fps_ids_iter = list(fps["FPS_ID"]) if "FPS_ID" in fps.columns else []
    for fps_id in fps_ids_iter:
        s = stock_levels[(stock_levels["Entity_Type"]=="FPS") & (stock_levels["Entity_ID"]==fps_id) & (stock_levels["Date"]==actual_end_date)]["Stock_Level_tons"] if "Date" in stock_levels.columns else pd.Series(dtype=float)
        stock_now = float(s.iloc[0]) if not s.empty else 0.0
        future = dispatch_lg[(dispatch_lg["FPS_ID"]==fps_id) & (dispatch_lg["Date"]> actual_end_date)]["Date"] if "Date" in dispatch_lg.columns else pd.Series(dtype='object')
        next_day = min(future) if not future.empty else None
        days_to = (next_day - actual_end_date).days if next_day else None
        fps_data.append({
            "FPS_ID": fps_id,
            "FPS_Name": fps.set_index("FPS_ID").loc[fps_id,"FPS_Name"] if "FPS_Name" in fps.columns else None,
            "Current_Stock_tons": stock_now,
            "Next_Receipt_Date": next_day,
            "Days_To_Receipt": days_to
        })
    fps_data_df = pd.DataFrame(fps_data)
    st.dataframe(fps_data_df, use_container_width=True)
    st.download_button("Download FPS Data (Excel)", to_excel(fps_data_df), "fps_data.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ————————————————————————————————
# 12. Downloads
# ————————————————————————————————
with tab7:
    st.subheader("Download FPS Report")
    st.download_button("Excel", to_excel(report), f"FPS_Report_{start_date}_to_{end_date}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ✅ Only build the PDF if there are rows to avoid IndexError from empty table
    if isinstance(report, pd.DataFrame) and not report.empty:
        pdf_buf = BytesIO()
        with PdfPages(pdf_buf) as pdf:
            fig, ax = plt.subplots(figsize=(8, max(1, len(report)*0.3) + 1))
            ax.axis('off')
            tbl = ax.table(cellText=report.values, colLabels=report.columns, loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            pdf.savefig(fig, bbox_inches='tight')
        st.download_button("PDF", pdf_buf.getvalue(),
                           f"FPS_Report_{start_date}_to_{end_date}.pdf",
                           mime="application/pdf")
    else:
        st.info("No rows in the selected window to export as PDF.")

# ————————————————————————————————
# 13. Metrics
# ————————————————————————————————
with tab8:
    st.subheader("Key Performance Indicators")
    # ensure an actual end_date that exists in stock_levels if possible
    if not stock_levels.empty and "Date" in stock_levels.columns:
        max_stock_date = stock_levels["Date"].dropna().max()
        end_date_for_metrics = min(end_date, max_stock_date) if pd.notna(max_stock_date) else end_date
    else:
        end_date_for_metrics = end_date

    # selected days (inclusive)
    sel_days = (end_date - start_date).days + 1 if end_date >= start_date else 0
    if not dispatch_cg.empty and "Date" in dispatch_cg.columns:
        cg_sel   = dispatch_cg[(dispatch_cg["Date"] >= start_date) & (dispatch_cg["Date"] <= end_date)]["Quantity_tons"].sum()
    else:
        cg_sel = 0.0
    if not dispatch_lg.empty and "Date" in dispatch_lg.columns:
        lg_sel   = dispatch_lg[(dispatch_lg["Date"] >= start_date) & (dispatch_lg["Date"] <= end_date)]["Quantity_tons"].sum()
    else:
        lg_sel = 0.0

    avg_daily_cg = cg_sel/sel_days if sel_days>0 else 0
    avg_daily_lg = lg_sel/sel_days if sel_days>0 else 0

    # average trips/day over window (already trips, not unique vehicles)
    avg_trips = 0.0
    if not D["veh_usage"].empty:
        # Map veh_usage Day -> Date if possible using dispatch_lg Day->Date mapping (best-effort)
        if "Date" in dispatch_lg.columns and "Day" in dispatch_lg.columns:
            mapping = dispatch_lg.drop_duplicates("Day").set_index("Day")["Date"].to_dict()
            veh_usage_dates = D["veh_usage"].copy()
            veh_usage_dates["Date"] = veh_usage_dates["Day"].map(mapping)
            window = veh_usage_dates[(veh_usage_dates["Date"] >= start_date) & (veh_usage_dates["Date"] <= end_date)]["Trips_Used"]
            avg_trips = float(window.mean()) if not window.empty else 0.0
        else:
            # fallback: use Days numeric if available
            window = D["veh_usage"].query("Day>=@0 & Day<=@1".format(1, DAYS))["Trips_Used"]
            avg_trips = float(window.mean()) if not window.empty else 0.0

    if not lg_stock_date.empty and end_date_for_metrics in lg_stock_date.index and selected_lgs:
        # selected_lgs are LG IDs (ints)
        cols_to_sum = [c for c in lg_stock_date.columns if int(c) in selected_lg_ids]
        lg_onhand = lg_stock_date.loc[end_date_for_metrics, cols_to_sum].sum() if cols_to_sum else 0.0
    else:
        lg_onhand = 0.0

    fps_onhand   = fps_stock_date[(fps_stock_date["Date"]==end_date_for_metrics)]["Stock_Level_tons"].sum() if not fps_stock_date.empty else 0.0
    if "Storage_Capacity_tons" in lgs.columns:
        lg_caps = lgs[lgs["LG_ID"].isin(selected_lg_ids)]["Storage_Capacity_tons"].sum()
    else:
        lg_caps = 0.0
    pct_lg_filled= (lg_onhand/lg_caps)*100 if lg_caps else 0.0
    fps_zero     = fps_stock_date[(fps_stock_date["Date"]==end_date_for_metrics) & (fps_stock_date["Stock_Level_tons"]==0)]["FPS_ID"].nunique() if not fps_stock_date.empty else 0
    fps_risk     = fps_stock_date[(fps_stock_date["Date"]==end_date_for_metrics) & (fps_stock_date["At_Risk"])]["FPS_ID"].nunique() if not fps_stock_date.empty else 0
    dispatched_cum = dispatch_lg[dispatch_lg["Date"] <= end_date_for_metrics]["Quantity_tons"].sum() if not dispatch_lg.empty and "Date" in dispatch_lg.columns else 0.0
    total_plan   = dispatch_lg["Quantity_tons"].sum() if not dispatch_lg.empty else 0.0
    pct_plan     = (dispatched_cum/total_plan)*100 if total_plan else 0.0
    remaining_t  = total_plan - dispatched_cum
    days_rem     = math.ceil(remaining_t/DAILY_CAP) if DAILY_CAP else None

    def c(v):
        try:
            return int(math.ceil(float(v)))
        except Exception:
            return 0

    c_cg_sel        = c(cg_sel)
    c_lg_sel        = c(lg_sel)
    c_avg_daily_cg  = c(avg_daily_cg)
    c_avg_daily_lg  = c(avg_daily_lg)
    c_avg_trips     = c(avg_trips)
    c_lg_onhand     = c(lg_onhand)
    c_fps_onhand    = c(fps_onhand)
    c_pct_lg_filled = c(pct_lg_filled)
    c_pct_plan      = c(pct_plan)
    c_fps_zero      = c(fps_zero)
    c_fps_risk      = c(fps_risk)
    c_days_rem      = (None if days_rem is None else c(days_rem))
    # ---------------------------------------------

    metrics = [
        ("Total CG→LG (t)",       f"{c_cg_sel:,d}"),
        ("Total LG→FPS (t)",      f"{c_lg_sel:,d}"),
        ("Avg Daily CG→LG (t/d)", f"{c_avg_daily_cg:,d}"),
        ("Avg Daily LG→FPS (t/d)",f"{c_avg_daily_lg:,d}"),
        ("Avg Trips/Day",         f"{c_avg_trips:,d}"),
        ("LG Stock on Hand (t)",  f"{c_lg_onhand:,d}"),
        ("FPS Stock on Hand (t)", f"{c_fps_onhand:,d}"),
        ("% LG Cap Filled",       f"{c_pct_lg_filled}%"),
        ("FPS Stock-Outs",        f"{c_fps_zero}"),
        ("FPS At-Risk Count",     f"{c_fps_risk}"),
        ("% Plan Completed",      f"{c_pct_plan}%"),
        ("Days Remaining",        f"{c_days_rem if c_days_rem is not None else '—'}")
    ]
    cols = st.columns(3)
    for i, (label, val) in enumerate(metrics):
        cols[i%3].metric(label, val)
