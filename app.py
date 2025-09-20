# app.py
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date

st.set_page_config(page_title="Grain Distribution Dashboard", layout="wide")
st.title("🚛 Grain Distribution Dashboard")

# ------------------------
# Helpers
# ------------------------
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

SHEET_ALIASES = {
    "Settings": ["Settings"],
    "LGs": ["LGs"],
    "FPS": ["FPS"],
    "Vehicles": ["Vehicles"],
    "CG_to_LG": ["CG_to_LG", "CG_to_LG_Dispatch"],
    "LG_to_FPS": ["LG_to_FPS", "LG_to_FPS_Dispatch", "LG_to_LPS_Dispatch"],
    "Stock_Levels": ["Stock_Levels"],
}

REQUIRED_COLS = {
    "Settings": {"Parameter", "Value"},
    "LGs": {"LG_ID", "LG_Name"},
    "FPS": {"FPS_ID"},
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
    """
    Load expected sheets from workbook bytes and return a dict of processed DataFrames.
    This function is defensive: it tolerates missing optional columns and computes derived fields.
    """
    bio = BytesIO(xls_bytes)
    xfile = pd.ExcelFile(bio)
    names = set(xfile.sheet_names)

    def read_one(tag: str) -> pd.DataFrame:
        for nm in SHEET_ALIASES[tag]:
            if nm in names:
                df = pd.read_excel(xfile, sheet_name=nm)
                # normalize some old column variants
                if tag in ("CG_to_LG","LG_to_FPS"):
                    if "Dispatch_Day" in df.columns and "Day" not in df.columns:
                        df = df.rename(columns={"Dispatch_Day": "Day"})
                return df
        raise ValueError(f"Workbook is missing sheet for '{tag}'. Tried: {SHEET_ALIASES[tag]}")

    # read sheets (if a required sheet is missing, read_one will raise)
    settings     = read_one("Settings")
    lgs          = read_one("LGs")
    fps          = read_one("FPS")
    # Vehicles might be optional historically, but try to read it — if missing create empty
    try:
        vehicles = read_one("Vehicles")
    except Exception:
        vehicles = pd.DataFrame(columns=["Vehicle_ID", "Capacity_tons", "Mapped_LG_IDs"])

    dispatch_cg  = read_one("CG_to_LG")
    dispatch_lg  = read_one("LG_to_FPS")
    stock_levels = read_one("Stock_Levels")

    # Validate minimal columns for required sheets (throws friendly error)
    for tag, need in REQUIRED_COLS.items():
        # map tag -> df
        df = {
            "Settings": settings,
            "LGs": lgs,
            "FPS": fps,
            "Vehicles": vehicles,
            "CG_to_LG": dispatch_cg,
            "LG_to_FPS": dispatch_lg,
            "Stock_Levels": stock_levels
        }[tag]
        _need_cols(df, need, tag)

    # Numeric coercions and ensure Vehicle_ID is string (we treat each row as a trip)
    # dispatch_cg numeric columns
    for c in ("Day","LG_ID","Quantity_tons","AAY_tons","PHH_tons","APL_tons","NFSA_tons"):
        if c in dispatch_cg.columns:
            dispatch_cg[c] = pd.to_numeric(dispatch_cg[c], errors="coerce")
    if "Vehicle_ID" in dispatch_cg.columns:
        dispatch_cg["Vehicle_ID"] = dispatch_cg["Vehicle_ID"].astype(str).str.strip()

    # dispatch_lg numeric columns
    for c in ("Day","LG_ID","FPS_ID","Quantity_tons","AAY_tons","PHH_tons","APL_tons","NFSA_tons"):
        if c in dispatch_lg.columns:
            dispatch_lg[c] = pd.to_numeric(dispatch_lg[c], errors="coerce")
    if "Vehicle_ID" in dispatch_lg.columns:
        dispatch_lg["Vehicle_ID"] = dispatch_lg["Vehicle_ID"].astype(str).str.strip()

    # stock_levels numeric
    for c in ("Day","Entity_ID","Stock_Level_tons"):
        if c in stock_levels.columns:
            stock_levels[c] = pd.to_numeric(stock_levels[c], errors="coerce")

    # Ensure Date exists (parse if present). We'll use Date slider if any Date is present
    def ensure_date_col(df):
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"]).dt.date
            except Exception:
                # leave as-is if parsing fails
                pass
        else:
            df["Date"] = pd.NaT
    ensure_date_col(dispatch_cg)
    ensure_date_col(dispatch_lg)
    ensure_date_col(stock_levels)

    # Ensure category columns exist (fill 0 if absent)
    for col in ("AAY_tons","PHH_tons","APL_tons","NFSA_tons"):
        if col not in dispatch_lg.columns:
            dispatch_lg[col] = 0.0
        if col not in dispatch_cg.columns:
            dispatch_cg[col] = 0.0

    # Settings-derived values & compute FPS thresholds if missing
    DEFAULT_LT = _get_setting(settings, "Default_Lead_Time_days", 3.0, float)
    AAY_kg = _get_setting(settings, "AAY_kg_per_card", 35.0, float)
    PHH_kg = _get_setting(settings, "PHH_kg_per_beneficiary", 5.0, float)
    APL_kg = _get_setting(settings, "APL_kg_per_card", 0.0, float)

    fps = fps.copy()
    if "Lead_Time_days" not in fps.columns:
        fps["Lead_Time_days"] = DEFAULT_LT
    else:
        fps["Lead_Time_days"] = fps["Lead_Time_days"].fillna(DEFAULT_LT)

    # Ensure count columns exist
    for col in ("AAY_Count","PHH_Beneficiaries","APL_Count"):
        if col not in fps.columns:
            fps[col] = 0.0

    fps["Monthly_from_counts_kg"] = (
        pd.to_numeric(fps["AAY_Count"], errors="coerce").fillna(0.0) * AAY_kg
        + pd.to_numeric(fps["PHH_Beneficiaries"], errors="coerce").fillna(0.0) * PHH_kg
        + pd.to_numeric(fps["APL_Count"], errors="coerce").fillna(0.0) * APL_kg
    )

    # Monthly_Demand_tons: prefer explicit column if filled, else derived from counts
    fps["Monthly_Demand_tons"] = fps.get("Monthly_Demand_tons")
    fps["Monthly_Demand_tons"] = pd.to_numeric(fps["Monthly_Demand_tons"], errors="coerce")
    fps["Monthly_Demand_tons"] = fps["Monthly_Demand_tons"].where(
        fps["Monthly_Demand_tons"].notna(), fps["Monthly_from_counts_kg"]/1000.0
    ).fillna(0.0)

    fps["Daily_Demand_tons"] = fps["Monthly_Demand_tons"]/30.0
    fps["Reorder_Threshold_tons"] = fps.get("Reorder_Threshold_tons")
    if "Reorder_Threshold_tons" not in fps.columns or fps["Reorder_Threshold_tons"].isna().all():
        fps["Reorder_Threshold_tons"] = fps["Daily_Demand_tons"] * fps["Lead_Time_days"]

    # Pre-aggregate day totals (defensive: empty-checks)
    day_totals_cg = (dispatch_cg.groupby("Day", as_index=False)["Quantity_tons"].sum()
                     if not dispatch_cg.empty else pd.DataFrame(columns=["Day","Quantity_tons"]))
    day_totals_lg = (dispatch_lg.groupby("Day", as_index=False)["Quantity_tons"].sum()
                     if not dispatch_lg.empty else pd.DataFrame(columns=["Day","Quantity_tons"]))

    # Veh usage (trips / day)
    veh_usage = (
        dispatch_lg.groupby("Day").size().reset_index(name="Trips_Used")
        if not dispatch_lg.empty else pd.DataFrame(columns=["Day","Trips_Used"])
    )
    VEH_TOTAL = int(_get_setting(settings, "Vehicles_Total", 30, int))
    MAX_TRIPS = int(_get_setting(settings, "Max_Trips_Per_Vehicle_Per_Day", 3, int))
    veh_usage["Max_Trips"] = VEH_TOTAL * MAX_TRIPS

    # LG stock pivot: aggregate duplicates (Day,Entity_ID) before pivot to avoid duplicate-index pivot errors
    lg_src = stock_levels[stock_levels["Entity_Type"]=="LG"].copy()
    if not lg_src.empty:
        lg_src = lg_src[pd.notna(lg_src["Day"])]
        lg_agg = lg_src.groupby(["Day","Entity_ID"], as_index=False)["Stock_Level_tons"].sum()
        lg_stock = lg_agg.pivot(index="Day", columns="Entity_ID", values="Stock_Level_tons").sort_index().ffill()
    else:
        lg_stock = pd.DataFrame()

    # FPS stock: aggregate duplicates & attach reorder threshold
    fps_src = stock_levels[stock_levels["Entity_Type"]=="FPS"].copy()
    if not fps_src.empty:
        fps_src = fps_src[pd.notna(fps_src["Day"])]
        fps_agg = fps_src.groupby(["Day","Entity_ID"], as_index=False)["Stock_Level_tons"].sum()
        # merge reorder thresh if available
        if "FPS_ID" in fps.columns and "Reorder_Threshold_tons" in fps.columns:
            fps_stock = fps_agg.merge(fps[["FPS_ID","Reorder_Threshold_tons"]], left_on="Entity_ID", right_on="FPS_ID", how="left")
            fps_stock["At_Risk"] = fps_stock["Stock_Level_tons"] <= fps_stock["Reorder_Threshold_tons"]
        else:
            fps_stock = fps_agg.copy()
            fps_stock["Reorder_Threshold_tons"] = pd.NA
            fps_stock["At_Risk"] = False
    else:
        fps_stock = pd.DataFrame(columns=["Day","Entity_ID","Stock_Level_tons","FPS_ID","Reorder_Threshold_tons","At_Risk"])

    return {
        "settings": settings, "lgs": lgs, "fps": fps, "vehicles": vehicles,
        "dispatch_cg": dispatch_cg, "dispatch_lg": dispatch_lg,
        "stock_levels": stock_levels, "lg_stock": lg_stock, "fps_stock": fps_stock,
        "day_totals_cg": day_totals_cg, "day_totals_lg": day_totals_lg,
        "veh_usage": veh_usage,
        "params": dict(DAYS=int(_get_setting(settings, "Distribution_Days", 30, int)),
                       TRUCK_CAP=float(_get_setting(settings, "Vehicle_Capacity_tons", 11.5, float)),
                       VEH_TOTAL=VEH_TOTAL, MAX_TRIPS=MAX_TRIPS)
    }

# ------------------------
# Sidebar: upload & history
# ------------------------
with st.sidebar:
    st.header("📤 Load Simulation Output")
    upl = st.file_uploader("Upload Excel (simulation output)", type="xlsx")

    if "runs" not in st.session_state:
        st.session_state.runs = []

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

# ------------------------
# Load active workbook bytes
# ------------------------
active_bytes = None
if upl is not None and not pub:
    active_bytes = upl.read()
elif sel != "(none)":
    idx = int(sel.split(".")[0]) - 1
    active_bytes = st.session_state.runs[idx]["bytes"]

if active_bytes is None:
    st.info("Upload a simulation output Excel or pick a published run from the sidebar.")
    st.stop()

# ------------------------
# Load data
# ------------------------
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
MAX_TRIPS    = D["params"]["MAX_TRIPS"]
VEH_TOTAL    = D["params"]["VEH_TOTAL"]

DAILY_CAP = VEH_TOTAL * MAX_TRIPS * TRUCK_CAP

# ------------------------
# Filters (Date slider preferred)
# ------------------------
with st.sidebar:
    st.header("Filters")
    # Prefer date slider if any Date column contains valid dates
    has_dates = False
    for df in (dispatch_cg, dispatch_lg, stock_levels):
        if "Date" in df.columns and df["Date"].notna().any():
            has_dates = True
            break

    if has_dates:
        # safe min/max
        all_dates = pd.concat([dispatch_cg["Date"].dropna(), dispatch_lg["Date"].dropna()])
        min_dt = all_dates.min() if not all_dates.empty else pd.to_datetime(date.today()).date()
        max_dt = all_dates.max() if not all_dates.empty else pd.to_datetime(date.today()).date()
        date_range = st.slider("Dispatch window (dates)",
                               min_value=pd.to_datetime(min_dt).date(),
                               max_value=pd.to_datetime(max_dt).date(),
                               value=(pd.to_datetime(min_dt).date(), pd.to_datetime(max_dt).date()))
        day_slider_is_date = True
    else:
        # fallback to days
        if not day_totals_cg.empty or not day_totals_lg.empty:
            min_day = int(pd.concat([day_totals_cg["Day"], day_totals_lg["Day"]], ignore_index=True).min())
            max_day = int(pd.concat([day_totals_cg["Day"], day_totals_lg["Day"]], ignore_index=True).max())
        else:
            min_day, max_day = 1, DAYS
        day_range = st.slider("Dispatch window (days)",
                              min_value=min_day, max_value=max_day, value=(min_day, max_day))
        day_slider_is_date = False

    st.subheader("Select LGs")
    try:
        lg_id_to_name = {int(i): str(n) for i, n in zip(pd.to_numeric(lgs["LG_ID"], errors="coerce"), lgs["LG_Name"]) if pd.notna(i)}
    except Exception:
        lg_id_to_name = {}

    cols = st.columns(4)
    selected_lgs = []
    lg_ids_for_check = list(lg_stock.columns) if not lg_stock.empty else list(lgs["LG_ID"].astype(int))
    for i, lg_id in enumerate(lg_ids_for_check):
        label = lg_id_to_name.get(int(lg_id) if pd.notna(lg_id) else lg_id, str(lg_id))
        if cols[i % 4].checkbox(label, value=True, key=f"lg_{lg_id}"):
            selected_lgs.append(lg_id)
    selected_lg_ids = pd.to_numeric(pd.Series(selected_lgs), errors="coerce").dropna().astype(int).tolist()

    st.markdown("---")
    st.header("Quick KPIs")
    if day_slider_is_date:
        cg_sel = dispatch_cg.query("Date >= @date_range[0] and Date <= @date_range[1]")["Quantity_tons"].sum() if not dispatch_cg.empty else 0.0
        lg_sel = dispatch_lg.query("Date >= @date_range[0] and Date <= @date_range[1]")["Quantity_tons"].sum() if not dispatch_lg.empty else 0.0
    else:
        cg_sel = day_totals_cg.query("Day>=@day_range[0] & Day<=@day_range[1]")["Quantity_tons"].sum() if not day_totals_cg.empty else 0.0
        lg_sel = day_totals_lg.query("Day>=@day_range[0] & Day<=@day_range[1]")["Quantity_tons"].sum() if not day_totals_lg.empty else 0.0

    st.metric("CG→LG Total (t)", f"{cg_sel:,.1f}")
    st.metric("LG→FPS Total (t)", f"{lg_sel:,.1f}")
    st.metric("Max Trips/Day", VEH_TOTAL * MAX_TRIPS)
    st.metric("Vehicles Available", VEH_TOTAL)
    st.metric("Truck Capacity (t)", TRUCK_CAP)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "CG→LG Overview", "LG→FPS Overview",
    "CG→LG Report", "FPS Report",
    "FPS At-Risk", "FPS Data",
    "Downloads", "Metrics"
])

# filtering helper
def filter_window(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if day_slider_is_date:
        if "Date" in df.columns:
            return df[(df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])]
        return df
    else:
        if "Day" in df.columns:
            return df[(df["Day"] >= day_range[0]) & (df["Day"] <= day_range[1])]
        return df

# Tab 1: CG→LG Overview
with tab1:
    st.subheader("CG → LG Dispatch")
    base = filter_window(dispatch_cg.copy())
    if not base.empty and selected_lg_ids:
        base = base[base["LG_ID"].isin(selected_lg_ids)]
    xcol = "Date" if day_slider_is_date else "Day"
    df1 = base.groupby(xcol, as_index=False)["Quantity_tons"].sum() if not base.empty else pd.DataFrame(columns=[xcol,"Quantity_tons"])
    fig1 = px.bar(df1, x=xcol, y="Quantity_tons", text="Quantity_tons")
    fig1.update_traces(texttemplate="%{text:.1f}t", textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

# Tab 2: LG→FPS Overview
with tab2:
    st.subheader("LG → FPS Dispatch")
    base = filter_window(dispatch_lg.copy())
    if not base.empty and selected_lg_ids:
        base = base[base["LG_ID"].isin(selected_lg_ids)]
    xcol = "Date" if day_slider_is_date else "Day"
    df2 = base.groupby(xcol, as_index=False)["Quantity_tons"].sum() if not base.empty else pd.DataFrame(columns=[xcol,"Quantity_tons"])
    fig2 = px.bar(df2, x=xcol, y="Quantity_tons", text="Quantity_tons")
    fig2.update_traces(texttemplate="%{text:.1f}t", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

# Tab 3: CG→LG Report
with tab3:
    st.subheader("CG → LG Dispatch Details (with category splits)")
    cg_df = filter_window(dispatch_cg.copy())
    if not cg_df.empty and selected_lg_ids:
        cg_df = cg_df[cg_df["LG_ID"].isin(selected_lg_ids)]
    if not cg_df.empty:
        group_cols = ["LG_ID","Date"] if day_slider_is_date else ["LG_ID","Day"]
        cg_report = (
            cg_df.groupby(group_cols, as_index=False)
                 .agg(Total_Dispatched_tons=("Quantity_tons","sum"),
                      AAY_tons=("AAY_tons","sum"),
                      PHH_tons=("PHH_tons","sum"),
                      APL_tons=("APL_tons","sum"),
                      Trips_Count=("Vehicle_ID","count"))
                 .merge(lgs[["LG_ID","LG_Name"]], on="LG_ID", how="left")
                 .sort_values(group_cols + ["LG_Name","LG_ID"])
        )
    else:
        cg_report = pd.DataFrame()
    st.dataframe(cg_report, use_container_width=True)
    if not cg_report.empty:
        st.download_button("Download CG→LG Report (Excel)", to_excel(cg_report), "CG_to_LG_Report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Tab 4: FPS Report
with tab4:
    st.subheader("FPS-wise Dispatch Details (with category splits)")
    fps_df = filter_window(dispatch_lg.copy())
    if not fps_df.empty and selected_lg_ids:
        fps_df = fps_df[fps_df["LG_ID"].isin(selected_lg_ids)]
    if fps_df.empty:
        report = pd.DataFrame()
    else:
        report = (
            fps_df.groupby("FPS_ID", as_index=False)
                  .agg(Total_Dispatched_tons=("Quantity_tons","sum"),
                       AAY_tons=("AAY_tons","sum"),
                       PHH_tons=("PHH_tons","sum"),
                       APL_tons=("APL_tons","sum"))
        )
        trips = fps_df.groupby("FPS_ID").size().reset_index(name="Trips_Count")
        veh_ids = (fps_df.dropna(subset=["Vehicle_ID"])
                       .assign(Vehicle_ID=fps_df["Vehicle_ID"].astype(str).str.strip())
                       .groupby("FPS_ID")["Vehicle_ID"]
                       .apply(lambda s: ", ".join(sorted(pd.unique(s))))
                       .reset_index(name="Vehicle_IDs"))
        report = report.merge(trips, on="FPS_ID", how="left").merge(veh_ids, on="FPS_ID", how="left")
        if "FPS_Name" in fps.columns:
            report = report.merge(fps[["FPS_ID","FPS_Name"]], on="FPS_ID", how="left")
    st.dataframe(report, use_container_width=True)

# Tab 5: FPS At-Risk
with tab5:
    st.subheader("FPS At-Risk List")
    if not fps_stock.empty:
        # If date slider used, fps_stock is Day-based; we show all at-risk rows and users can inspect Day column
        if day_slider_is_date:
            # filter fps_stock by Day corresponding to Date range is not guaranteed here; show At_Risk rows (best-effort)
            arf = fps_stock[fps_stock["At_Risk"]].copy()
        else:
            arf = fps_stock.query("Day>=@day_range[0] & Day<=@day_range[1] & At_Risk")[["Day","FPS_ID","Stock_Level_tons","Reorder_Threshold_tons"]]
    else:
        arf = pd.DataFrame()
    st.dataframe(arf, use_container_width=True)
    if not arf.empty:
        st.download_button("Download At-Risk (Excel)", to_excel(arf), "fps_at_risk.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Tab 6: FPS Data
with tab6:
    st.subheader("FPS Stock & Upcoming Receipts")
    # end_day selection: if date slider used, try to map Date -> Day from stock_levels (best-effort), else use day_range
    if day_slider_is_date:
        end_dt = date_range[1]
        # attempt mapping
        matches = stock_levels[stock_levels["Date"] == pd.to_datetime(end_dt).date()]
        end_day = int(matches["Day"].max()) if (not matches.empty and "Day" in matches.columns) else None
    else:
        end_day = min(day_range[1], int(stock_levels["Day"].max() if "Day" in stock_levels.columns and not stock_levels.empty else day_range[1]))

    fps_data = []
    if "FPS_ID" in fps.columns:
        for fps_id in fps["FPS_ID"]:
            stock_now = 0.0
            if end_day is not None and not fps_stock.empty:
                s = fps_stock[(fps_stock["FPS_ID"]==fps_id) & (fps_stock["Day"]==end_day)]["Stock_Level_tons"]
                stock_now = float(s.iloc[0]) if not s.empty else 0.0
            future = dispatch_lg[dispatch_lg["FPS_ID"]==fps_id]
            if day_slider_is_date:
                future = future[future["Date"] > date_range[1]]
                next_day = int(future["Day"].min()) if not future.empty else None
            else:
                future = future[future["Day"] > end_day] if end_day is not None else pd.DataFrame()
                next_day = int(future["Day"].min()) if not future.empty else None
            days_to = (next_day - end_day) if (next_day is not None and end_day is not None) else None
            fps_data.append({
                "FPS_ID": fps_id,
                "FPS_Name": fps.set_index("FPS_ID").loc[fps_id,"FPS_Name"] if "FPS_Name" in fps.columns else None,
                "Current_Stock_tons": stock_now,
                "Next_Receipt_Day": next_day,
                "Days_To_Receipt": days_to
            })
    fps_data_df = pd.DataFrame(fps_data)
    st.dataframe(fps_data_df, use_container_width=True)
    if not fps_data_df.empty:
        st.download_button("Download FPS Data (Excel)", to_excel(fps_data_df), "fps_data.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Tab 7: Downloads (export last generated report if present)
with tab7:
    st.subheader("Downloads")
    if 'report' in locals() and isinstance(report, pd.DataFrame) and not report.empty:
        st.download_button("Download FPS Report (Excel)", to_excel(report), "FPS_Report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        pdf_buf = BytesIO()
        with PdfPages(pdf_buf) as pdf:
            fig, ax = plt.subplots(figsize=(8, max(1, len(report)*0.3) + 1))
            ax.axis('off')
            tbl = ax.table(cellText=report.values, colLabels=report.columns, loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            pdf.savefig(fig, bbox_inches='tight')
        st.download_button("Download FPS Report (PDF)", pdf_buf.getvalue(), "FPS_Report.pdf", mime="application/pdf")
    else:
        st.info("No FPS report available to download for selected window.")

# Tab 8: Metrics & Ratios
with tab8:
    st.subheader("Key Performance Indicators & Ratios")

    # window-filtered frames
    if day_slider_is_date:
        cg_win = dispatch_cg[(dispatch_cg["Date"] >= date_range[0]) & (dispatch_cg["Date"] <= date_range[1])] if not dispatch_cg.empty else pd.DataFrame()
        lg_win = dispatch_lg[(dispatch_lg["Date"] >= date_range[0]) & (dispatch_lg["Date"] <= date_range[1])] if not dispatch_lg.empty else pd.DataFrame()
    else:
        cg_win = dispatch_cg[(dispatch_cg["Day"] >= day_range[0]) & (dispatch_cg["Day"] <= day_range[1])] if not dispatch_cg.empty else pd.DataFrame()
        lg_win = dispatch_lg[(dispatch_lg["Day"] >= day_range[0]) & (dispatch_lg["Day"] <= day_range[1])] if not dispatch_lg.empty else pd.DataFrame()

    if selected_lg_ids:
        cg_win = cg_win[cg_win["LG_ID"].isin(selected_lg_ids)]
        lg_win = lg_win[lg_win["LG_ID"].isin(selected_lg_ids)]

    cg_tot = float(cg_win["Quantity_tons"].sum()) if not cg_win.empty else 0.0
    lg_tot = float(lg_win["Quantity_tons"].sum()) if not lg_win.empty else 0.0

    def cat_totals(df):
        if df.empty:
            return dict(AAY=0.0, PHH=0.0, APL=0.0, NFSA=0.0)
        a = float(df["AAY_tons"].sum()) if "AAY_tons" in df.columns else 0.0
        p = float(df["PHH_tons"].sum()) if "PHH_tons" in df.columns else 0.0
        apl = float(df["APL_tons"].sum()) if "APL_tons" in df.columns else 0.0
        nf = a + p
        return dict(AAY=a, PHH=p, APL=apl, NFSA=nf)

    cg_cat = cat_totals(cg_win)
    lg_cat = cat_totals(lg_win)

    def triple_str(aay, phh, apl):
        tot = aay + phh + apl
        if tot == 0:
            return "—"
        return f"AAY {100.0*aay/tot:.1f}%, PHH {100.0*phh/tot:.1f}%, APL {100.0*apl/tot:.1f}%"

    def nfsa_apl_str(nfsa, apl):
        tot = nfsa + apl
        if tot == 0:
            return "—"
        pct = 100.0 * nfsa / tot
        return f"{pct:.1f}% NFSA : {100.0-pct:.1f}% APL  ({nfsa:.1f}t : {apl:.1f}t)"

    cols = st.columns(3)
    cols[0].metric("CG→LG Total (t)", f"{cg_tot:,.1f}")
    cols[1].metric("LG→FPS Total (t)", f"{lg_tot:,.1f}")
    cols[2].metric("Daily Fleet Capacity (t)", f"{DAILY_CAP:,.1f}")

    st.markdown("### Category breakdowns (selected window)")
    st.write("**CG→LG**: ", nfsa_apl_str(cg_cat["NFSA"], cg_cat["APL"]), " | Composition:", triple_str(cg_cat["AAY"], cg_cat["PHH"], cg_cat["APL"]))
    st.write("**LG→FPS**: ", nfsa_apl_str(lg_cat["NFSA"], lg_cat["APL"]), " | Composition:", triple_str(lg_cat["AAY"], lg_cat["PHH"], lg_cat["APL"]))

    cat_df = pd.DataFrame({
        "Channel": ["CG→LG","LG→FPS"],
        "AAY_tons": [cg_cat["AAY"], lg_cat["AAY"]],
        "PHH_tons": [cg_cat["PHH"], lg_cat["PHH"]],
        "APL_tons": [cg_cat["APL"], lg_cat["APL"]],
        "NFSA_tons": [cg_cat["NFSA"], lg_cat["NFSA"]],
    })
    st.dataframe(cat_df, use_container_width=True)

    # additional KPIs: stock at window-end, percent LG cap
    if day_slider_is_date:
        end_date = date_range[1]
        matches = stock_levels[stock_levels["Date"] == pd.to_datetime(end_date).date()]
        end_day_for_calc = int(matches["Day"].max()) if (not matches.empty and "Day" in matches.columns) else None
    else:
        end_day_for_calc = min(day_range[1], int(stock_levels["Day"].max() if "Day" in stock_levels.columns and not stock_levels.empty else day_range[1]))

    if not lg_stock.empty and end_day_for_calc is not None and selected_lg_ids:
        lg_onhand = lg_stock.loc[end_day_for_calc, [c for c in lg_stock.columns if c in selected_lg_ids]].sum()
    else:
        lg_onhand = 0.0

    fps_onhand = fps_stock.query("Day==@end_day_for_calc")["Stock_Level_tons"].sum() if (not fps_stock.empty and end_day_for_calc is not None) else 0.0
    lg_caps = lgs[lgs["LG_ID"].isin(selected_lg_ids)]["Storage_Capacity_tons"].sum() if "Storage_Capacity_tons" in lgs.columns else 0.0
    pct_lg_filled = (lg_onhand/lg_caps)*100 if lg_caps else 0.0
    fps_zero = fps_stock.query("Day==@end_day_for_calc & Stock_Level_tons==0")["FPS_ID"].nunique() if (not fps_stock.empty and end_day_for_calc is not None) else 0
    fps_risk = fps_stock.query("Day==@end_day_for_calc & At_Risk")["FPS_ID"].nunique() if (not fps_stock.empty and end_day_for_calc is not None) else 0

    st.markdown("---")
    st.write(f"LG stock on hand (selected LGs) at window end: {lg_onhand:.1f} t")
    st.write(f"FPS stock on hand at window end: {fps_onhand:.1f} t")
    st.write(f"LG capacity filled (selected LGs): {pct_lg_filled:.1f}%")
    st.write(f"FPS stock-outs at end: {fps_zero}  |  FPS at-risk: {fps_risk}")
