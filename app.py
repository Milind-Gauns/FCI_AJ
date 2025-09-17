# app.py
import time
from io import BytesIO
import streamlit as st
import pandas as pd
import plotly.express as px
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title="Grain Distribution Dashboard", layout="wide")
st.title("🚛 Grain Distribution Dashboard")

# ---------------------------
# Helpers
# ---------------------------
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

# sheet aliases & required cols
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

    dfs = {
        "Settings": settings,
        "LGs": lgs,
        "FPS": fps,
        "Vehicles": vehicles,
        "CG_to_LG": dispatch_cg,
        "LG_to_FPS": dispatch_lg,
        "Stock_Levels": stock_levels,
    }

    # minimal columns check (relaxed: FPS may not have Monthly_Demand_tons if counts were used)
    for tag, need in REQUIRED_COLS.items():
        # allow FPS to skip Monthly_Demand_tons when using counts
        if tag == "FPS":
            need_check = set(need)
            need_check.discard("Monthly_Demand_tons")
        else:
            need_check = need
        _need_cols(dfs[tag], need_check, tag)

    # coerce numeric where appropriate and normalize Vehicle_ID as string
    for c in ("Day","LG_ID","Quantity_tons"):
        if c in dispatch_cg.columns:
            dispatch_cg[c] = pd.to_numeric(dispatch_cg[c], errors="coerce")
    if "Vehicle_ID" in dispatch_cg.columns:
        dispatch_cg["Vehicle_ID"] = dispatch_cg["Vehicle_ID"].astype(str).str.strip()

    for c in ("Day","LG_ID","FPS_ID","Quantity_tons"):
        if c in dispatch_lg.columns:
            dispatch_lg[c] = pd.to_numeric(dispatch_lg[c], errors="coerce")
    if "Vehicle_ID" in dispatch_lg.columns:
        dispatch_lg["Vehicle_ID"] = dispatch_lg["Vehicle_ID"].astype(str).str.strip()

    for c in ("Day","Entity_ID","Stock_Level_tons"):
        if c in stock_levels.columns:
            stock_levels[c] = pd.to_numeric(stock_levels[c], errors="coerce")

    # ensure Date columns are parsed to datetime if present
    for df in (dispatch_lg, dispatch_cg, stock_levels):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # settings params
    DAYS       = _get_setting(settings, "Distribution_Days", 30, int)
    TRUCK_CAP  = _get_setting(settings, "Vehicle_Capacity_tons", 11.5, float)
    VEH_TOTAL  = _get_setting(settings, "Vehicles_Total", 30, int)
    MAX_TRIPS  = _get_setting(settings, "Max_Trips_Per_Vehicle_Per_Day", 3, int)
    DEFAULT_LT = _get_setting(settings, "Default_Lead_Time_days", 3, float)

    # fps thresholds computed if missing
    fps = fps.copy()
    if "Lead_Time_days" not in fps.columns:
        fps["Lead_Time_days"] = DEFAULT_LT
    else:
        fps["Lead_Time_days"] = fps["Lead_Time_days"].fillna(DEFAULT_LT)

    if "Monthly_Demand_tons" in fps.columns:
        fps["Daily_Demand_tons"] = pd.to_numeric(fps["Monthly_Demand_tons"], errors="coerce")/30.0
    else:
        # if Monthly_Demand_tons absent, dashboard will rely on any Daily_* columns present
        fps["Daily_Demand_tons"] = fps.get("Daily_Demand_tons", 0.0)

    if "Reorder_Threshold_tons" not in fps.columns:
        fps["Reorder_Threshold_tons"] = fps["Daily_Demand_tons"] * fps["Lead_Time_days"]

    day_totals_cg = (dispatch_cg.groupby("Day", as_index=False)["Quantity_tons"].sum() if not dispatch_cg.empty else pd.DataFrame(columns=["Day","Quantity_tons"]))
    day_totals_lg = (dispatch_lg.groupby("Day", as_index=False)["Quantity_tons"].sum() if not dispatch_lg.empty else pd.DataFrame(columns=["Day","Quantity_tons"]))

    veh_usage = (dispatch_lg.groupby("Day").size().reset_index(name="Trips_Used") if not dispatch_lg.empty else pd.DataFrame(columns=["Day","Trips_Used"]))
    veh_usage["Max_Trips"] = VEH_TOTAL * MAX_TRIPS

    lg_stock = (stock_levels[stock_levels["Entity_Type"]=="LG"]
                .pivot(index="Day", columns="Entity_ID", values="Stock_Level_tons")
                .sort_index().ffill()) if not stock_levels.empty else pd.DataFrame()

    fps_stock = (stock_levels[stock_levels["Entity_Type"]=="FPS"]
                 .merge(fps[["FPS_ID","Reorder_Threshold_tons"]], left_on="Entity_ID", right_on="FPS_ID", how="left")) if not stock_levels.empty else pd.DataFrame()
    if not fps_stock.empty:
        fps_stock["At_Risk"] = fps_stock["Stock_Level_tons"] <= fps_stock["Reorder_Threshold_tons"]

    return {
        "settings": settings, "lgs": lgs, "fps": fps, "vehicles": vehicles,
        "dispatch_cg": dispatch_cg, "dispatch_lg": dispatch_lg,
        "stock_levels": stock_levels, "lg_stock": lg_stock, "fps_stock": fps_stock,
        "day_totals_cg": day_totals_cg, "day_totals_lg": day_totals_lg,
        "veh_usage": veh_usage,
        "params": dict(DAYS=DAYS, TRUCK_CAP=TRUCK_CAP, VEH_TOTAL=VEH_TOTAL, MAX_TRIPS=MAX_TRIPS)
    }

# ---------------------------
# Sidebar: upload & history
# ---------------------------
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

# source bytes
active_bytes = None
if upl is not None and not pub:
    active_bytes = upl.read()
elif sel != "(none)":
    idx = int(sel.split(".")[0]) - 1
    active_bytes = st.session_state.runs[idx]["bytes"]

if active_bytes is None:
    st.info("Upload a simulation output Excel or pick a published run from the sidebar.")
    st.stop()

# ---------------------------
# Load data
# ---------------------------
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

# ---------------------------
# Filters in sidebar
# ---------------------------
with st.sidebar:
    st.header("Filters")

    # Determine slider range. If Date exists in dispatch tables, use dates, else fall back to Day integers.
    if "Date" in dispatch_lg.columns or "Date" in dispatch_cg.columns:
        # gather dates from available frames
        all_dates = pd.Series(dtype="datetime64[ns]")
        if "Date" in dispatch_lg.columns:
            all_dates = all_dates.append(dispatch_lg["Date"].dropna(), ignore_index=True)
        if "Date" in dispatch_cg.columns:
            all_dates = all_dates.append(dispatch_cg["Date"].dropna(), ignore_index=True)
        if not all_dates.empty:
            min_date = all_dates.min().date()
            max_date = all_dates.max().date()
        else:
            min_date = pd.Timestamp.today().date()
            max_date = pd.Timestamp.today().date()
        date_range = st.slider("Dispatch window (dates, Sundays excluded by sim)", value=(min_date, max_date), min_value=min_date, max_value=max_date, format="YYYY-MM-DD")
        use_dates = True
    else:
        min_day = int(pd.concat([day_totals_cg["Day"], day_totals_lg["Day"]], ignore_index=True).min()) if not day_totals_cg.empty or not day_totals_lg.empty else 1
        max_day = int(pd.concat([day_totals_cg["Day"], day_totals_lg["Day"]], ignore_index=True).max()) if not day_totals_cg.empty or not day_totals_lg.empty else DAYS
        day_range = st.slider("Dispatch Window (days)", min_value=min_day, max_value=max_day, value=(min_day, max_day), format="%d")
        use_dates = False

    st.subheader("Select LGs")
    try:
        lg_id_to_name = {int(i): str(n) for i, n in zip(pd.to_numeric(lgs["LG_ID"], errors="coerce"), lgs["LG_Name"]) if pd.notna(i)}
    except Exception:
        lg_id_to_name = {}

    cols = st.columns(4)
    selected_lgs = []
    # use lg_stock columns if present else use LGs listing
    lg_iter = lg_stock.columns if not lg_stock.empty else lgs["LG_ID"].tolist()
    for i, lg_id in enumerate(lg_iter):
        label = lg_id_to_name.get(int(lg_id) if pd.notna(lg_id) else lg_id, str(lg_id))
        if cols[i % 4].checkbox(label, value=True, key=f"lg_{lg_id}"):
            selected_lgs.append(int(lg_id))

    selected_lg_ids = selected_lgs  # list of ints

    st.markdown("---")
    st.header("Quick KPIs")
    if use_dates:
        # filter day_totals by matching date mapping if available
        if "Date" in day_totals_lg.columns:
            df_cg_tot = day_totals_cg[(pd.to_datetime(day_totals_cg.get("Date", pd.NaT)).dt.date >= date_range[0]) & (pd.to_datetime(day_totals_cg.get("Date", pd.NaT)).dt.date <= date_range[1])] if not day_totals_cg.empty else pd.DataFrame(columns=["Day","Quantity_tons"])
            df_lg_tot = day_totals_lg[(pd.to_datetime(day_totals_lg.get("Date", pd.NaT)).dt.date >= date_range[0]) & (pd.to_datetime(day_totals_lg.get("Date", pd.NaT)).dt.date <= date_range[1])] if not day_totals_lg.empty else pd.DataFrame(columns=["Day","Quantity_tons"])
        else:
            # if day_totals don't have Date, fallback to dispatch frames
            df_cg_tot = dispatch_cg[(dispatch_cg["Date"].dt.date >= date_range[0]) & (dispatch_cg["Date"].dt.date <= date_range[1])] if not dispatch_cg.empty else pd.DataFrame(columns=["Quantity_tons"])
            df_lg_tot = dispatch_lg[(dispatch_lg["Date"].dt.date >= date_range[0]) & (dispatch_lg["Date"].dt.date <= date_range[1])] if not dispatch_lg.empty else pd.DataFrame(columns=["Quantity_tons"])
        cg_sel = df_cg_tot["Quantity_tons"].sum() if not df_cg_tot.empty else 0.0
        lg_sel = df_lg_tot["Quantity_tons"].sum() if not df_lg_tot.empty else 0.0
    else:
        cg_sel = day_totals_cg.query("Day>=@day_range[0] & Day<=@day_range[1]")["Quantity_tons"].sum() if not day_totals_cg.empty else 0.0
        lg_sel = day_totals_lg.query("Day>=@day_range[0] & Day<=@day_range[1]")["Quantity_tons"].sum() if not day_totals_lg.empty else 0.0

    st.metric("CG→LG Total (t)", f"{cg_sel:,.1f}")
    st.metric("LG→FPS Total (t)", f"{lg_sel:,.1f}")
    st.metric("Max Trips/Day", VEH_TOTAL * MAX_TRIPS)
    st.metric("Vehicles Available", VEH_TOTAL)
    st.metric("Truck Capacity (t)", TRUCK_CAP)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "CG→LG Overview", "LG→FPS Overview",
    "CG→LG Report", "FPS Report",
    "FPS At-Risk", "FPS Data",
    "Downloads", "Metrics"
])

# Utility to apply date/day filter
def apply_window(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    if use_dates and "Date" in df.columns:
        df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]
    else:
        df = df.query("Day>=@day_range[0] & Day<=@day_range[1]") if not df.empty else df
    return df

# ---------------------------
# CG->LG Overview
# ---------------------------
with tab1:
    st.subheader("CG → LG Dispatch")
    base = apply_window(dispatch_cg)
    if base is None or base.empty:
        df1 = pd.DataFrame(columns=(["Day","Date","Quantity_tons"] if "Date" in dispatch_cg.columns else ["Day","Quantity_tons"]))
    else:
        if selected_lg_ids:
            base = base[base["LG_ID"].isin(selected_lg_ids)]
        if "Date" in base.columns:
            df1 = base.groupby("Date", as_index=False)["Quantity_tons"].sum()
            fig1 = px.bar(df1, x="Date", y="Quantity_tons", text="Quantity_tons")
        else:
            df1 = base.groupby("Day", as_index=False)["Quantity_tons"].sum()
            fig1 = px.bar(df1, x="Day", y="Quantity_tons", text="Quantity_tons")
        fig1.update_traces(texttemplate="%{text:.1f}t", textposition="outside")
        st.plotly_chart(fig1, use_container_width=True, key="fig_cg_overview")

# ---------------------------
# LG->FPS Overview
# ---------------------------
with tab2:
    st.subheader("LG → FPS Dispatch")
    base = apply_window(dispatch_lg)
    if base is None or base.empty:
        df2 = pd.DataFrame(columns=(["Day","Date","Quantity_tons"] if "Date" in dispatch_lg.columns else ["Day","Quantity_tons"]))
    else:
        if selected_lg_ids:
            base = base[base["LG_ID"].isin(selected_lg_ids)]
        if "Date" in base.columns:
            df2 = base.groupby("Date", as_index=False)["Quantity_tons"].sum()
            fig2 = px.bar(df2, x="Date", y="Quantity_tons", text="Quantity_tons")
        else:
            df2 = base.groupby("Day", as_index=False)["Quantity_tons"].sum()
            fig2 = px.bar(df2, x="Day", y="Quantity_tons", text="Quantity_tons")
        fig2.update_traces(texttemplate="%{text:.1f}t", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True, key="fig_lg_overview")

# ---------------------------
# CG->LG Report
# ---------------------------
with tab3:
    st.subheader("CG → LG Dispatch Details")
    cg_df = apply_window(dispatch_cg)
    if cg_df is None or cg_df.empty:
        cg_df = pd.DataFrame(columns=["LG_ID","Day","Date","Total_Dispatched_tons","Trips_Count","LG_Name"])
    else:
        if selected_lg_ids:
            cg_df = cg_df[cg_df["LG_ID"].isin(selected_lg_ids)]
        # aggregate by LG & Day/Date
        agg_cols = ["AAY_tons","PHH_tons","APL_tons","NSFA_tons"] if "AAY_tons" in cg_df.columns else []
        cg_report = (
            cg_df.groupby(["LG_ID", "Day", "Date"], as_index=False)
                 .agg(Total_Dispatched_tons=("Quantity_tons", "sum"),
                      Trips_Count=("Vehicle_ID", "count"),
                      **({c: (c, "sum") for c in agg_cols} if agg_cols else {}))
                 .merge(lgs[["LG_ID", "LG_Name"]], on="LG_ID", how="left")
                 .sort_values(["Day", "LG_Name", "LG_ID"])
        )
        cg_df = cg_report
    st.dataframe(cg_df, use_container_width=True)

    st.download_button(
        "Download CG→LG Report (Excel)",
        to_excel(cg_df),
        f"CG_to_LG_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ---------------------------
# FPS Report
# ---------------------------
with tab4:
    st.subheader("FPS-wise Dispatch Details")
    fps_df = apply_window(dispatch_lg)
    if fps_df is None or fps_df.empty:
        report = pd.DataFrame(columns=["FPS_ID", "FPS_Name", "Total_Dispatched_tons", "Trips_Count", "Vehicle_IDs"])
    else:
        if selected_lg_ids:
            fps_df = fps_df[fps_df["LG_ID"].isin(selected_lg_ids)]
        # total tons per FPS
        report = fps_df.groupby("FPS_ID", as_index=False)["Quantity_tons"].sum().rename(columns={"Quantity_tons":"Total_Dispatched_tons"})
        trips = fps_df.groupby("FPS_ID").size().reset_index(name="Trips_Count")
        veh_ids = (fps_df.dropna(subset=["Vehicle_ID"])
                   .assign(Vehicle_ID=fps_df["Vehicle_ID"].astype(str).str.strip())
                   .groupby("FPS_ID")["Vehicle_ID"].apply(lambda s: ", ".join(sorted(pd.unique(s)))).reset_index(name="Vehicle_IDs"))
        report = report.merge(trips, on="FPS_ID", how="left").merge(veh_ids, on="FPS_ID", how="left")
        if "FPS_Name" in fps.columns:
            report = report.merge(fps[["FPS_ID","FPS_Name"]], on="FPS_ID", how="left")
        else:
            report["FPS_Name"] = ""
        # if category columns exist in fps_df, present aggregates and ratios
        if "AAY_tons" in fps_df.columns:
            cats = fps_df.groupby("FPS_ID")[["AAY_tons","PHH_tons","APL_tons","NSFA_tons"]].sum().reset_index()
            report = report.merge(cats, on="FPS_ID", how="left").fillna({c:0 for c in ["AAY_tons","PHH_tons","APL_tons","NSFA_tons"]})
            # ratios strings
            def ratio_row(r):
                a = r.get("AAY_tons",0.0)
                p = r.get("PHH_tons",0.0)
                l = r.get("APL_tons",0.0)
                ns = r.get("NSFA_tons",0.0)
                # AAY:PHH:APL
                denom = max(1e-9, a+p+l)
                return f"{a/denom:.2f}:{p/denom:.2f}:{l/denom:.2f}"
            report["AAY:PHH:APL_ratio"] = report.apply(ratio_row, axis=1)
            report["NSFA:APL_ratio"] = report.apply(lambda r: (r.get("NSFA_tons",0.0)/ (r.get("APL_tons",1e-9)) if r.get("APL_tons",0.0)>0 else float("inf")), axis=1)
        report["Trips_Count"] = report["Trips_Count"].fillna(0).astype(int)
        report["Vehicle_IDs"] = report["Vehicle_IDs"].fillna("")
        report = report[["FPS_ID","FPS_Name","Total_Dispatched_tons","Trips_Count","Vehicle_IDs"] + ([c for c in ["AAY_tons","PHH_tons","APL_tons","NSFA_tons","AAY:PHH:APL_ratio","NSFA:APL_ratio"] if c in report.columns])]
        report = report.sort_values("Total_Dispatched_tons", ascending=False)
    st.dataframe(report, use_container_width=True)

# ---------------------------
# FPS At-Risk
# ---------------------------
with tab5:
    st.subheader("FPS At-Risk List")
    if not fps_stock.empty:
        arf = fps_stock.copy()
        if use_dates and "Date" in arf.columns:
            arf = arf[(arf["Date"].dt.date >= date_range[0]) & (arf["Date"].dt.date <= date_range[1])]
        else:
            arf = arf.query("Day>=@day_range[0] & Day<=@day_range[1]") if not arf.empty else arf
        arf = arf[["Day","Date","FPS_ID","Stock_Level_tons","Reorder_Threshold_tons","At_Risk"]]
    else:
        arf = pd.DataFrame(columns=["Day","Date","FPS_ID","Stock_Level_tons","Reorder_Threshold_tons","At_Risk"])
    st.dataframe(arf, use_container_width=True)
    st.download_button("Download At-Risk (Excel)", to_excel(arf), "fps_at_risk.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------
# FPS Data
# ---------------------------
with tab6:
    st.subheader("FPS Stock & Upcoming Receipts")
    end_day = None
    if use_dates:
        end_day = date_range[1]
    else:
        end_day = min(day_range[1], int(stock_levels["Day"].max() if not stock_levels.empty else day_range[1]))
    fps_data = []
    for fps_id in (fps["FPS_ID"] if "FPS_ID" in fps.columns else []):
        if use_dates:
            s = fps_stock[(fps_stock["FPS_ID"]==fps_id) & (fps_stock["Date"].dt.date==end_day)]["Stock_Level_tons"]
        else:
            s = fps_stock[(fps_stock["FPS_ID"]==fps_id) & (fps_stock["Day"]==end_day)]["Stock_Level_tons"]
        stock_now = float(s.iloc[0]) if not s.empty else 0.0
        future = dispatch_lg[(dispatch_lg["FPS_ID"]==fps_id) & ((dispatch_lg["Date"].dt.date> end_day) if use_dates else (dispatch_lg["Day"]> end_day))]["Day" if not use_dates else "Date"]
        next_day = int(future.min()) if (not future.empty and not use_dates) else (future.min() if not future.empty and use_dates else None)
        days_to = ( (pd.to_datetime(next_day).date() - end_day).days if use_dates and next_day is not None else (next_day - end_day if (next_day is not None and not use_dates) else None) )
        fps_data.append({
            "FPS_ID": fps_id,
            "FPS_Name": fps.set_index("FPS_ID").loc[fps_id,"FPS_Name"] if "FPS_Name" in fps.columns else None,
            "Current_Stock_tons": stock_now,
            "Next_Receipt_Day": next_day,
            "Days_To_Receipt": days_to
        })
    fps_data_df = pd.DataFrame(fps_data)
    st.dataframe(fps_data_df, use_container_width=True)
    st.download_button("Download FPS Data (Excel)", to_excel(fps_data_df), "fps_data.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------
# Downloads
# ---------------------------
with tab7:
    st.subheader("Download FPS Report")
    st.download_button("Excel", to_excel(report), f"FPS_Report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if isinstance(report, pd.DataFrame) and not report.empty:
        pdf_buf = BytesIO()
        with PdfPages(pdf_buf) as pdf:
            fig, ax = plt.subplots(figsize=(8, max(1, len(report)*0.3) + 1))
            ax.axis('off')
            tbl = ax.table(cellText=report.values, colLabels=report.columns, loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            pdf.savefig(fig, bbox_inches='tight')
        st.download_button("PDF", pdf_buf.getvalue(), "FPS_Report.pdf", mime="application/pdf")
    else:
        st.info("No rows in the selected window to export as PDF.")

# ---------------------------
# Metrics
# ---------------------------
with tab8:
    st.subheader("Key Performance Indicators")
    if use_dates:
        end_date = date_range[1]
        sel_days = (date_range[1] - date_range[0]).days + 1
    else:
        end_day = min(day_range[1], int(stock_levels["Day"].max() if not stock_levels.empty else day_range[1]))
        sel_days = day_range[1] - max(day_range[0],1) + 1

    cg_sel   = (dispatch_cg[(dispatch_cg["Date"].dt.date >= date_range[0]) & (dispatch_cg["Date"].dt.date <= date_range[1])]["Quantity_tons"].sum() if (use_dates and not dispatch_cg.empty) else (day_totals_cg.query("Day>=@day_range[0] & Day<=@day_range[1]")["Quantity_tons"].sum() if not day_totals_cg.empty else 0.0))
    lg_sel   = (dispatch_lg[(dispatch_lg["Date"].dt.date >= date_range[0]) & (dispatch_lg["Date"].dt.date <= date_range[1])]["Quantity_tons"].sum() if (use_dates and not dispatch_lg.empty) else (day_totals_lg.query("Day>=@day_range[0] & Day<=@day_range[1]")["Quantity_tons"].sum() if not day_totals_lg.empty else 0.0))
    avg_daily_cg = cg_sel / sel_days if sel_days>0 else 0
    avg_daily_lg = lg_sel / sel_days if sel_days>0 else 0

    avg_trips = 0.0
    if not veh_usage.empty:
        if use_dates and "Date" in veh_usage.columns:
            window = veh_usage[(veh_usage["Date"].dt.date >= date_range[0]) & (veh_usage["Date"].dt.date <= date_range[1])]["Trips_Used"]
        else:
            window = veh_usage.query("Day>=@day_range[0] & Day<=@day_range[1]")["Trips_Used"]
        avg_trips = float(window.mean()) if not window.empty else 0.0

    if not lg_stock.empty and ( (use_dates and any(pd.to_datetime(lg_stock.index, errors='coerce').notna())) or (not use_dates and sel_days) ) and selected_lg_ids:
        # If lg_stock pivoted by Day x LGs exist, sum columns matching selected_lg_ids for the end day
        try:
            if use_dates:
                # find day number corresponding to last date
                # we can map by matching stock_levels DataFrame instead
                dflg = stock_levels[(stock_levels["Entity_Type"]=="LG") & (stock_levels["Date"].dt.date==date_range[1])]
                lg_onhand = dflg[dflg["Entity_ID"].isin(selected_lg_ids)]["Stock_Level_tons"].sum() if not dflg.empty else 0.0
            else:
                if end_day in lg_stock.index:
                    lg_onhand = lg_stock.loc[end_day, [c for c in lg_stock.columns if c in selected_lg_ids]].sum()
                else:
                    lg_onhand = 0.0
        except Exception:
            lg_onhand = 0.0
    else:
        lg_onhand = 0.0

    fps_onhand = (fps_stock[(fps_stock["Date"].dt.date == date_range[1])]["Stock_Level_tons"].sum() if (use_dates and not fps_stock.empty and "Date" in fps_stock.columns) else (fps_stock.query("Day==@end_day")["Stock_Level_tons"].sum() if not fps_stock.empty and not use_dates else 0.0))

    if "Storage_Capacity_tons" in lgs.columns:
        lg_caps = lgs[lgs["LG_ID"].isin(selected_lg_ids)]["Storage_Capacity_tons"].sum()
    else:
        lg_caps = 0.0
    pct_lg_filled = (lg_onhand/lg_caps)*100 if lg_caps else 0.0

    fps_zero = (fps_stock[(fps_stock["Date"].dt.date==date_range[1])]["FPS_ID"].nunique() if (use_dates and "Date" in fps_stock.columns) else (fps_stock.query("Day==@end_day")["FPS_ID"].nunique() if not fps_stock.empty and not use_dates else 0))
    fps_risk = (fps_stock[(fps_stock["Date"].dt.date==date_range[1]) & (fps_stock["Stock_Level_tons"]<=fps_stock["Reorder_Threshold_tons"])]["FPS_ID"].nunique() if (use_dates and "Date" in fps_stock.columns) else (fps_stock.query("Day==@end_day & At_Risk")["FPS_ID"].nunique() if not fps_stock.empty and not use_dates else 0))

    dispatched_cum = (dispatch_lg[dispatch_lg["Date"].dt.date <= date_range[1]]["Quantity_tons"].sum() if (use_dates and not dispatch_lg.empty) else (day_totals_lg.query("Day<=@end_day")["Quantity_tons"].sum() if not day_totals_lg.empty else 0.0))
    total_plan = (day_totals_lg["Quantity_tons"].sum() if not day_totals_lg.empty else 0.0)
    pct_plan = (dispatched_cum/total_plan)*100 if total_plan else 0.0
    remaining_t = total_plan - dispatched_cum
    days_rem = math.ceil(remaining_t/DAILY_CAP) if DAILY_CAP else None

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
