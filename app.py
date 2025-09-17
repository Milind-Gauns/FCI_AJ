# app.py
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title="Grain Distribution Dashboard", layout="wide")
st.title("🚛 Grain Distribution Dashboard")

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
    "LG_to_FPS": ["LG_to_FPS", "LG_to_LPS_Dispatch", "LG_to_FPS_Dispatch"],
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

    for tag, need in REQUIRED_COLS.items():
        _need_cols(dfs[tag], need, tag)

    # keep Vehicle_ID as string; numeric-coerce only numeric fields
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

    # Accept that the simulator added Date, AAY/PHH/APL splits
    # Ensure Date exists (fallback: Day -> string)
    if "Date" not in dispatch_lg.columns:
        dispatch_lg["Date"] = dispatch_lg["Day"].apply(lambda d: str(int(d)))
    if "Date" not in dispatch_cg.columns:
        dispatch_cg["Date"] = dispatch_cg["Day"].apply(lambda d: str(int(d)))
    if "Date" not in stock_levels.columns:
        stock_levels["Date"] = stock_levels["Day"].apply(lambda d: str(int(d)))

    # Category columns may be missing if old outputs were used; ensure they exist
    for col in ("AAY_tons","PHH_tons","APL_tons","NFSA_tons"):
        if col not in dispatch_lg.columns:
            dispatch_lg[col] = 0.0
        if col not in dispatch_cg.columns:
            dispatch_cg[col] = 0.0

    # FPS monthly demand: compute from counts if provided (app1 already does this, but be tolerant)
    for col in ("AAY_Count","PHH_Beneficiaries","APL_Count"):
        if col not in fps.columns:
            fps[col] = 0.0
    fps["Monthly_from_counts_kg"] = fps["AAY_Count"] * _get_setting(settings, "AAY_kg_per_card", 35.0) + \
                                    fps["PHH_Beneficiaries"] * _get_setting(settings, "PHH_kg_per_beneficiary", 5.0) + \
                                    fps["APL_Count"] * _get_setting(settings, "APL_kg_per_card", 0.0)
    fps["Monthly_Demand_tons"] = (fps["Monthly_from_counts_kg"] / 1000.0).fillna(0.0)
    fps["Daily_Demand_tons"] = fps["Monthly_Demand_tons"] / 30.0

    # aggregates
    day_totals_cg = (dispatch_cg.groupby("Day", as_index=False)["Quantity_tons"].sum()
                     if not dispatch_cg.empty else pd.DataFrame(columns=["Day","Quantity_tons"]))
    day_totals_lg = (dispatch_lg.groupby("Day", as_index=False)["Quantity_tons"].sum()
                     if not dispatch_lg.empty else pd.DataFrame(columns=["Day","Quantity_tons"]))

    veh_usage = (
        dispatch_lg.groupby("Day").size().reset_index(name="Trips_Used")
        if not dispatch_lg.empty else pd.DataFrame(columns=["Day","Trips_Used"])
    )
    VEH_TOTAL = int(_get_setting(settings, "Vehicles_Total", 30, int))
    MAX_TRIPS = int(_get_setting(settings, "Max_Trips_Per_Vehicle_Per_Day", 3, int))
    veh_usage["Max_Trips"] = VEH_TOTAL * MAX_TRIPS

    # LG stock pivot (Date axis keeps Day index)
    lg_stock = (stock_levels[stock_levels["Entity_Type"]=="LG"]
                .pivot(index="Day", columns="Entity_ID", values="Stock_Level_tons")
                .sort_index().ffill())

    fps_stock = (stock_levels[stock_levels["Entity_Type"]=="FPS"]
                 .merge(fps[["FPS_ID","Reorder_Threshold_tons"]], left_on="Entity_ID", right_on="FPS_ID", how="left"))
    fps_stock["At_Risk"] = fps_stock["Stock_Level_tons"] <= fps_stock["Reorder_Threshold_tons"]

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
