# simulation.py
import streamlit
import io
import pandas as pd
import math
import numpy as np

def run_simulation(
    master_workbook,          # str path or file-like buffer
    settings: pd.DataFrame,
    lgs: pd.DataFrame,
    fps: pd.DataFrame,
    vehicles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs a two-phase simulation (unchanged logic) but:
    - Adds 'Date' mapping for each Day (skips Sundays).
    - Computes FPS demand from RC counts (AAY, PHH, APL) using Settings eligibilities.
    - Splits dispatched quantities into AAY / PHH / APL (and NSFA=AAY+PHH).
    Returns:
        (dispatch_cg, dispatch_lg, stock_levels)
    """

    # -----------------------------
    # 0) Read key parameters safely
    # -----------------------------
    def _get_setting(param_name, default=None, cast=float):
        try:
            val = settings.loc[settings["Parameter"] == param_name, "Value"].iloc[0]
            return cast(val)
        except Exception:
            if default is None:
                raise ValueError(f"Missing required setting: {param_name}")
            return cast(default)

    # numeric settings
    DAYS       = _get_setting("Distribution_Days", default=30, cast=int)
    TRUCK_CAP  = _get_setting("Vehicle_Capacity_tons", default=11.5, cast=float)
    TOT_V      = _get_setting("Vehicles_Total", default=30, cast=int)
    MAX_TRIPS  = _get_setting("Max_Trips_Per_Vehicle_Per_Day", default=3, cast=int)
    DEFAULT_LEAD = _get_setting("Default_Lead_Time_days", default=3.0, cast=float)

    # optional eligibility settings (kg). Defaults: AAY=35 kg, PHH=5 kg, APL=0 kg
    try:
        AAY_kg = float(settings.loc[settings["Parameter"] == "AAY_kg_per_card", "Value"].iloc[0])
    except Exception:
        AAY_kg = 35.0
    try:
        PHH_kg = float(settings.loc[settings["Parameter"] == "PHH_kg_per_beneficiary", "Value"].iloc[0])
    except Exception:
        PHH_kg = 5.0
    try:
        APL_kg = float(settings.loc[settings["Parameter"] == "APL_kg_per_card", "Value"].iloc[0])
    except Exception:
        APL_kg = 0.0

    # Start_Date (optional). If provided, must be parseable by pandas.to_datetime.
    try:
        sd = settings.loc[settings["Parameter"] == "Start_Date", "Value"].iloc[0]
        start_date = pd.to_datetime(sd)
    except Exception:
        start_date = pd.Timestamp.today().normalize()

    # -----------------------------
    # Build Day -> Date mapping (skip Sundays)
    # -----------------------------
    dates = []
    cur = start_date
    # keep iterating until we collect DAYS dates that are not Sundays
    while len(dates) < DAYS:
        # Monday=0 ... Sunday=6
        if cur.weekday() != 6:  # skip Sundays
            dates.append(pd.Timestamp(cur).normalize())
        cur = cur + pd.Timedelta(days=1)
    day_to_date = {i+1: dates[i] for i in range(len(dates))}

    # -----------------------------
    # 1) Prepare LG & FPS mappings
    # -----------------------------
    lgs = lgs.copy()
    if "LG_ID" not in lgs.columns or "LG_Name" not in lgs.columns:
        raise ValueError("LGs sheet must contain columns: LG_ID, LG_Name")

    lgid_by_name = {str(nm).strip().lower(): int(lg_id) for lg_id, nm in zip(lgs["LG_ID"], lgs["LG_Name"])}
    valid_lg_ids = set(int(x) for x in lgs["LG_ID"])

    def normalize_lg_ref(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        try:
            i = int(float(s))
            return i if i in valid_lg_ids else None
        except Exception:
            return lgid_by_name.get(s.lower())

    req_cols = {"FPS_ID", "Max_Capacity_tons", "Linked_LG_ID"}
    missing = req_cols - set(fps.columns)
    if missing:
        raise ValueError(f"FPS sheet missing required columns: {missing}")

    fps = fps.copy()
    # Ensure Lead_Time_days exists and fill NaN with default
    if "Lead_Time_days" not in fps.columns:
        fps["Lead_Time_days"] = DEFAULT_LEAD
    else:
        fps["Lead_Time_days"] = fps["Lead_Time_days"].fillna(DEFAULT_LEAD)

    # -----------------------------
    # 1a) Compute demand from counts if Monthly_Demand_tons missing
    # Acceptable FPS count columns (we detect common names):
    #    AAY_Count  OR AAY_Cards
    #    PHH_Beneficiaries OR PHH_Count
    #    APL_Count OR APL_Cards
    # If Monthly_Demand_tons exists it will be used; otherwise monthly demand derived from counts.
    # Counts are assumed to be monthly totals (if a different cadence is used, change later).
    # -----------------------------
    # helper to find column
    def first_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    aay_col = first_col(fps, ["AAY_Count", "AAY_Cards"])
    phh_col = first_col(fps, ["PHH_Beneficiaries", "PHH_Count", "PHH_Beneficiary"])
    apl_col = first_col(fps, ["APL_Count", "APL_Cards"])

    # Build per-FPS monthly demand (kg) from counts if Monthly_Demand_tons absent or NA
    # If Monthly_Demand_tons exists we'll still compute per-category breakdowns if counts present.
    # compute counts filled with zeros if missing
    fps["AAY_Count"] = fps[aay_col].fillna(0).astype(float) if aay_col is not None else 0.0
    fps["PHH_Count"] = fps[phh_col].fillna(0).astype(float) if phh_col is not None else 0.0
    fps["APL_Count"] = fps[apl_col].fillna(0).astype(float) if apl_col is not None else 0.0

    # monthly kg from counts
    fps["Monthly_from_counts_kg"] = (
        fps["AAY_Count"] * AAY_kg +
        fps["PHH_Count"] * PHH_kg +
        fps["APL_Count"] * APL_kg
    )

    # If Monthly_Demand_tons present, prefer it; else derive from counts
    if "Monthly_Demand_tons" in fps.columns:
        fps["Monthly_Demand_tons"] = pd.to_numeric(fps["Monthly_Demand_tons"], errors="coerce")
    fps["Monthly_Demand_tons"] = fps.get("Monthly_Demand_tons").fillna(fps["Monthly_from_counts_kg"] / 1000.0)

    # per-category monthly tons (from counts)
    fps["AAY_Monthly_tons"] = fps["AAY_Count"] * (AAY_kg / 1000.0)
    fps["PHH_Monthly_tons"] = fps["PHH_Count"] * (PHH_kg / 1000.0)
    fps["APL_Monthly_tons"] = fps["APL_Count"] * (APL_kg / 1000.0)

    # per-category daily tons
    fps["Daily_Demand_tons"] = fps["Monthly_Demand_tons"] / 30.0
    fps["AAY_Daily_tons"] = fps["AAY_Monthly_tons"] / 30.0
    fps["PHH_Daily_tons"] = fps["PHH_Monthly_tons"] / 30.0
    fps["APL_Daily_tons"] = fps["APL_Monthly_tons"] / 30.0

    # NSFA components (AAY + PHH)
    fps["NSFA_Daily_tons"] = fps["AAY_Daily_tons"] + fps["PHH_Daily_tons"]
    fps["NSFA_Monthly_tons"] = fps["AAY_Monthly_tons"] + fps["PHH_Monthly_tons"]

    # Reorder threshold uses total Daily_Demand_tons and Lead_Time_days
    fps["Reorder_Threshold_tons"] = fps["Daily_Demand_tons"] * fps["Lead_Time_days"]

    # Attach LG_ID normalized
    fps["LG_ID"] = fps["Linked_LG_ID"].apply(normalize_lg_ref)
    if fps["LG_ID"].isna().any():
        bad_rows = fps[fps["LG_ID"].isna()][["FPS_ID", "Linked_LG_ID"]]
        raise ValueError(
            "Some FPS rows couldn't map Linked_LG_ID to a valid LG_ID. "
            f"Examples:\n{bad_rows.head(5).to_string(index=False)}"
        )
    fps["LG_ID"] = fps["LG_ID"].astype(int)

    # -----------------------------
    # 2) Prepare Vehicles mapping
    # -----------------------------
    vehicles = vehicles.copy()
    if vehicles.empty:
        vehicles = pd.DataFrame({
            "Vehicle_ID": list(range(1, TOT_V + 1)),
            "Capacity_tons": [TRUCK_CAP] * TOT_V,
            "Mapped_LG_IDs": [",".join(str(x) for x in sorted(valid_lg_ids))] * TOT_V
        })
    else:
        if "Vehicle_ID" not in vehicles.columns:
            raise ValueError("Vehicles sheet must contain 'Vehicle_ID'")
        if "Capacity_tons" not in vehicles.columns:
            vehicles["Capacity_tons"] = TRUCK_CAP
        if "Mapped_LG_IDs" not in vehicles.columns:
            vehicles["Mapped_LG_IDs"] = ",".join(str(x) for x in sorted(valid_lg_ids))

    def parse_lg_list(val):
        if pd.isna(val):
            return []
        out = []
        for token in str(val).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                i = int(float(token))
                if i in valid_lg_ids:
                    out.append(i)
                    continue
            except Exception:
                pass
            mapped = normalize_lg_ref(token)
            if mapped is not None:
                out.append(mapped)
        return sorted(set(out))

    vehicles["Mapped_LGs_List"] = vehicles["Mapped_LG_IDs"].apply(parse_lg_list)
    if vehicles["Mapped_LGs_List"].apply(len).eq(0).any():
        bad = vehicles[vehicles["Mapped_LGs_List"].apply(len).eq(0)][["Vehicle_ID", "Mapped_LG_IDs"]]
        raise ValueError(
            "Some vehicles couldn't map any LGs from 'Mapped_LG_IDs'. "
            f"Examples:\n{bad.head(5).to_string(index=False)}"
        )

    # -----------------------------
    # 3) LG → FPS SIMULATION
    # -----------------------------
    if "Initial_Allocation_tons" not in lgs.columns:
        lgs["Initial_Allocation_tons"] = 0.0

    lg_stock = {int(row["LG_ID"]): float(row["Initial_Allocation_tons"]) for _, row in lgs.iterrows()}
    fps_stock = {int(fid): 0.0 for fid in fps["FPS_ID"]}

    dispatch_lg_rows = []
    stock_rows = []

    for day in range(1, DAYS + 1):
        # 3a) FPS consumes daily demand
        for _, r in fps.iterrows():
            fid = int(r["FPS_ID"])
            fps_stock[fid] = max(0.0, fps_stock[fid] - float(r["Daily_Demand_tons"]))

        # 3b) Compute needs (total demand-driven logic unchanged)
        needs = []
        for _, r in fps.iterrows():
            fid  = int(r["FPS_ID"])
            lgid = int(r["LG_ID"])
            current = fps_stock[fid]
            threshold = float(r["Reorder_Threshold_tons"])
            max_cap  = float(r["Max_Capacity_tons"])
            if current <= threshold:
                available_at_lg = lg_stock.get(lgid, 0.0)
                need_qty = min(max_cap - current, available_at_lg)
                if need_qty > 0:
                    urgency = (threshold - current) / float(r["Daily_Demand_tons"]) if r["Daily_Demand_tons"] > 0 else 0
                    needs.append((urgency, fid, lgid, need_qty))
        needs.sort(reverse=True, key=lambda x: x[0])

        # 3c) Reset vehicle usage counters for the day
        vehicles["Trips_Used"] = 0

        # 3d) Dispatch loop
        for urgency, fid, lgid, need_qty in needs:
            cand = vehicles[vehicles["Mapped_LGs_List"].apply(lambda lst: lgid in lst)].copy()
            cand = cand[cand["Trips_Used"] < MAX_TRIPS]
            if cand.empty:
                continue

            cand["is_shared"] = cand["Mapped_LGs_List"].apply(lambda lst: len(lst) > 1)
            cand = cand.sort_values(["is_shared"], ascending=False)
            chosen = cand.iloc[0]

            vid = chosen["Vehicle_ID"]
            cap = float(chosen["Capacity_tons"])
            qty = min(cap, need_qty, lg_stock.get(lgid, 0.0))
            if qty <= 0:
                continue

            # Determine fractional split of delivered qty into AAY/PHH/APL using the FPS per-category daily demands
            r = fps.loc[fps["FPS_ID"] == fid].iloc[0]
            total_d = float(r["Daily_Demand_tons"]) if r["Daily_Demand_tons"] > 0 else 0.0
            aay_d = float(r.get("AAY_Daily_tons", 0.0))
            phh_d = float(r.get("PHH_Daily_tons", 0.0))
            apl_d = float(r.get("APL_Daily_tons", 0.0))

            if total_d <= 0:
                # If no demand recorded, attribute everything to APL=0 and NSFA=0 (keep 0s)
                aay_del = phh_del = apl_del = 0.0
            else:
                aay_del = qty * (aay_d / total_d)
                phh_del = qty * (phh_d / total_d)
                apl_del = qty * (apl_d / total_d)

            nsfa_del = aay_del + phh_del

            dispatch_lg_rows.append({
                "Day": int(day),
                "Date": day_to_date[int(day)],
                "Vehicle_ID": vid,
                "LG_ID": int(lgid),
                "FPS_ID": int(fid),
                "Quantity_tons": float(qty),
                "AAY_tons": float(aay_del),
                "PHH_tons": float(phh_del),
                "APL_tons": float(apl_del),
                "NSFA_tons": float(nsfa_del)
            })

            # update stocks & vehicle usage
            lg_stock[lgid] = lg_stock.get(lgid, 0.0) - qty
            fps_stock[fid] = fps_stock.get(fid, 0.0) + qty
            vehicles.loc[vehicles["Vehicle_ID"] == vid, "Trips_Used"] += 1

        # 3e) Record end-of-day stocks (add Date)
        for lgid, st in lg_stock.items():
            stock_rows.append({"Day": int(day), "Date": day_to_date[int(day)], "Entity_Type": "LG",  "Entity_ID": int(lgid), "Stock_Level_tons": float(st)})
        for fid, st in fps_stock.items():
            stock_rows.append({"Day": int(day), "Date": day_to_date[int(day)], "Entity_Type": "FPS", "Entity_ID": int(fid),  "Stock_Level_tons": float(st)})

    dispatch_lg = pd.DataFrame(dispatch_lg_rows, columns=[
        "Day","Date","Vehicle_ID","LG_ID","FPS_ID","Quantity_tons","AAY_tons","PHH_tons","APL_tons","NSFA_tons"
    ])
    stock_levels = pd.DataFrame(stock_rows, columns=["Day","Date","Entity_Type","Entity_ID","Stock_Level_tons"])

    if dispatch_lg.empty:
        dispatch_lg = pd.DataFrame(columns=["Day","Date","Vehicle_ID","LG_ID","FPS_ID","Quantity_tons","AAY_tons","PHH_tons","APL_tons","NSFA_tons"])

    # -----------------------------------------------
    # 4) Derive LG daily requirement from dispatch_lg
    # -----------------------------------------------
    required_cols = {"LG_ID", "Day", "Quantity_tons"}
    missing = required_cols - set(dispatch_lg.columns)
    if missing:
        raise ValueError(f"dispatch_lg is missing required columns: {missing}")

    if dispatch_lg.empty:
        lg_daily_req = (
            pd.MultiIndex.from_product([sorted(valid_lg_ids), range(1, DAYS + 1)], names=["LG_ID","Day"])
            .to_frame(index=False)
            .assign(Daily_Requirement_tons=0.0)
        )
    else:
        lg_daily_req = (
            dispatch_lg
            .groupby(["LG_ID", "Day"])["Quantity_tons"]
            .sum()
            .reset_index()
            .rename(columns={"Quantity_tons": "Daily_Requirement_tons"})
        )

    req_pivot = lg_daily_req.pivot_table(
        index="LG_ID", columns="Day",
        values="Daily_Requirement_tons",
        aggfunc="sum", fill_value=0.0
    )

    # Also build LG-level per-category demand pivot (AAY/PHH/APL) from fps definition:
    # sum of per-FPS AAY/PHH/APL daily demand grouped by LG
    fps_cats = fps[["LG_ID", "AAY_Daily_tons", "PHH_Daily_tons", "APL_Daily_tons", "Daily_Demand_tons"]].copy()
    fps_cats["LG_ID"] = fps_cats["LG_ID"].astype(int)

    lg_cat = fps_cats.groupby("LG_ID").sum().reindex(index=sorted(valid_lg_ids), fill_value=0.0)
    # But we need per-day breakdown (for CG scheduling): use the fps per-FPS per-day values and expand over days (constant across days)
    # so LG per-day AAY/PHH/APL demand = per-LG total daily from fps (constant each day)
    # Build pivot frames where each Day column contains the per-LG daily category demand
    lg_ids_sorted = sorted(int(x) for x in lgs["LG_ID"].dropna().astype(int).unique())
    days_list = list(range(1, DAYS + 1))

    lg_aay_piv = pd.DataFrame(
        {d: lg_cat["AAY_Daily_tons"].values for d in days_list},
        index=lg_cat.index
    ).reindex(index=lg_ids_sorted, fill_value=0.0)

    lg_phh_piv = pd.DataFrame(
        {d: lg_cat["PHH_Daily_tons"].values for d in days_list},
        index=lg_cat.index
    ).reindex(index=lg_ids_sorted, fill_value=0.0)

    lg_apl_piv = pd.DataFrame(
        {d: lg_cat["APL_Daily_tons"].values for d in days_list},
        index=lg_cat.index
    ).reindex(index=lg_ids_sorted, fill_value=0.0)

    # -----------------------------------------------
    # 5) CG → LG PRE-DISPATCH (same DAYS timeline)
    # -----------------------------------------------
    try:
        cap_df = pd.read_excel(master_workbook, sheet_name="LG_Capacity")
        if {"LG_ID", "Capacity_tons"} <= set(cap_df.columns):
            capacity = {int(r["LG_ID"]): float(r["Capacity_tons"]) for _, r in cap_df.iterrows()}
        else:
            raise ValueError
    except Exception:
        if "Storage_Capacity_tons" not in lgs.columns:
            raise ValueError("Provide LG_Capacity sheet or 'Storage_Capacity_tons' in LGs.")
        capacity = {int(r["LG_ID"]): float(r["Storage_Capacity_tons"]) for _, r in lgs.iterrows()}

    lg_stock_base = {int(r["LG_ID"]): float(r.get("Initial_LG_stock", 0.0)) for _, r in lgs.iterrows()}

    req_pivot = req_pivot.copy()
    req_pivot.index = [int(x) for x in req_pivot.index]
    req_pivot.columns = [int(c) for c in req_pivot.columns]

    def _get_demand(lg_id: int, day: int) -> float:
        try:
            return float(req_pivot.at[lg_id, day])
        except Exception:
            return 0.0

    def _get_cat_demand(lg_id: int, day: int):
        # return (aay, phh, apl) daily demand for this lg/day (we use the constant daily per-LG values)
        a = float(lg_aay_piv.at[lg_id, day]) if (lg_id in lg_aay_piv.index and day in lg_aay_piv.columns) else 0.0
        p = float(lg_phh_piv.at[lg_id, day]) if (lg_id in lg_phh_piv.index and day in lg_phh_piv.columns) else 0.0
        l = float(lg_apl_piv.at[lg_id, day]) if (lg_id in lg_apl_piv.index and day in lg_apl_piv.columns) else 0.0
        return a, p, l

    def _free_room(stock: dict, lg_id: int) -> float:
        return max(0.0, capacity.get(lg_id, 0.0) - stock.get(lg_id, 0.0))

    def _simulate(pre_days: int, collect_rows: bool = False, include_pre_days: bool = False):
        start_day = 1 - pre_days
        stock = {lg: lg_stock_base.get(lg, 0.0) for lg in lg_ids}
        rows = [] if collect_rows else None

        for day in range(start_day, DAYS + 1):
            trips_left = TOT_V

            if day >= 1:
                order = sorted(lg_ids, key=lambda lg: -(_get_demand(lg, day) - stock[lg]))
                for lg in order:
                    demand_today = _get_demand(lg, day)
                    need_today = max(0.0, demand_today - stock[lg])

                    while trips_left > 0 and need_today > 1e-9:
                        room = _free_room(stock, lg)
                        if room <= 1e-9:
                            break
                        qty = min(TRUCK_CAP, need_today, room)
                        if qty <= 1e-9:
                            break

                        # compute category split for this LG/day
                        aay_d, phh_d, apl_d = _get_cat_demand(lg, day)
                        total_cat = aay_d + phh_d + apl_d
                        if total_cat <= 0:
                            aay_del = phh_del = apl_del = 0.0
                        else:
                            aay_del = qty * (aay_d / total_cat)
                            phh_del = qty * (phh_d / total_cat)
                            apl_del = qty * (apl_d / total_cat)

                        if collect_rows and (include_pre_days or day >= 1):
                            vid = TOT_V - trips_left + 1
                            rows.append({
                                "Day": int(day),
                                "Date": (day_to_date[int(day)] if int(day) in day_to_date else pd.NaT),
                                "Vehicle_ID": int(vid),
                                "LG_ID": int(lg),
                                "Quantity_tons": float(qty),
                                "AAY_tons": float(aay_del),
                                "PHH_tons": float(phh_del),
                                "APL_tons": float(apl_del),
                                "NSFA_tons": float(aay_del + phh_del)
                            })

                        stock[lg] += qty
                        trips_left -= 1
                        need_today -= qty

                    if stock[lg] + 1e-6 < demand_today:
                        return False, (rows or []), start_day, stock

            # B) Pre-stock round-robin with remaining trips
            if trips_left > 0:
                future_unmet = {
                    lg: max(0.0, sum(_get_demand(lg, d) for d in range(max(1, day + 1), DAYS + 1)) - stock[lg])
                    for lg in lg_ids
                }
                candidates = [lg for lg, fu in future_unmet.items() if fu > 1e-6 and _free_room(stock, lg) > 1e-6]
                idx = 0
                while trips_left > 0 and candidates:
                    lg = candidates[idx % len(candidates)]
                    room = _free_room(stock, lg)
                    deliver = min(TRUCK_CAP, future_unmet[lg], room)

                    if deliver > 1e-9:
                        # category split for pre-stocking delivery (same method)
                        aay_d, phh_d, apl_d = _get_cat_demand(lg, day)
                        total_cat = aay_d + phh_d + apl_d
                        if total_cat <= 0:
                            aay_del = phh_del = apl_del = 0.0
                        else:
                            aay_del = deliver * (aay_d / total_cat)
                            phh_del = deliver * (phh_d / total_cat)
                            apl_del = deliver * (apl_d / total_cat)

                        if collect_rows and (include_pre_days or day >= 1):
                            vid = TOT_V - trips_left + 1
                            rows.append({
                                "Day": int(day),
                                "Date": (day_to_date[int(day)] if int(day) in day_to_date else pd.NaT),
                                "Vehicle_ID": int(vid),
                                "LG_ID": int(lg),
                                "Quantity_tons": float(deliver),
                                "AAY_tons": float(aay_del),
                                "PHH_tons": float(phh_del),
                                "APL_tons": float(apl_del),
                                "NSFA_tons": float(aay_del + phh_del)
                            })
                        stock[lg] += deliver
                        future_unmet[lg] = max(0.0, future_unmet[lg] - deliver)
                        trips_left -= 1

                    if future_unmet[lg] < 1e-6 or _free_room(stock, lg) < 1e-6:
                        candidates.remove(lg)
                        idx -= 1
                    idx += 1

            if day >= 1:
                for lg in lg_ids:
                    stock[lg] = max(0.0, stock[lg] - _get_demand(lg, day))

        return True, (rows or []), start_day, stock

    MAX_PRE_DAYS = 30
    pre_days = None
    for x in range(0, MAX_PRE_DAYS + 1):
        ok, _, start_day, _ = _simulate(pre_days=x, collect_rows=False)
        if ok:
            pre_days = x
            break

    if pre_days is None:
        raise RuntimeError("Unable to meet all demands within MAX_PRE_DAYS.")

    ok, rows, start_day, _ = _simulate(pre_days=pre_days, collect_rows=True, include_pre_days=True)
    assert ok

    dispatch_cg = pd.DataFrame(rows, columns=["Day","Date","Vehicle_ID","LG_ID","Quantity_tons","AAY_tons","PHH_tons","APL_tons","NSFA_tons"])

    # === Accurate LG stock levels, include pre-days as before ===
    lg_ids_sorted = sorted(int(x) for x in lgs["LG_ID"].dropna().astype(int).unique())

    if "Initial_LG_stock" in lgs.columns:
        init_series = (
            lgs.assign(LG_ID=lgs["LG_ID"].astype(int))
               .set_index("LG_ID")["Initial_LG_stock"]
               .reindex(lg_ids_sorted).fillna(0.0)
        )
    else:
        init_series = pd.Series(0.0, index=lg_ids_sorted)

    if not dispatch_cg.empty:
        dcg = dispatch_cg.copy()
        dcg["LG_ID"] = dcg["LG_ID"].astype(int)
        dcg["Day"]   = dcg["Day"].astype(int)
        cg_piv = dcg.pivot_table(index="LG_ID", columns="Day",
                                 values="Quantity_tons", aggfunc="sum", fill_value=0.0)
        full_cols = list(range(start_day, DAYS + 1))
        cg_piv = cg_piv.reindex(index=lg_ids_sorted, columns=full_cols, fill_value=0.0)
        cg_cum = cg_piv.cumsum(axis=1).reindex(columns=list(range(1, DAYS + 1)), fill_value=0.0)
    else:
        cg_cum = pd.DataFrame(0.0, index=lg_ids_sorted, columns=list(range(1, DAYS + 1)))

    if not dispatch_lg.empty:
        dlg = dispatch_lg.copy()
        dlg["LG_ID"] = dlg["LG_ID"].astype(int)
        dlg["Day"]   = dlg["Day"].astype(int)
        lg_piv = dlg.pivot_table(index="LG_ID", columns="Day",
                                 values="Quantity_tons", aggfunc="sum", fill_value=0.0)
        lg_piv = lg_piv.reindex(index=lg_ids_sorted, columns=list(range(1, DAYS + 1)), fill_value=0.0)
        lg_cum = lg_piv.cumsum(axis=1)
    else:
        lg_cum = pd.DataFrame(0.0, index=lg_ids_sorted, columns=list(range(1, DAYS + 1)))

    stock_matrix = init_series.to_numpy()[:, None] + cg_cum.to_numpy() - lg_cum.to_numpy()
    stock_matrix = np.where(np.abs(stock_matrix) < 1e-9, 0.0, stock_matrix)

    lg_stock_levels = (
        pd.DataFrame(stock_matrix, index=lg_ids_sorted, columns=list(range(1, DAYS + 1)))
          .stack().rename("Stock_Level_tons")
          .rename_axis(index=["LG_ID", "Day"]).reset_index()
          .rename(columns={"LG_ID": "Entity_ID"})
          .assign(Entity_Type="LG")[["Day", "Entity_Type", "Entity_ID", "Stock_Level_tons"]]
    )

    pre_cols = list(range(start_day, 1))
    if pre_cols:
        if not dispatch_cg.empty:
            cg_pre = cg_piv.reindex(index=lg_ids_sorted, columns=pre_cols, fill_value=0.0)
            cg_pre_cum = cg_pre.cumsum(axis=1)
        else:
            cg_pre_cum = pd.DataFrame(0.0, index=lg_ids_sorted, columns=pre_cols)

        stock_pre_matrix = init_series.to_numpy()[:, None] + cg_pre_cum.to_numpy()
        stock_pre_matrix = np.where(np.abs(stock_pre_matrix) < 1e-9, 0.0, stock_pre_matrix)

        lg_stock_levels_pre = (
            pd.DataFrame(stock_pre_matrix, index=lg_ids_sorted, columns=pre_cols)
              .stack().rename("Stock_Level_tons")
              .rename_axis(index=["LG_ID", "Day"]).reset_index()
              .rename(columns={"LG_ID": "Entity_ID"})
              .assign(Entity_Type="LG")[["Day", "Entity_Type", "Entity_ID", "Stock_Level_tons"]]
        )

    lg_stock_levels = pd.concat([lg_stock_levels_pre, lg_stock_levels], ignore_index=True)
    # Add Date column for stock_levels using day_to_date (if day exists)
    lg_stock_levels["Date"] = lg_stock_levels["Day"].apply(lambda d: day_to_date.get(int(d), pd.NaT))

    # ensure stock_levels (FPS rows) also have Date (we already added above)
    if "Date" not in stock_levels.columns:
        stock_levels["Date"] = stock_levels["Day"].apply(lambda d: day_to_date.get(int(d), pd.NaT))

    stock_levels = pd.concat(
        [stock_levels[stock_levels["Entity_Type"] == "FPS"], lg_stock_levels[["Day","Date","Entity_Type","Entity_ID","Stock_Level_tons"]].rename(columns={"Entity_ID":"Entity_ID"})],
        ignore_index=True
    )

    # Ensure dispatch_lg and dispatch_cg contain the category columns and Date column
    return dispatch_cg, dispatch_lg, stock_levels
