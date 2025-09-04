
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import os, random

st.set_page_config(page_title="CB Ops — ISO Certification Body SaaS (Demo)", layout="wide")

DATA_FILES = ["auditors.csv","clients.csv","quotes.csv","schedule.csv","certificates.csv","invoices.csv"]

# ------------------------- 1) DATA BOOTSTRAP -------------------------
def ensure_data():
    # If all files exist, do nothing
    if all(os.path.exists(f) for f in DATA_FILES):
        return

    np.random.seed(42); random.seed(42)
    standards = ["ISO 9001","ISO 14001","ISO 45001","ISO 27001","ISO 13485","ISO 22000"]
    iaf_codes = ["01","03","07","09","12","14","17","18","19","21","22","23","24","29","30","31","33","34","35","37","39"]
    countries = ["US","CA","MX","UK","DE","FR","NL","ES","IT","PL","SE","NO","DK","FI","IE","PT","CZ","RO","HU","TR","AE","IN","SG","AU","NZ","JP","KR","CN","BR","AR"]

    # Auditors
    auditors = []
    for i in range(24):
        a_id = f"A{i+1:03d}"
        name = random.choice(
            ["Alex","Jordan","Taylor","Morgan","Casey","Riley","Sam","Jamie","Cameron","Drew",
             "Avery","Harper","Rowan","Elliot","Quinn","Reese","Parker","Logan","Devon","Kendall",
             "Skyler","Shawn","Cody","Sydney"]
        ) + " " + random.choice(
            ["Smith","Brown","Lee","Garcia","Miller","Wilson","Moore","Taylor","Anderson","Thomas",
             "Jackson","White","Harris","Martin","Thompson"]
        )
        country = random.choice(countries)
        stnds = ",".join(sorted(random.sample(standards, k=random.randint(2,4))))
        iafs = ",".join(sorted(random.sample(iaf_codes, k=random.randint(4,8))))
        grade = random.choice(["Lead","Principal","Associate","Technical Expert"])
        capacity = random.choice([8,10,12,14,16,18])
        auditors.append({
            "auditor_id": a_id, "name": name, "country": country, "grade": grade,
            "standards": stnds, "iaf_codes": iafs, "capacity_days_month": capacity
        })
    pd.DataFrame(auditors).to_csv("auditors.csv", index=False)

    # Clients
    clients = []
    for i in range(80):
        c_id = f"C{i+1:03d}"
        name = random.choice(
            ["Acme","Globex","Initech","Umbrella","Hooli","Vehement","Stark","Wayne","Wonka","Aperture",
             "Gringotts","Cyberdyne","Tyrell","Soylent","Monarch","Pied Piper","Massive Dynamic","Blue Sun",
             "Black Mesa","Dunder Mifflin"]
        ) + " " + random.choice(
            ["Manufacturing","Logistics","Foods","Tech","Pharma","Medical","Aerospace","Automotive",
             "Energy","Services","Plastics","Packaging"]
        )
        country = random.choice(countries)
        std = random.choice(standards)
        iaf = random.choice(iaf_codes)
        sites = random.randint(1,4)
        employees = random.choice(["<50","50-200","200-500","500-1000",">1000"])
        risk = random.choice(["Low","Medium","High"])
        clients.append({
            "client_id": c_id, "name": name, "country": country, "standard": std,
            "iaf_code": iaf, "sites": sites, "employees_band": employees, "risk_level": risk
        })
    clients_df = pd.DataFrame(clients)
    clients_df.to_csv("clients.csv", index=False)

    # Quotes
    today = date.today()
    quotes = []
    for i in range(120):
        q_id = f"Q{i+1:04d}"
        client = clients_df.sample(1).iloc[0]
        stage = random.choice(
            ["Initial - Stage 1","Initial - Stage 2","Surveillance 1","Surveillance 2","Recertification"]
        )
        base_days = {"Initial - Stage 1":1, "Initial - Stage 2":2, "Surveillance 1":1, "Surveillance 2":1, "Recertification":2}[stage]
        complexity = {"<50":0.0, "50-200":0.2, "200-500":0.5, "500-1000":0.8, ">1000":1.2}[client["employees_band"]]
        duration = max(1, int(round(base_days + complexity + random.choice([0,0,0,1]))))
        travel = random.choice([0,0,1])
        start = pd.Timestamp(today + timedelta(days=random.randint(7,120))).normalize()
        end = (start + pd.Timedelta(days=30)).normalize()
        status = random.choices(["Draft","Approved","Rejected","Expired"], weights=[35,45,10,10])[0]
        amount = 1500*duration + 600*travel
        quotes.append({
            "quote_id": q_id, "client_id": client["client_id"], "client_name": client["name"],
            "country": client["country"], "standard": client["standard"], "iaf_code": client["iaf_code"],
            "stage": stage, "duration_days": duration, "travel_days": travel,
            "target_window_start": start, "target_window_end": end, "status": status, "amount_usd": amount
        })
    quotes_df = pd.DataFrame(quotes)
    quotes_df.to_csv("quotes.csv", index=False)

    # Schedule
    schedule = []
    def random_free_auditor(std, iaf, start, days):
        aud = pd.read_csv("auditors.csv")
        eligible = aud[aud["standards"].str.contains(std) & aud["iaf_codes"].str.contains(iaf)]
        if eligible.empty: return None
        for _ in range(50):
            row = eligible.sample(1).iloc[0]
            aid = row["auditor_id"]
            conflict = False
            for ev in schedule:
                if ev["auditor_id"]==aid:
                    if not (start + timedelta(days=days-1) < ev["start_date"] or start > ev["end_date"]):
                        conflict = True; break
            if not conflict: return aid
        return None

    for _, q in quotes_df.iterrows():
        if q["status"]!="Approved": continue
        start = (pd.Timestamp(q["target_window_start"]).normalize() + pd.Timedelta(days=random.randint(0,10))).date()
        days = int(q["duration_days"])
        aid = random_free_auditor(q["standard"], q["iaf_code"], start, days)
        assigned = aid is not None and random.random()>0.2
        schedule.append({
            "event_id": f"E{len(schedule)+1:05d}",
            "quote_id": q["quote_id"], "client_id": q["client_id"], "client_name": q["client_name"],
            "standard": q["standard"], "iaf_code": q["iaf_code"],
            "start_date": start, "end_date": start + timedelta(days=days-1),
            "days": days, "auditor_id": aid if assigned else None,
            "status": "Assigned" if assigned else "Unassigned", "stage": q["stage"]
        })
    schedule_df = pd.DataFrame(schedule)
    schedule_df["start_date"] = pd.to_datetime(schedule_df["start_date"]).dt.normalize()
    schedule_df["end_date"] = pd.to_datetime(schedule_df["end_date"]).dt.normalize()
    schedule_df.to_csv("schedule.csv", index=False)

    # Certificates
    certs = []
    for _, q in quotes_df.iterrows():
        if q["status"]=="Approved" and random.random()>0.7:
            issued = (pd.Timestamp(today - timedelta(days=random.randint(10,365)))).normalize()
            expiry = (issued + pd.Timedelta(days=3*365)).normalize()
            certs.append({
                "cert_id": f"CB-{q['client_id']}-{q['standard'].split()[-1]}-{random.randint(1000,9999)}",
                "client_id": q["client_id"], "client_name": q["client_name"], "standard": q["standard"],
                "issue_date": issued, "expiry_date": expiry,
                "status": random.choices(["Active","Suspended","Withdrawn","Expired"], weights=[70,10,5,15])[0]
            })
    pd.DataFrame(certs).to_csv("certificates.csv", index=False)

    # Invoices
    invoices = []
    sched_df = pd.read_csv("schedule.csv", parse_dates=["start_date","end_date"])
    for _, s in sched_df.iterrows():
        if random.random()<0.85:
            amt = quotes_df.loc[quotes_df["quote_id"]==s["quote_id"], "amount_usd"].values[0]
            issued = (s["start_date"] - pd.Timedelta(days=14)).normalize()
            due = (issued + pd.Timedelta(days=30)).normalize()
            status = random.choices(["Unpaid","Paid","Overdue"], weights=[40,55,5])[0]
            invoices.append({
                "invoice_id": f"INV{len(invoices)+1:05d}", "quote_id": s["quote_id"],
                "client_id": s["client_id"], "client_name": s["client_name"], "amount_usd": amt,
                "date_issued": issued, "due_date": due, "status": status, "description": f"{s['stage']} audit fees"
            })
    pd.DataFrame(invoices).to_csv("invoices.csv", index=False)

ensure_data()

# ------------------------- 2) LOAD + NORMALIZE -------------------------
@st.cache_data
def load_data():
    auditors = pd.read_csv("auditors.csv")
    clients = pd.read_csv("clients.csv")
    quotes = pd.read_csv("quotes.csv", parse_dates=["target_window_start","target_window_end"])
    schedule = pd.read_csv("schedule.csv", parse_dates=["start_date","end_date"])
    certs = pd.read_csv("certificates.csv", parse_dates=["issue_date","expiry_date"])
    invoices = pd.read_csv("invoices.csv", parse_dates=["date_issued","due_date"])

    for c in ["start_date","end_date"]:
        schedule[c] = pd.to_datetime(schedule[c]).dt.normalize()
    for c in ["date_issued","due_date"]:
        invoices[c] = pd.to_datetime(invoices[c]).dt.normalize()
    for c in ["issue_date","expiry_date"]:
        certs[c] = pd.to_datetime(certs[c]).dt.normalize()
    for c in ["target_window_start","target_window_end"]:
        quotes[c] = pd.to_datetime(quotes[c]).dt.normalize()
    return auditors, clients, quotes, schedule, certs, invoices

def save_df(name, df):
    if name.endswith("schedule.csv"):
        for c in ["start_date","end_date"]:
            df[c] = pd.to_datetime(df[c]).dt.normalize()
    if name.endswith("invoices.csv"):
        for c in ["date_issued","due_date"]:
            df[c] = pd.to_datetime(df[c]).dt.normalize()
    if name.endswith("certificates.csv"):
        for c in ["issue_date","expiry_date"]:
            df[c] = pd.to_datetime(df[c]).dt.normalize()
    if name.endswith("quotes.csv"):
        for c in ["target_window_start","target_window_end"]:
            df[c] = pd.to_datetime(df[c]).dt.normalize()
    df.to_csv(name, index=False)
    st.success(f"Saved {name}")

auditors, clients, quotes, schedule, certs, invoices = load_data()

# ------------------------- 3) HELPERS -------------------------
def overlapping(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or a_start > b_end)

def month_bounds(d: date):
    start_ts = pd.Timestamp(d).normalize().replace(day=1)
    end_ts = (start_ts + pd.offsets.MonthEnd(0))
    return start_ts, end_ts

def available_auditors(std, iaf, start_date, days):
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = start_ts + pd.Timedelta(days=days-1)
    eligible = auditors[
        auditors["standards"].astype(str).str.contains(str(std), na=False) &
        auditors["iaf_codes"].astype(str).str.contains(str(iaf), na=False)
    ]
    out = []
    for _, row in eligible.iterrows():
        aid = row["auditor_id"]
        s = schedule[schedule["auditor_id"]==aid]
        conflict = False
        for _, ev in s.iterrows():
            if overlapping(ev["start_date"], ev["end_date"], start_ts, end_ts):
                conflict = True; break
        if not conflict:
            out.append(row)
    return pd.DataFrame(out)

def utilization(auditor_id, month_start_date):
    m_start, m_end = month_bounds(month_start_date)
    s = schedule[
        (schedule["auditor_id"]==auditor_id) &
        (schedule["start_date"]<=m_end) &
        (schedule["end_date"]>=m_start)
    ]
    days_used = 0
    for _, ev in s.iterrows():
        d1 = max(ev["start_date"], m_start)
        d2 = min(ev["end_date"], m_end)
        if d1 <= d2:
            days_used += int((d2 - d1).days) + 1
    cap = auditors.set_index("auditor_id").loc[auditor_id, "capacity_days_month"]
    return days_used, cap, days_used/max(cap,1)

# ------------------------- 4) UI -------------------------
st.title("CB Ops — ISO Certification Body SaaS (Demo)")
nav = st.sidebar.radio("Go to", ["Dashboard","Quotes","Scheduling","Auditors","Certificates","Invoicing","Reports"])

# Dashboard
if nav=="Dashboard":
    today = pd.Timestamp(date.today()).normalize()
    next_90 = today + pd.Timedelta(days=90)
    open_quotes = quotes[quotes["status"].isin(["Draft","Approved"])].shape[0]
    booked_days = schedule[(schedule["start_date"]>=today) & (schedule["start_date"]<=next_90)]["days"].sum()
    overdue = invoices[(invoices["status"]=="Unpaid") & (invoices["due_date"]<today)].shape[0]
    active_certs = certs[certs["status"]=="Active"].shape[0]

    c = st.columns(4)
    c[0].metric("Open Quotes", int(open_quotes))
    c[1].metric("Booked Audit-Days (90d)", int(booked_days))
    c[2].metric("Overdue Invoices", int(overdue))
    c[3].metric("Active Certificates", int(active_certs))

    st.markdown("### Auditor Utilization (This Month)")
    m_start, _ = month_bounds(date.today())
    util_rows = []
    for _, a in auditors.iterrows():
        used, cap, pct = utilization(a["auditor_id"], m_start.to_pydatetime().date())
        util_rows.append({"auditor": a["name"], "used_days": used, "capacity": cap, "utilization": pct})
    util_df = pd.DataFrame(util_rows).sort_values("utilization", ascending=False)

    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(util_df["auditor"].head(12), util_df["utilization"].head(12))
    ax.set_ylabel("Utilization")
    ax.set_xticklabels(util_df["auditor"].head(12), rotation=45, ha="right")
    st.pyplot(fig)

    st.markdown("### Pipeline by Standard (Quotes)")
    pip = quotes.groupby(["standard","status"]).size().unstack(fill_value=0)
    st.dataframe(pip, use_container_width=True)

# Quotes
elif nav=="Quotes":
    st.subheader("Quote Repository")
    st.dataframe(
        quotes.sort_values("target_window_start")[
            ["quote_id","client_name","standard","iaf_code","stage","duration_days",
             "target_window_start","target_window_end","status","amount_usd"]
        ],
        use_container_width=True
    )

    with st.expander("Create New Quote"):
        client = st.selectbox("Client", clients["name"])
        c_row = clients[clients["name"]==client].iloc[0]
        std = st.selectbox("Standard", sorted(clients[clients["name"]==client]["standard"].unique().tolist()))
        iaf = st.text_input("IAF Code", value=str(c_row["iaf_code"]))
        stage = st.selectbox("Stage", ["Initial - Stage 1","Initial - Stage 2","Surveillance 1","Surveillance 2","Recertification"])
        duration = st.number_input("Audit Duration (days)", 1, 30, 2)
        travel = st.number_input("Travel Days", 0, 10, 0)
        start = st.date_input("Target Window Start", value=date.today() + timedelta(days=21))
        end = st.date_input("Target Window End", value=date.today() + timedelta(days=60))
        amount = st.number_input("Quoted Amount (USD)", 0, 100000, 3000, 100)
        status = st.selectbox("Status", ["Draft","Approved","Rejected","Expired"])
        if st.button("Add Quote"):
            new = pd.DataFrame([{
                "quote_id": f"Q{quotes.shape[0]+1:04d}",
                "client_id": c_row["client_id"],
                "client_name": client,
                "country": c_row["country"],
                "standard": std, "iaf_code": iaf,
                "stage": stage, "duration_days": int(duration),
                "travel_days": int(travel),
                "target_window_start": pd.Timestamp(start).normalize(),
                "target_window_end": pd.Timestamp(end).normalize(),
                "status": status, "amount_usd": float(amount)
            }])
            quotes2 = pd.concat([quotes, new], ignore_index=True)
            save_df("quotes.csv", quotes2)
            st.experimental_rerun()

# Scheduling
elif nav=="Scheduling":
    st.subheader("Scheduling & Assignment")
    pending = quotes[quotes["status"]=="Approved"].merge(schedule[["quote_id"]], on="quote_id", how="left", indicator=True)
    pending = pending[pending["_merge"]=="left_only"]
    st.caption(f"Unscheduled approved quotes: {pending.shape[0]}")
    st.dataframe(
        pending[["quote_id","client_name","standard","iaf_code","stage","duration_days",
                 "target_window_start","target_window_end","amount_usd"]].head(30),
        use_container_width=True
    )

    with st.expander("Create Schedule from Quote"):
        qid = st.selectbox("Quote", pending["quote_id"].tolist() if not pending.empty else [])
        if qid:
            q = quotes[quotes["quote_id"]==qid].iloc[0]
            start = st.date_input("Start Date", value=q["target_window_start"].date())
            days = st.number_input("Audit Days", 1, 30, int(q["duration_days"]))
            st.write("Suggested auditors (matching standard & IAF, conflict-free):")
            avail = available_auditors(q["standard"], q["iaf_code"], start, int(days))
            if not avail.empty:
                st.dataframe(avail[["auditor_id","name","grade","capacity_days_month"]], use_container_width=True)
            else:
                st.info("No conflict-free eligible auditors for the selected window.")
            auditor_id = st.selectbox("Assign Auditor", ["Auto-select"] + (avail["auditor_id"].tolist() if not avail.empty else []))
            if st.button("Create Schedule Entry"):
                if auditor_id=="Auto-select" and not avail.empty:
                    auditor_id = avail.sample(1)["auditor_id"].iloc[0]
                elif auditor_id=="Auto-select":
                    auditor_id = None
                start_ts = pd.Timestamp(start).normalize()
                new = pd.DataFrame([{
                    "event_id": f"E{schedule.shape[0]+1:05d}",
                    "quote_id": q["quote_id"], "client_id": q["client_id"], "client_name": q["client_name"],
                    "standard": q["standard"], "iaf_code": q["iaf_code"],
                    "start_date": start_ts, "end_date": start_ts + pd.Timedelta(days=int(days)-1),
                    "days": int(days), "auditor_id": auditor_id, "status": "Assigned" if auditor_id else "Unassigned",
                    "stage": q["stage"]
                }])
                schedule2 = pd.concat([schedule, new], ignore_index=True)
                save_df("schedule.csv", schedule2)
                st.experimental_rerun()

    st.markdown("### Calendar — Next 60 Days (count of scheduled audit-days per day)")
    today_ts = pd.Timestamp(date.today()).normalize()
    horizon = today_ts + pd.Timedelta(days=60)
    rng = pd.date_range(today_ts, horizon, freq="D")
    day_counts = pd.Series(0, index=rng)
    for _, ev in schedule.iterrows():
        d1 = max(ev["start_date"], today_ts)
        d2 = min(ev["end_date"], horizon)
        if d1<=d2:
            day_counts.loc[d1:d2] += 1
    fig, ax = plt.subplots()
    ax.plot(day_counts.index, day_counts.values)
    ax.set_ylabel("Scheduled Audit-Days")
    st.pyplot(fig)

# Auditors
elif nav=="Auditors":
    st.subheader("Auditor Directory")
    st.dataframe(
        auditors[["auditor_id","name","country","grade","standards","iaf_codes","capacity_days_month"]],
        use_container_width=True
    )

    with st.expander("Check Auditor Availability"):
        aid = st.selectbox("Auditor", auditors["auditor_id"])
        mn = st.date_input("Month", value=date.today().replace(day=1))
        m_start = pd.Timestamp(mn).normalize()
        m_end = (m_start + pd.offsets.MonthEnd(0))
        s = schedule[schedule["auditor_id"]==aid]
        used = 0
        for _, ev in s.iterrows():
            d1 = max(ev["start_date"], m_start)
            d2 = min(ev["end_date"], m_end)
            if d1<=d2: used += int((d2-d1).days) + 1
        cap = auditors.set_index("auditor_id").loc[aid, "capacity_days_month"]
        pct = used/max(cap,1)
        c = st.columns(3)
        c[0].metric("Booked Days", used)
        c[1].metric("Capacity (days)", cap)
        c[2].metric("Utilization", f"{pct:.0%}")

# Certificates
elif nav=="Certificates":
    st.subheader("Certificates")
    st.dataframe(certs[["cert_id","client_name","standard","issue_date","expiry_date","status"]], use_container_width=True)
    with st.expander("Issue New Certificate"):
        client = st.selectbox("Client", clients["name"])
        std = st.selectbox("Standard", sorted(clients["standard"].unique().tolist()))
        issue = st.date_input("Issue Date", value=date.today())
        expiry = st.date_input("Expiry Date", value=date.today() + timedelta(days=3*365))
        status = st.selectbox("Status", ["Active","Suspended","Withdrawn","Expired"])
        if st.button("Issue Certificate"):
            new = pd.DataFrame([{
                "cert_id": f"CB-{clients[clients['name']==client].iloc[0]['client_id']}-{std.split()[-1]}-{np.random.randint(1000,9999)}",
                "client_id": clients[clients['name']==client].iloc[0]['client_id"],
                "client_name": client, "standard": std,
                "issue_date": pd.Timestamp(issue).normalize(), "expiry_date": pd.Timestamp(expiry).normalize(),
                "status": status
            }])
            certs2 = pd.concat([certs, new], ignore_index=True)
            save_df("certificates.csv", certs2)
            st.experimental_rerun()

# Invoicing
elif nav=="Invoicing":
    st.subheader("Invoices")
    st.dataframe(
        invoices.sort_values("date_issued", ascending=False)[
            ["invoice_id","client_name","amount_usd","date_issued","due_date","status","description"]
        ],
        use_container_width=True
    )
    with st.expander("Create Invoice from Schedule"):
        ev = st.selectbox("Schedule Event", schedule["event_id"])
        srow = schedule[schedule["event_id"]==ev].iloc[0]
        amt = quotes.set_index("quote_id").loc[srow["quote_id"], "amount_usd"]
        issued = st.date_input("Issue Date", value=date.today())
        due = st.date_input("Due Date", value=date.today()+timedelta(days=30))
        status = st.selectbox("Status", ["Unpaid","Paid","Overdue"])
        desc = st.text_input("Description", value=f"{srow['stage']} audit fees")
        if st.button("Create Invoice"):
            new = pd.DataFrame([{
                "invoice_id": f"INV{invoices.shape[0]+1:05d}", "quote_id": srow["quote_id"],
                "client_id": srow["client_id"], "client_name": srow["client_name"],
                "amount_usd": amt, "date_issued": pd.Timestamp(issued).normalize(),
                "due_date": pd.Timestamp(due).normalize(), "status": status, "description": desc
            }])
            inv2 = pd.concat([invoices, new], ignore_index=True)
            save_df("invoices.csv", inv2)
            st.experimental_rerun()

# Reports
else:
    st.subheader("Coverage & Performance")
    st.markdown("**IAF Code Coverage (Auditor Qualifications)**")
    rows = []
    for _, r in auditors.iterrows():
        for code in str(r["iaf_codes"]).split(","):
            rows.append({"auditor": r["name"], "iaf_code": code.strip()})
    iaf_df = pd.DataFrame(rows)
    cov = iaf_df.groupby("iaf_code").size().sort_index()
    fig, ax = plt.subplots()
    ax.bar(cov.index, cov.values); ax.set_ylabel("Auditors Qualified")
    st.pyplot(fig)

    st.markdown("**Standards Coverage (Auditor Qualifications)**")
    rows2 = []
    for _, r in auditors.iterrows():
        for s in str(r["standards"]).split(","):
            rows2.append({"auditor": r["name"], "standard": s.strip()})
    std_df = pd.DataFrame(rows2)
    cov2 = std_df.groupby("standard").size()
    fig2, ax2 = plt.subplots()
    ax2.bar(cov2.index, cov2.values); ax2.set_ylabel("Auditors Qualified")
    st.pyplot(fig2)

    st.markdown("**Revenue Pipeline (Quotes by Status)**")
    rev = quotes.groupby("status")["amount_usd"].sum().reindex(["Draft","Approved","Rejected","Expired"], fill_value=0)
    fig3, ax3 = plt.subplots()
    ax3.bar(rev.index, rev.values); ax3.set_ylabel("Amount (USD)")
    st.pyplot(fig3)
