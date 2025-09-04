
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta

st.set_page_config(page_title="CB Ops — ISO Certification Body SaaS (Demo)", layout="wide")

@st.cache_data
def load_data():
    auditors = pd.read_csv("auditors.csv")
    clients = pd.read_csv("clients.csv")
    quotes = pd.read_csv("quotes.csv", parse_dates=["target_window_start","target_window_end"])
    schedule = pd.read_csv("schedule.csv", parse_dates=["start_date","end_date"])
    certs = pd.read_csv("certificates.csv", parse_dates=["issue_date","expiry_date"])
    invoices = pd.read_csv("invoices.csv", parse_dates=["date_issued","due_date"])
    return auditors, clients, quotes, schedule, certs, invoices

def save_df(name, df):
    df.to_csv(name, index=False)
    st.success(f"Saved {name}")

auditors, clients, quotes, schedule, certs, invoices = load_data()

def overlapping(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or a_start > b_end)

def available_auditors(std, iaf, start, days):
    end = start + timedelta(days=days-1)
    eligible = auditors[auditors["standards"].str.contains(std) & auditors["iaf_codes"].str.contains(str(iaf))]
    out = []
    for _, row in eligible.iterrows():
        aid = row["auditor_id"]
        s = schedule[schedule["auditor_id"]==aid]
        conflict = False
        for _, ev in s.iterrows():
            if overlapping(ev["start_date"].date(), ev["end_date"].date(), start, end):
                conflict = True; break
        if not conflict:
            out.append(row)
    return pd.DataFrame(out)

def utilization(auditor_id, month_start):
    month_end = (month_start + timedelta(days=40)).replace(day=1) - timedelta(days=1)
    s = schedule[(schedule["auditor_id"]==auditor_id) & (schedule["start_date"]<=month_end) & (schedule["end_date"]>=month_start)]
    days = 0
    for _, ev in s.iterrows():
        d1 = max(ev["start_date"].date(), month_start)
        d2 = min(ev["end_date"].date(), month_end)
        days += max(0,(d2 - d1).days + 1)
    cap = auditors.set_index("auditor_id").loc[auditor_id, "capacity_days_month"]
    return days, cap, days/max(cap,1)

st.title("CB Ops — ISO Certification Body SaaS (Demo)")
nav = st.sidebar.radio("Go to", ["Dashboard","Quotes","Scheduling","Auditors","Certificates","Invoicing","Reports"])

if nav=="Dashboard":
    today = date.today()
    next_90 = today + timedelta(days=90)
    open_quotes = quotes[quotes["status"].isin(["Draft","Approved"])].shape[0]
    booked_days = schedule[(schedule["start_date"]>=pd.Timestamp(today)) & (schedule["start_date"]<=pd.Timestamp(next_90))]["days"].sum()
    overdue = invoices[(invoices["status"]=="Unpaid") & (invoices["due_date"]<pd.Timestamp(today))].shape[0]
    active_certs = certs[certs["status"]=="Active"].shape[0]

    c = st.columns(4)
    c[0].metric("Open Quotes", int(open_quotes))
    c[1].metric("Booked Audit-Days (90d)", int(booked_days))
    c[2].metric("Overdue Invoices", int(overdue))
    c[3].metric("Active Certificates", int(active_certs))

    st.markdown("### Auditor Utilization (This Month)")
    month_start = today.replace(day=1)
    util_rows = []
    for _, a in auditors.iterrows():
        used, cap, pct = utilization(a["auditor_id"], month_start)
        util_rows.append({"auditor": a["name"], "used_days": used, "capacity": cap, "utilization": pct})
    util_df = pd.DataFrame(util_rows).sort_values("utilization", ascending=False)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(util_df["auditor"].head(12), util_df["utilization"].head(12))
    ax.set_ylabel("Utilization"); ax.set_xticklabels(util_df["auditor"].head(12), rotation=45, ha="right")
    st.pyplot(fig)

    st.markdown("### Pipeline by Standard (Quotes)")
    pip = quotes.groupby(["standard","status"]).size().unstack(fill_value=0)
    st.dataframe(pip)

elif nav=="Quotes":
    st.subheader("Quote Repository")
    st.dataframe(quotes.sort_values("target_window_start")[["quote_id","client_name","standard","iaf_code","stage","duration_days","target_window_start","target_window_end","status","amount_usd"]])

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
                "target_window_start": pd.Timestamp(start),
                "target_window_end": pd.Timestamp(end),
                "status": status, "amount_usd": float(amount)
            }])
            quotes2 = pd.concat([quotes, new], ignore_index=True)
            quotes2.to_csv("quotes.csv", index=False)
            st.success("Saved quotes.csv")
            st.experimental_rerun()

elif nav=="Scheduling":
    st.subheader("Scheduling & Assignment")
    pending = quotes[quotes["status"]=="Approved"].merge(schedule[["quote_id"]], on="quote_id", how="left", indicator=True)
    pending = pending[pending["_merge"]=="left_only"]
    st.caption(f"Unscheduled approved quotes: {pending.shape[0]}")
    st.dataframe(pending[["quote_id","client_name","standard","iaf_code","stage","duration_days","target_window_start","target_window_end","amount_usd"]].head(30))

    def available_auditors(std, iaf, start, days):
        end = start + timedelta(days=days-1)
        eligible = auditors[auditors["standards"].str.contains(std) & auditors["iaf_codes"].str.contains(str(iaf))]
        out = []
        for _, row in eligible.iterrows():
            aid = row["auditor_id"]
            s = schedule[schedule["auditor_id"]==aid]
            conflict = False
            for _, ev in s.iterrows():
                if not (ev["end_date"].date() < start or ev["start_date"].date() > end):
                    conflict = True; break
            if not conflict:
                out.append(row)
        return pd.DataFrame(out)

    with st.expander("Create Schedule from Quote"):
        qid = st.selectbox("Quote", pending["quote_id"].tolist())
        q = quotes[quotes["quote_id"]==qid].iloc[0]
        start = st.date_input("Start Date", value=q["target_window_start"].date())
        days = st.number_input("Audit Days", 1, 30, int(q["duration_days"]))
        st.write("Suggested auditors (matching standard & IAF, conflict-free):")
        avail = available_auditors(q["standard"], q["iaf_code"], start, int(days))
        st.dataframe(avail[["auditor_id","name","grade","capacity_days_month"]])

        auditor_id = st.selectbox("Assign Auditor", ["Auto-select"] + avail["auditor_id"].tolist())
        if st.button("Create Schedule Entry"):
            if auditor_id=="Auto-select" and not avail.empty:
                auditor_id = avail.sample(1)["auditor_id"].iloc[0]
            elif auditor_id=="Auto-select":
                auditor_id = None
            new = pd.DataFrame([{
                "event_id": f"E{schedule.shape[0]+1:05d}",
                "quote_id": q["quote_id"], "client_id": q["client_id"], "client_name": q["client_name"],
                "standard": q["standard"], "iaf_code": q["iaf_code"],
                "start_date": pd.Timestamp(start), "end_date": pd.Timestamp(start) + pd.Timedelta(days=int(days)-1),
                "days": int(days), "auditor_id": auditor_id, "status": "Assigned" if auditor_id else "Unassigned",
                "stage": q["stage"]
            }])
            sched2 = pd.concat([schedule, new], ignore_index=True)
            sched2.to_csv("schedule.csv", index=False)
            st.success("Saved schedule.csv")
            st.experimental_rerun()

    st.markdown("### Calendar — Next 60 Days (count of scheduled audit-days per day)")
    today = date.today(); horizon = today + timedelta(days=60)
    rng = pd.date_range(today, horizon, freq="D")
    day_counts = pd.Series(0, index=rng)
    for _, ev in schedule.iterrows():
        d1 = max(ev["start_date"].date(), today)
        d2 = min(ev["end_date"].date(), horizon)
        if d1<=d2:
            for d in pd.date_range(d1, d2, freq="D"):
                day_counts.loc[pd.Timestamp(d)] += 1
    fig, ax = plt.subplots()
    ax.plot(day_counts.index, day_counts.values)
    ax.set_ylabel("Scheduled Audit-Days")
    st.pyplot(fig)

elif nav=="Auditors":
    st.subheader("Auditor Directory")
    st.dataframe(auditors[["auditor_id","name","country","grade","standards","iaf_codes","capacity_days_month"]])

    with st.expander("Check Auditor Availability"):
        aid = st.selectbox("Auditor", auditors["auditor_id"])
        mn = st.date_input("Month", value=date.today().replace(day=1))
        used, cap, pct = 0, auditors.set_index("auditor_id").loc[aid, "capacity_days_month"], 0.0
        s = schedule[schedule["auditor_id"]==aid]
        month_start = mn
        month_end = (month_start + timedelta(days=40)).replace(day=1) - timedelta(days=1)
        for _, ev in s.iterrows():
            d1 = max(ev["start_date"].date(), month_start)
            d2 = min(ev["end_date"].date(), month_end)
            if d1<=d2: used += (d2-d1).days + 1
        pct = used/max(cap,1)
        st.metric("Booked Days", used); st.metric("Capacity (days)", cap); st.metric("Utilization", f"{pct:.0%}")

elif nav=="Certificates":
    st.subheader("Certificates")
    st.dataframe(certs[["cert_id","client_name","standard","issue_date","expiry_date","status"]])
    with st.expander("Issue New Certificate"):
        client = st.selectbox("Client", clients["name"])
        std = st.selectbox("Standard", sorted(clients["standard"].unique().tolist()))
        issue = st.date_input("Issue Date", value=date.today())
        expiry = st.date_input("Expiry Date", value=date.today() + timedelta(days=3*365))
        status = st.selectbox("Status", ["Active","Suspended","Withdrawn","Expired"])
        if st.button("Issue Certificate"):
            new = pd.DataFrame([{
                "cert_id": f"CB-{clients[clients['name']==client].iloc[0]['client_id']}-{std.split()[-1]}-{np.random.randint(1000,9999)}",
                "client_id": clients[clients['name']==client].iloc[0]['client_id'],
                "client_name": client, "standard": std,
                "issue_date": pd.Timestamp(issue), "expiry_date": pd.Timestamp(expiry),
                "status": status
            }])
            certs2 = pd.concat([certs, new], ignore_index=True)
            certs2.to_csv("certificates.csv", index=False)
            st.success("Saved certificates.csv")
            st.experimental_rerun()

elif nav=="Invoicing":
    st.subheader("Invoices")
    st.dataframe(invoices.sort_values("date_issued", ascending=False)[["invoice_id","client_name","amount_usd","date_issued","due_date","status","description"]])
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
                "amount_usd": amt, "date_issued": pd.Timestamp(issued),
                "due_date": pd.Timestamp(due), "status": status, "description": desc
            }])
            inv2 = pd.concat([invoices, new], ignore_index=True)
            inv2.to_csv("invoices.csv", index=False)
            st.success("Saved invoices.csv")
            st.experimental_rerun()

else:
    st.subheader("Coverage & Performance")
    st.markdown("**IAF Code Coverage (Auditor Qualifications)**")
    aud_iaf = auditors.copy()
    rows = []
    for _, r in aud_iaf.iterrows():
        for code in str(r["iaf_codes"]).split(","):
            rows.append({"auditor": r["name"], "iaf_code": code.strip()})
    iaf_df = pd.DataFrame(rows)
    cov = iaf_df.groupby("iaf_code").size().sort_index()
    fig, ax = plt.subplots()
    ax.bar(cov.index, cov.values)
    ax.set_ylabel("Auditors Qualified")
    st.pyplot(fig)

    st.markdown("**Standards Coverage (Auditor Qualifications)**")
    aud_std = auditors.copy()
    rows2 = []
    for _, r in aud_std.iterrows():
        for s in str(r["standards"]).split(","):
            rows2.append({"auditor": r["name"], "standard": s.strip()})
    std_df = pd.DataFrame(rows2)
    cov2 = std_df.groupby("standard").size()
    fig2, ax2 = plt.subplots()
    ax2.bar(cov2.index, cov2.values)
    ax2.set_ylabel("Auditors Qualified")
    st.pyplot(fig2)

    st.markdown("**Revenue Pipeline (Quotes by Status)**")
    rev = quotes.groupby("status")["amount_usd"].sum().reindex(["Draft","Approved","Rejected","Expired"], fill_value=0)
    fig3, ax3 = plt.subplots()
    ax3.bar(rev.index, rev.values)
    ax3.set_ylabel("Amount (USD)")
    st.pyplot(fig3)
