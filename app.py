# app.py — Statement Analyzer
# Multi-provider credit card statement intelligence tool

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from parser import combine_files
from analyzer import (
    get_data_summary,
    get_top_13,
    get_recurring_charges,
    get_possible_subscriptions,
    get_yoy_changes,
    build_llm_summary,
)
from llm import get_ai_insights

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Statement Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { text-align: center; padding: 1.2rem 0 0.25rem; }
    .main-header h1 { font-size: 2rem; font-weight: 600; }
    .tagline {
        text-align: center; color: #6b7280;
        font-size: 0.9rem; margin-bottom: 0.5rem;
    }
    .privacy-badge {
        background: #f0fdf4; border: 1px solid #bbf7d0;
        border-radius: 8px; padding: 0.5rem 0.85rem;
        font-size: 0.8rem; color: #166534; margin-bottom: 0.75rem;
    }
    .data-quality-banner {
        border-radius: 8px; padding: 0.75rem 1rem;
        font-size: 0.85rem; margin-bottom: 1rem;
    }
    .stat-row {
        display: flex; gap: 12px; flex-wrap: wrap;
        margin-bottom: 1.25rem;
    }
    .stat-card {
        background: #f9fafb; border: 1px solid #e5e7eb;
        border-radius: 10px; padding: 0.75rem 1rem;
        flex: 1; min-width: 130px; text-align: center;
    }
    .stat-label { font-size: 0.75rem; color: #9ca3af; margin-bottom: 2px; }
    .stat-value { font-size: 1.3rem; font-weight: 600; color: #111827; }
    .increase-row { color: #dc2626; }
    .decrease-row { color: #16a34a; }
    .section-note {
        font-size: 0.8rem; color: #9ca3af;
        font-style: italic; margin-bottom: 0.5rem;
    }
    .footer {
        text-align: center; margin-top: 2rem;
        padding-top: 1rem; border-top: 1px solid #e5e7eb;
        color: #9ca3af; font-size: 0.78rem;
    }
    /* Streamlit table tweaks */
    [data-testid="stDataFrame"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for key in ["df", "summary", "top13", "recurring", "subscriptions", "yoy",
            "llm_summary_text", "ai_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header"><h1>💳 Statement Analyzer</h1></div>
<div class="tagline">
    Upload your credit card statements and uncover what your spending is really telling you.<br>
    <strong>Your statements never leave your session — processed in memory, never stored.</strong>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ AI Provider")
    st.markdown('<div class="section-note">Required only for the AI Insights tab</div>',
                unsafe_allow_html=True)

    provider = st.selectbox(
        "Provider",
        ["OpenAI (GPT-4o)", "Google Gemini", "Anthropic Claude"],
        label_visibility="collapsed",
    )
    provider_hints = {
        "OpenAI (GPT-4o)": "platform.openai.com",
        "Google Gemini": "aistudio.google.com",
        "Anthropic Claude": "console.anthropic.com",
    }
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your key here...",
        help=f"Get your key at {provider_hints[provider]}",
    )
    if api_key:
        st.markdown(
            '<div class="privacy-badge">🔒 Key used only this session. Never stored or shared.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📖 Tips")
    st.markdown("""
- Upload **1 year minimum** for recurring detection
- Upload **2+ years** to unlock Year-over-Year changes
- Supported: **PDF, CSV, XLS, XLSX, DOCX**
- Upload multiple files at once — one per month is fine
- Most banks offer CSV export in their online portal
""")
    st.markdown("---")
    st.markdown(
        '<div class="footer">Made with ❤️ for people who actually want to know where their money goes.</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Upload zone
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload Your Statements")

col_upload, col_tip = st.columns([2, 1])
with col_upload:
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "csv", "xls", "xlsx", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

with col_tip:
    st.info(
        "**Better results with more data**\n\n"
        "🟡 1 statement — basic insights only\n\n"
        "🟠 6 months — recurring detection\n\n"
        "🟢 12 months — full annual cost view\n\n"
        "🔵 24+ months — Year-over-Year unlocked"
    )

analyze_btn = st.button(
    "🔍 Analyze Statements",
    type="primary",
    use_container_width=False,
    disabled=not uploaded_files,
)

# ─────────────────────────────────────────────────────────────────────────────
# Run analysis
# ─────────────────────────────────────────────────────────────────────────────
if analyze_btn and uploaded_files:
    with st.spinner("Parsing files and running analysis..."):
        df, parse_warnings = combine_files(uploaded_files)
        # Show any file-level parse warnings so the user knows which files failed
        if parse_warnings:
            for w in parse_warnings:
                st.warning(w)

        if df.empty:
            st.error(
                "Could not extract any transactions from the uploaded files. "
                "Please check the file formats and try again."
            )
            st.stop()

        summary = get_data_summary(df)
        top13 = get_top_13(df)
        recurring = get_recurring_charges(df)
        subscriptions = get_possible_subscriptions(df)
        yoy = get_yoy_changes(df)
        llm_summary_text = build_llm_summary(df, summary, top13, recurring, subscriptions, yoy)

        # Persist to session
        st.session_state.df = df
        st.session_state.summary = summary
        st.session_state.top13 = top13
        st.session_state.recurring = recurring
        st.session_state.subscriptions = subscriptions
        st.session_state.yoy = yoy
        st.session_state.llm_summary_text = llm_summary_text
        st.session_state.ai_result = None  # reset on re-analyze
        st.session_state.parse_warnings = parse_warnings

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.df is not None:
    summary = st.session_state.summary
    df = st.session_state.df
    parse_warnings = st.session_state.get("parse_warnings", [])

    # Parse warnings
    for w in parse_warnings:
        st.warning(w)

    # Data quality banner
    months = summary["months_covered"]
    has_yoy = summary["has_yoy"]
    years = summary["years_covered"]

    if months < 6:
        quality_color = "#fef3c7"
        quality_border = "#f59e0b"
        quality_msg = (
            f"📊 **{months} month(s)** of data detected. "
            "Upload at least 6 months for recurring charge detection and 12+ for full annual cost analysis."
        )
    elif months < 12:
        quality_color = "#fff7ed"
        quality_border = "#f97316"
        quality_msg = (
            f"📊 **{months} months** of data detected ({', '.join(str(y) for y in years)}). "
            "Upload 12+ months to see true annual costs. Upload 2+ years to unlock Year-over-Year."
        )
    elif not has_yoy:
        quality_color = "#eff6ff"
        quality_border = "#3b82f6"
        quality_msg = (
            f"📊 **{months} months** of data detected. "
            "Great for annual analysis! Upload statements from another year to unlock Year-over-Year comparison."
        )
    else:
        quality_color = "#f0fdf4"
        quality_border = "#22c55e"
        quality_msg = (
            f"✅ **{months} months across {len(years)} years** — full analysis unlocked including Year-over-Year!"
        )

    st.markdown(
        f'<div class="data-quality-banner" style="background:{quality_color};border-left:4px solid {quality_border};">'
        f"{quality_msg}</div>",
        unsafe_allow_html=True,
    )

    # Summary stat cards
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-label">Total Spent</div>
                <div class="stat-value">${summary['total_spent']:,.0f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Transactions</div>
                <div class="stat-value">{summary['total_transactions']:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Date Range</div>
                <div class="stat-value" style="font-size:0.85rem;">{summary['date_range_start']}<br>→ {summary['date_range_end']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Months</div>
                <div class="stat-value">{summary['months_covered']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg/Month</div>
                <div class="stat-value">${summary['total_spent']/max(summary['months_covered'],1):,.0f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Top 13",
        "🔁 Recurring Charges",
        "📋 Possible Subscriptions",
        "📈 Year-over-Year",
        "🔍 AI Insights",
    ])

    # ── Tab 1: Top 13 ─────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### 💰 Top 13 Most Expensive Single Purchases")
        st.markdown(
            '<div class="section-note">Ranked by transaction amount. '
            "Charges marked 🔁 also appear as recurring charges.</div>",
            unsafe_allow_html=True,
        )

        top13 = st.session_state.top13
        if top13.empty:
            st.info("No transactions found.")
        else:
            display = top13.copy()
            display["merchant"] = display.apply(
                lambda r: f"🔁 {r['merchant']}" if r["is_recurring"] else r["merchant"],
                axis=1,
            )
            st.dataframe(
                display[["date_fmt", "merchant", "amount_fmt", "source_file"]].rename(columns={
                    "date_fmt": "Date",
                    "merchant": "Merchant",
                    "amount_fmt": "Amount",
                    "source_file": "Statement File",
                }),
                use_container_width=True,
                hide_index=False,
            )
            total_top13 = top13["amount"].sum()
            pct = (total_top13 / summary["total_spent"] * 100) if summary["total_spent"] > 0 else 0
            st.markdown(
                f"**Top 13 total: ${total_top13:,.2f}** — "
                f"that's **{pct:.1f}%** of all spending in this period."
            )

    # ── Tab 2: Recurring ──────────────────────────────────────────────────
    with tab2:
        st.markdown("#### 🔁 Recurring Charges — True Annual Cost")
        st.markdown(
            '<div class="section-note">'
            "These charges appear on a regular schedule. The annual cost column shows what you're "
            "actually paying per year — a number most people have never seen laid out clearly."
            "</div>",
            unsafe_allow_html=True,
        )

        recurring = st.session_state.recurring
        if months < 3:
            st.warning("Upload at least 3 months of statements to detect recurring charges.")
        elif recurring is None or recurring.empty:
            st.info("No recurring charges detected in the uploaded statements.")
        else:
            st.dataframe(
                recurring[["merchant", "frequency", "avg_charge_fmt",
                            "annual_cost_fmt", "occurrences",
                            "first_seen_fmt", "last_seen_fmt"]].rename(columns={
                    "merchant": "Merchant",
                    "frequency": "Frequency",
                    "avg_charge_fmt": "Avg Charge",
                    "annual_cost_fmt": "Est. Annual Cost",
                    "occurrences": "Times Seen",
                    "first_seen_fmt": "First Seen",
                    "last_seen_fmt": "Last Seen",
                }),
                use_container_width=True,
                hide_index=False,
            )
            total_recurring_annual = recurring["annual_cost"].sum()
            st.markdown(
                f"**Estimated total annual cost of recurring charges: "
                f"${total_recurring_annual:,.2f}**"
            )

    # ── Tab 3: Subscriptions ──────────────────────────────────────────────
    with tab3:
        st.markdown("#### 📋 Possible Forgotten Subscriptions")
        st.markdown(
            '<div class="section-note">'
            "Small, consistent charges that are easy to forget about. "
            "Sorted by 'forgettability' — the ones most likely to be autopilot spending. "
            "Could you cancel any of these?"
            "</div>",
            unsafe_allow_html=True,
        )

        subscriptions = st.session_state.subscriptions
        if months < 2:
            st.warning("Upload at least 2 months of statements to detect subscriptions.")
        elif subscriptions is None or subscriptions.empty:
            st.info("No small recurring subscriptions detected.")
        else:
            st.dataframe(
                subscriptions[["merchant", "frequency", "avg_charge_fmt",
                               "annual_cost_fmt", "occurrences", "first_seen_fmt"]].rename(columns={
                    "merchant": "Merchant",
                    "frequency": "Frequency",
                    "avg_charge_fmt": "Per Period",
                    "annual_cost_fmt": "Per Year",
                    "occurrences": "Times Seen",
                    "first_seen_fmt": "Paying Since",
                }),
                use_container_width=True,
                hide_index=False,
            )
            total_sub_annual = subscriptions["annual_cost"].sum()
            st.markdown(
                f"**Total possible subscription spend: ${total_sub_annual:,.2f}/year** — "
                f"that's **${total_sub_annual/12:,.2f}/month** in charges you might not be thinking about."
            )

    # ── Tab 4: Year-over-Year ─────────────────────────────────────────────
    with tab4:
        st.markdown("#### 📈 Year-over-Year Spending Changes")

        yoy = st.session_state.yoy
        if not has_yoy:
            st.info(
                "📅 Year-over-Year analysis requires at least 2 years of statements.\n\n"
                f"Currently loaded: **{', '.join(str(y) for y in years)}**.\n\n"
                "Upload statements from an additional year to unlock this tab."
            )
        elif yoy is None or yoy.empty:
            st.info("No significant year-over-year changes found in the data.")
        else:
            increases = yoy[yoy["delta"] > 0]
            decreases = yoy[yoy["delta"] < 0]

            if not increases.empty:
                st.markdown("##### ↑ Charges That Increased")
                st.markdown(
                    '<div class="section-note">These cost you more this year than last year.</div>',
                    unsafe_allow_html=True,
                )
                inc_display = increases[["merchant", "year_a", "year_b",
                                         "amount_a_fmt", "amount_b_fmt",
                                         "delta_fmt", "pct_fmt"]].rename(columns={
                    "merchant": "Merchant",
                    "year_a": "Year A",
                    "year_b": "Year B",
                    "amount_a_fmt": "Spent (A)",
                    "amount_b_fmt": "Spent (B)",
                    "delta_fmt": "Change ($)",
                    "pct_fmt": "Change (%)",
                })
                st.dataframe(inc_display, use_container_width=True, hide_index=False)

            if not decreases.empty:
                st.markdown("##### ↓ Charges That Decreased")
                st.markdown(
                    '<div class="section-note">You spent less here — cancellations, negotiated rates, or reduced usage.</div>',
                    unsafe_allow_html=True,
                )
                dec_display = decreases[["merchant", "year_a", "year_b",
                                          "amount_a_fmt", "amount_b_fmt",
                                          "delta_fmt", "pct_fmt"]].rename(columns={
                    "merchant": "Merchant",
                    "year_a": "Year A",
                    "year_b": "Year B",
                    "amount_a_fmt": "Spent (A)",
                    "amount_b_fmt": "Spent (B)",
                    "delta_fmt": "Change ($)",
                    "pct_fmt": "Change (%)",
                })
                st.dataframe(dec_display, use_container_width=True, hide_index=False)

    # ── Tab 5: AI Insights ────────────────────────────────────────────────
    with tab5:
        st.markdown("#### 🔍 AI Insights")
        st.markdown(
            '<div class="section-note">'
            "The AI analyzes your aggregated spending data — not your raw transactions. "
            "Merchant names and totals are shared with the AI provider you select; "
            "no account numbers, card numbers, or personal details are ever sent."
            "</div>",
            unsafe_allow_html=True,
        )

        if not api_key:
            st.warning(
                "Enter your API key in the sidebar to use AI Insights. "
                "Choose any provider — OpenAI, Gemini, or Anthropic Claude."
            )
        else:
            depth = st.radio(
                "Analysis depth",
                ["Summary bullets", "Deep narrative analysis"],
                horizontal=True,
                help="Deep analysis uses more tokens (~3-5x the cost of summary).",
            )

            run_ai_btn = st.button(
                f"✨ Run AI Analysis ({provider})",
                type="secondary",
            )

            if run_ai_btn:
                with st.spinner(f"Analyzing with {provider}..."):
                    result = get_ai_insights(
                        data_summary=st.session_state.llm_summary_text,
                        provider=provider,
                        api_key=api_key,
                        depth=depth,
                    )
                    st.session_state.ai_result = result

            if st.session_state.ai_result:
                st.markdown(st.session_state.ai_result)

                st.download_button(
                    label="⬇️ Download AI Analysis",
                    data=st.session_state.ai_result,
                    file_name="statement_ai_insights.txt",
                    mime="text/plain",
                )

    # ── Download full analysis ─────────────────────────────────────────────
    st.markdown("---")
    st.download_button(
        label="⬇️ Download Full Analysis Data (text)",
        data=st.session_state.llm_summary_text,
        file_name="statement_analysis_summary.txt",
        mime="text/plain",
    )

else:
    # Landing state
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 1rem; color: #9ca3af;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">💳</div>
            <div style="font-size: 1rem;">
                Upload your credit card statements above and click <strong>Analyze</strong>.<br>
                Supports PDF, CSV, XLS, XLSX, and DOCX from any bank.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
