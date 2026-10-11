"""
AI Personal Finance Intelligence Assistant — Streamlit Edition
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

load_dotenv()  # reads GEMINI_API_KEY / OPENAI_API_KEY from the .env file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("finance-assistant")

GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"

CATEGORIES = [
    "Groceries",
    "Dining out",
    "Subscriptions & Utilities",
    "Shopping",
    "Transport",
    "Income",
    "Miscellaneous",
]

CATEGORY_COLORS = {
    "Groceries": "#10B981",
    "Dining out": "#F59E0B",
    "Subscriptions & Utilities": "#6366F1",
    "Shopping": "#EC4899",
    "Transport": "#3B82F6",
    "Miscellaneous": "#64748B",
    "Income": "#34D399",
}

PROVIDER_LABELS = {
    "gemini": "Google Gemini",
    "openai": "OpenAI",
    "rules": "Offline keyword rules",
    "preset": "Preset (sample data)",
}

CLASSIFIER_PROMPT = (
    "You are a bank-transaction classifier. You receive a JSON array of vendor or "
    "transaction description strings. Return ONLY a JSON object mapping each input "
    f"string to exactly one of these categories: {', '.join(CATEGORIES)}. "
    "Salary, payroll, deposits and interest are 'Income'. Bank fees, ATM withdrawals "
    "and anything ambiguous are 'Miscellaneous'. No commentary, no code fences."
)

CHAT_PROMPT = (
    "You are 'Ask My Finances', a concise personal-finance analyst inside a budgeting "
    "dashboard. Answer ONLY from the DATA SUMMARY provided — never invent numbers. "
    "Amounts are USD. Be specific (quote figures, dates, vendors), use short markdown "
    "bullets where helpful, and keep answers under 150 words unless asked for depth. "
    "If the summary lacks the data to answer, say so plainly."
)

KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("Income", ["salary", "payroll", "direct deposit", "paycheck", "invoice payment",
                "freelance", "interest paid", "dividend", "refund"]),
    ("Groceries", ["grocery", "supermarket", "whole foods", "trader joe", "kroger",
                   "aldi", "safeway", "publix", "costco", "wegmans", "food market"]),
    ("Dining out", ["restaurant", "cafe", "coffee", "pizza", "sushi", "taqueria",
                    "bistro", "deli", "taco", "burger", "mcdonald", "chipotle",
                    "starbucks", "domino", "doordash", "uber eats", "grubhub",
                    "sweetgreen", "dunkin", "diner", "kitchen", "grill"]),
    ("Subscriptions & Utilities", ["netflix", "spotify", "hulu", "disney+", "youtube",
                                   "icloud", "adobe", "microsoft 365", "electric",
                                   "power & light", "water", "internet", "comcast",
                                   "xfinity", "verizon", "at&t", "t-mobile", "phone bill",
                                   "rent", "insurance", "gym", "membership", "subscription"]),
    ("Shopping", ["amazon", "target", "walmart", "zara", "uniqlo", "h&m", "ikea",
                  "best buy", "apple store", "nike", "etsy", "sephora", "walgreens",
                  "cvs", "kindle", "mall", "outfitters"]),
    ("Transport", ["uber", "lyft", "shell", "chevron", "exxon", "gas station", "metro",
                   "transit", "mta", "parking", "toll", "amtrak", "airline", "delta",
                   "taxi", "mobil"]),
]


def usd(value: float) -> str:
    return f"${value:,.2f}"


def rule_classify(description: str) -> str:
    text = description.lower()
    for category, keywords in KEYWORD_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return "Miscellaneous"


_SAMPLE_ROWS: list[tuple[int, str, float, str]] = [
    (32, "ACME Corp Payroll", 4200.00, "Income"),
    (2, "ACME Corp Payroll", 4200.00, "Income"),
    (14, "Freelance Invoice #1042 - Design", 850.00, "Income"),
    (45, "ACME Corp Payroll", 4100.00, "Income"),
    (1, "Whole Foods Market", 86.40, "Groceries"),
    (5, "Trader Joe's", 54.17, "Groceries"),
    (8, "Kroger", 72.30, "Groceries"),
    (12, "Aldi", 41.65, "Groceries"),
    (15, "Whole Foods Market", 103.12, "Groceries"),
    (19, "Costco Wholesale", 148.77, "Groceries"),
    (22, "Trader Joe's", 38.90, "Groceries"),
    (26, "Kroger", 66.05, "Groceries"),
    (30, "Whole Foods Market", 92.40, "Groceries"),
    (40, "Aldi", 35.20, "Groceries"),
    (0, "Sweetgreen", 14.50, "Dining out"),
    (2, "Chipotle Mexican Grill", 18.75, "Dining out"),
    (4, "Starbucks", 6.45, "Dining out"),
    (6, "Domino's Pizza", 27.90, "Dining out"),
    (9, "Blue Bottle Coffee", 5.25, "Dining out"),
    (11, "Taqueria El Sol", 16.80, "Dining out"),
    (13, "Starbucks", 5.85, "Dining out"),
    (17, "Sushi Katsu", 48.60, "Dining out"),
    (21, "Chipotle Mexican Grill", 15.20, "Dining out"),
    (25, "The Rustic Bistro", 62.35, "Dining out"),
    (29, "Starbucks", 4.95, "Dining out"),
    (36, "Domino's Pizza", 22.45, "Dining out"),
    (3, "Comcast Xfinity Internet", 79.99, "Subscriptions & Utilities"),
    (7, "Netflix", 15.49, "Subscriptions & Utilities"),
    (7, "Spotify Premium", 10.99, "Subscriptions & Utilities"),
    (10, "City Power & Electric", 84.20, "Subscriptions & Utilities"),
    (16, "Planet Fitness Membership", 24.99, "Subscriptions & Utilities"),
    (16, "iCloud Storage 200GB", 2.99, "Subscriptions & Utilities"),
    (33, "Netflix", 15.49, "Subscriptions & Utilities"),
    (33, "Spotify Premium", 10.99, "Subscriptions & Utilities"),
    (35, "Verizon Wireless Phone Bill", 45.00, "Subscriptions & Utilities"),
    (44, "City Power & Electric", 91.65, "Subscriptions & Utilities"),
    (44, "Rent - Apt 12B", 1650.00, "Subscriptions & Utilities"),
    (44, "Renters Insurance", 18.50, "Subscriptions & Utilities"),
    (6, "Amazon Marketplace", 43.99, "Shopping"),
    (13, "Target", 67.20, "Shopping"),
    (18, "Zara", 89.90, "Shopping"),
    (24, "Amazon Marketplace", 132.45, "Shopping"),
    (28, "Apple Store", 1299.00, "Shopping"),
    (34, "IKEA Brooklyn", 96.30, "Shopping"),
    (41, "Uniqlo", 74.50, "Shopping"),
    (0, "Uber Ride", 18.30, "Transport"),
    (3, "Shell Gas Station", 46.10, "Transport"),
    (5, "MTA Metro Card", 2.90, "Transport"),
    (8, "Lyft Ride", 14.75, "Transport"),
    (14, "Uber Ride", 24.60, "Transport"),
    (20, "Shell Gas Station", 41.85, "Transport"),
    (27, "Parking Garage - Downtown", 15.00, "Transport"),
    (38, "Uber Ride", 21.40, "Transport"),
    (4, "CVS Pharmacy", 23.15, "Miscellaneous"),
    (9, "ATM Withdrawal", 60.00, "Miscellaneous"),
    (17, "Charity: Water Donation", 25.00, "Miscellaneous"),
    (31, "US Bank Service Fee", 3.00, "Miscellaneous"),
    (42, "Amazon Kindle Book", 12.99, "Miscellaneous"),
]


def sample_dataframe() -> pd.DataFrame:
    anchor = datetime.now().date()
    rows = [
        {
            "Date": anchor - timedelta(days=days_ago),
            "Description": description,
            "Amount": amount,
            "Category": category,
        }
        for days_ago, description, amount, category in _SAMPLE_ROWS
    ]
    frame = pd.DataFrame(rows)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.sort_values("Date", ascending=False).reset_index(drop=True)


class CsvParseError(Exception):
    pass


class CSVParser:
    COLUMN_ALIASES = {
        "date": ["date", "transaction date", "posted date", "post date", "txn date"],
        "description": ["description", "name", "transaction", "details", "payee",
                        "merchant", "narrative", "memo"],
        "amount": ["amount", "value", "transaction amount", "debit", "charge"],
        "category": ["category", "type", "label"],
    }
    REQUIRED = ("date", "description", "amount")

    def parse(self, raw: bytes, filename: str) -> pd.DataFrame:
        frame = self._read(raw, filename)
        frame = self._standardize_columns(frame)
        frame = self._coerce_types(frame)
        return frame

    @staticmethod
    def _read(raw: bytes, filename: str) -> pd.DataFrame:
        try:
            separator = ","
            try:
                import csv as _csv
                separator = _csv.Sniffer().sniff(raw[:2048].decode("utf-8", errors="replace"),
                                                 delimiters=",;\t|").delimiter
            except Exception:
                pass
            return pd.read_csv(io.BytesIO(raw), sep=separator)
        except Exception as exc:
            raise CsvParseError(f"Could not read '{filename}' as CSV: {exc}") from exc

    @classmethod
    def _standardize_columns(cls, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = {cls._normalize_header(col): col for col in frame.columns}
        found: dict[str, str] = {}
        for canonical, aliases in cls.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in normalized:
                    found[canonical] = normalized[alias]
                    break

        missing = [name for name in cls.REQUIRED if name not in found]
        if missing:
            raise CsvParseError(
                f"Missing required column(s): {', '.join(missing)}. "
                f"Found columns: {', '.join(map(str, frame.columns))}."
            )
        return frame.rename(columns={actual: canonical for canonical, actual in found.items()})

    @staticmethod
    def _normalize_header(name: object) -> str:
        return re.sub(r"[^a-z0-9 ]", " ", str(name).strip().lower()).strip()

    @staticmethod
    def _coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Amount"] = frame["Amount"].apply(_clean_amount)
        frame["Description"] = frame["Description"].astype(str).str.strip()

        frame = frame.dropna(subset=["Date", "Amount"])
        frame = frame[frame["Description"].ne("") & frame["Description"].str.lower().ne("nan")]
        if frame.empty:
            raise CsvParseError("No usable rows found in CSV.")

        if "Category" in frame.columns:
            frame["Category"] = frame["Category"].astype(str).str.strip()
            frame["Category"] = frame["Category"].replace({"nan": "", "None": "", "": ""})
        else:
            frame["Category"] = ""

        frame["Amount"] = frame["Amount"].abs().round(2)
        return frame[["Date", "Description", "Amount", "Category"]] \
            .sort_values("Date", ascending=False).reset_index(drop=True)


def _clean_amount(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace(",", "").replace("$", "").replace(" ", "")
    if not text or text in {"-", "--"}:
        return None
    negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
    text = text.strip("()-+")
    try:
        number = float(text)
    except ValueError:
        return None
    return -abs(number) if negative else number


class AICategorizer:
    BATCH_SIZE = 50

    def categorize_dataframe(self, df: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame, str]:
        mask = pd.Series(True, index=df.index) if force else ~df["Category"].isin(CATEGORIES)
        vendors = sorted(set(df.loc[mask, "Description"]))
        if not vendors:
            return df, ""

        mapping: dict[str, str] = {}
        provider = ""
        for chunk in self._chunks(vendors):
            chunk_mapping, chunk_provider = cached_classify(tuple(chunk))
            mapping.update(chunk_mapping)
            provider = chunk_provider or provider

        df = df.copy()
        df.loc[mask, "Category"] = df.loc[mask, "Description"].map(
            lambda vendor: mapping.get(vendor, "Miscellaneous")
        )
        return df, provider

    def _chunks(self, vendors: list[str]) -> list[list[str]]:
        return [vendors[i:i + self.BATCH_SIZE] for i in range(0, len(vendors), self.BATCH_SIZE)]

    def via_gemini(self, vendors: list[str]) -> dict[str, str]:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=json.dumps(vendors),
            config=genai_types.GenerateContentConfig(
                system_instruction=CLASSIFIER_PROMPT,
                response_mime_type="application/json",
            ),
        )
        return self._validate_mapping(response.text or "", vendors)

    def via_openai(self, vendors: list[str]) -> dict[str, str]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": json.dumps(vendors)},
            ],
        )
        return self._validate_mapping(completion.choices[0].message.content or "", vendors)

    def via_rules(self, vendors: list[str]) -> dict[str, str]:
        return {vendor: rule_classify(vendor) for vendor in vendors}

    @staticmethod
    def _validate_mapping(text: str, vendors: list[str]) -> dict[str, str]:
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("model reply contained no JSON object")
        raw = json.loads(cleaned[start:end + 1])

        lookup = {str(key).strip(): str(value).strip() for key, value in raw.items()}
        mapping: dict[str, str] = {}
        for vendor in vendors:
            category = lookup.get(vendor, "")
            mapping[vendor] = category if category in CATEGORIES else "Miscellaneous"
        return mapping


@st.cache_data(show_spinner=False)
def cached_classify(vendors: tuple[str, ...]) -> tuple[dict[str, str], str]:
    categorizer = AICategorizer()
    for provider, method in (("gemini", categorizer.via_gemini), ("openai", categorizer.via_openai)):
        try:
            return method(vendors), provider
        except Exception as exc:
            logger.warning("AI classification via %s failed: %s", provider, exc)
    return categorizer.via_rules(vendors), "rules"


class FinanceAnalytics:
    @staticmethod
    def metrics(df: pd.DataFrame) -> dict[str, float | int]:
        income = float(df.loc[df["Category"] == "Income", "Amount"].sum())
        expenses = float(df.loc[df["Category"] != "Income", "Amount"].sum())
        return {
            "income": round(income, 2),
            "expenses": round(expenses, 2),
            "net": round(income - expenses, 2),
            "count": len(df),
        }

    @staticmethod
    def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
        spend = df[df["Category"] != "Income"]
        if spend.empty:
            return pd.DataFrame(columns=["Category", "Amount"])
        return (spend.groupby("Category")["Amount"].sum().round(2)
                .sort_values(ascending=False).reset_index())

    @staticmethod
    def spend_over_time(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        spend = df[df["Category"] != "Income"].copy()
        if spend.empty:
            return pd.DataFrame(columns=["Period", "Amount"]), "day"

        span_days = (spend["Date"].max() - spend["Date"].min()).days
        if span_days <= 60:
            grouped = spend.groupby(pd.Grouper(key="Date", freq="D"))["Amount"].sum().reset_index()
            grouped["Period"] = grouped["Date"].dt.strftime("%b %d")
            return grouped[["Period", "Amount"]], "day"

        grouped = spend.groupby(pd.Grouper(key="Date", freq="W"))["Amount"].sum().reset_index()
        grouped["Period"] = grouped["Date"].dt.strftime("Week of %b %d")
        return grouped[["Period", "Amount"]], "week"

    @classmethod
    def summary_text(cls, df: pd.DataFrame) -> str:
        metrics = cls.metrics(df)
        lines = [
            f"Statement period: {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d} "
            f"({metrics['count']} transactions).",
            f"Total income: ${metrics['income']:,.2f}. Total expenses: "
            f"${metrics['expenses']:,.2f}. Net savings: ${metrics['net']:,.2f}.",
            "Spending by category:",
        ]

        breakdown = cls.category_breakdown(df)
        counts = df[df["Category"] != "Income"].groupby("Category").size()
        for row in breakdown.itertuples(index=False):
            lines.append(f"  - {row.Category}: ${row.Amount:,.2f} across "
                         f"{int(counts.get(row.Category, 0))} transactions")

        monthly = df[df["Category"] != "Income"].groupby(df["Date"].dt.to_period("M"))["Amount"].sum()
        if not monthly.empty:
            lines.append("Monthly spending: "
                         + ", ".join(f"{period}: ${value:,.2f}" for period, value in monthly.items()))

        top = df[df["Category"] != "Income"].nlargest(5, "Amount")
        if not top.empty:
            lines.append("Largest individual transactions: "
                         + "; ".join(f"{row.Description} ${row.Amount:,.2f} on {row.Date:%Y-%m-%d}"
                                     for row in top.itertuples(index=False)))
        return "\n".join(lines)


class FinanceChatbot:
    def ask(self, question: str, df: pd.DataFrame) -> str:
        summary = FinanceAnalytics.summary_text(df)
        system = f"{CHAT_PROMPT}\n\nDATA SUMMARY:\n{summary}"
        for method in (self._via_gemini, self._via_openai):
            try:
                answer = method(system, question)
                if answer:
                    return answer
            except Exception as exc:
                logger.warning("chat via %s failed: %s", method.__name__, exc)
        return self._local_answer(question, df)

    @staticmethod
    def _via_gemini(system: str, question: str) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=question,
            config=genai_types.GenerateContentConfig(system_instruction=system),
        )
        return (response.text or "").strip()

    @staticmethod
    def _via_openai(system: str, question: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        )
        return (completion.choices[0].message.content or "").strip()

    @staticmethod
    def _local_answer(question: str, df: pd.DataFrame) -> str:
        text = question.lower()
        spend = df[df["Category"] != "Income"]
        metrics = FinanceAnalytics.metrics(df)

        if any(word in text for word in ("highest", "largest", "biggest")):
            top = spend.nlargest(3, "Amount")
            if top.empty:
                return "There are no expense transactions in the current data."
            lines = ["Your largest expenses were:"] + [
                f"- {row.Description}: ${row.Amount:,.2f} on {row.Date:%b %d, %Y}"
                for row in top.itertuples(index=False)
            ]
            return "\n".join(lines)
        if "income" in text:
            return f"Total income in this statement is ${metrics['income']:,.2f}."
        if any(word in text for word in ("save", "reduce", "cut")) and not spend.empty:
            biggest = FinanceAnalytics.category_breakdown(df).iloc[0]
            return (f"To cut spending, start with **{biggest.Category}** — your largest "
                    f"category at ${biggest.Amount:,.2f}.")
        if any(word in text for word in ("total", "spend", "spent", "how much")):
            return (f"Total spending is ${metrics['expenses']:,.2f} across "
                    f"{metrics['count']} transactions.")
        return FinanceAnalytics.summary_text(df)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("📂 Data source")

        uploaded = st.file_uploader(
            "Bank statement (CSV)",
            type="csv",
            help="Expected columns: Date, Description, Amount — Category optional.",
        )
        if uploaded is not None and st.session_state.get("source") != uploaded.name:
            try:
                df_new = CSVParser().parse(uploaded.getvalue(), uploaded.name)
            except CsvParseError as exc:
                st.error(f"⚠️ {exc}")
            else:
                with st.spinner("Classifying vendors with AI…"):
                    df_new, provider = AICategorizer().categorize_dataframe(df_new)
                st.session_state.update(df=df_new, source=uploaded.name,
                                        classifier=provider or "rules", chat_history=[])
                st.rerun()

        if st.button("🧪 Load sample data", use_container_width=True):
            st.session_state.update(df=sample_dataframe(), source="Built-in sample data",
                                    classifier="preset", chat_history=[])
            st.rerun()

        if st.button("🧠 Re-categorize with AI", use_container_width=True):
            with st.spinner("Re-classifying every vendor…"):
                df_new, provider = AICategorizer().categorize_dataframe(
                    st.session_state["df"], force=True)
            if provider:
                st.session_state.update(df=df_new, classifier=provider)
            st.rerun()

        st.divider()
        st.caption(f"🔑 Gemini key: **{'set' if os.environ.get('GEMINI_API_KEY') else 'missing'}**")


def render_metrics(metrics: dict[str, float | int]) -> None:
    col_income, col_expenses, col_net = st.columns(3)
    col_income.metric("💵 Total Income", usd(float(metrics["income"])))
    col_expenses.metric("💳 Total Expenses", usd(float(metrics["expenses"])))
    col_net.metric(
        "🏦 Net Savings",
        usd(float(metrics["net"])),
        delta="saving" if metrics["net"] >= 0 else "overspending",
        delta_color="normal" if metrics["net"] >= 0 else "inverse",
    )


def render_category_pie(breakdown: pd.DataFrame) -> None:
    st.subheader("🥧 Expense distribution by category")
    if breakdown.empty:
        st.info("No expense transactions to chart yet.")
        return
    fig = px.pie(breakdown, values="Amount", names="Category", hole=0.45,
                 color="Category", color_discrete_map=CATEGORY_COLORS)
    fig.update_traces(textposition="inside", textinfo="percent")
    fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10),
                      legend_title_text=None)
    st.plotly_chart(fig, use_container_width=True)


def render_trend_bar(trend: pd.DataFrame, granularity: str) -> None:
    st.subheader(f"📊 Spending over time (by {granularity})")
    if trend.empty:
        st.info("No spending to chart yet.")
        return
    fig = px.bar(trend, x="Period", y="Amount", color_discrete_sequence=["#0284C7"])
    fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10),
                      xaxis_title=None, yaxis_title="Spent (USD)")
    st.plotly_chart(fig, use_container_width=True)


def render_ledger(df: pd.DataFrame) -> None:
    st.subheader("🧾 Transactions")
    st.caption("Override any AI category from the dropdown — the dashboard updates live.")
    edited = st.data_editor(
        df,
        key=f"ledger-{st.session_state.get('source', 'sample')}",
        column_config={
            "Date": st.column_config.DateColumn("Date", disabled=True),
            "Description": st.column_config.TextColumn("Description", disabled=True, width="large"),
            "Amount": st.column_config.NumberColumn("Amount", disabled=True, format="$%.2f"),
            "Category": st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True),
        },
        hide_index=True,
        use_container_width=True,
        height=420,
    )
    st.session_state["df"] = edited
    st.download_button("⬇️ Download classified CSV",
                       data=edited.to_csv(index=False).encode("utf-8"),
                       file_name="classified_transactions.csv", mime="text/csv")


def render_chat(df: pd.DataFrame) -> None:
    st.subheader("💬 Ask My Finances")
    with st.expander("What the AI knows about your data"):
        st.code(FinanceAnalytics.summary_text(df), language="markdown")

    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("e.g. What was my highest expense?"):
        st.session_state.setdefault("chat_history", []).append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your finances…"):
                answer = FinanceChatbot().ask(prompt, df)
            st.markdown(answer)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})


def main() -> None:
    st.set_page_config(page_title="AI Finance Assistant", page_icon="💳", layout="wide")
    st.title("💳 AI Personal Finance Intelligence Assistant")

    if "df" not in st.session_state:
        st.session_state.update(df=sample_dataframe(), source="Built-in sample data",
                                classifier="preset", chat_history=[])

    render_sidebar()
    df = st.session_state["df"]
    metrics = FinanceAnalytics.metrics(df)

    render_metrics(metrics)
    pie_col, bar_col = st.columns(2)
    with pie_col:
        render_category_pie(FinanceAnalytics.category_breakdown(df))
    with bar_col:
        render_trend_bar(*FinanceAnalytics.spend_over_time(df))

    render_ledger(df)
    render_chat(df)


if __name__ == "__main__":
    main()