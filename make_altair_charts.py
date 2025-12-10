"""Creates Altair visualizations and exports data for D3 charts."""

import os
import json
import pandas as pd
import altair as alt
import yfinance as yf

alt.data_transformers.disable_max_rows()

TICKERS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "XOM": "Energy",
    "CVX": "Energy",
    "JPM": "Finance",
    "BAC": "Finance",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
}

START_DATE = "2019-01-01"
END_DATE = "2025-12-01"
OUTPUT_DIR = "."

def download_price_data(tickers, start_date, end_date):
    ticker_list = list(tickers.keys())

    # Download all tickers at once for efficiency
    raw = yf.download(
        tickers=ticker_list,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    all_rows = []

    for ticker in ticker_list:
        if ticker not in raw.columns.get_level_values(0):
            # In case data is missing for some ticker
            continue

        # Raw has a multi index column (field, ticker)
        # or (ticker, field) depending on yfinance version
        # We handle both possibilities
        cols = raw.columns

        if ticker in cols.get_level_values(0):
            # layout: (ticker, field)
            df_t = raw[ticker].copy()
        else:
            # layout: (field, ticker)
            df_t = raw.xs(ticker, axis=1, level=1).copy()

        df_t = df_t.reset_index()
        df_t["Ticker"] = ticker
        all_rows.append(df_t)

    df = pd.concat(all_rows, ignore_index=True)

    # Standard column names
    df.rename(
        columns={
            "Date": "date",
            "Adj Close": "adj_close",
            "Close": "adj_close",  # fallback if Adj Close missing
            "Volume": "volume",
        },
        inplace=True,
    )

    # Keep only needed columns
    keep_cols = ["date", "Ticker", "adj_close", "volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Add sector labels
    df["sector"] = df["Ticker"].map(tickers)

    # Drop rows with missing price or sector
    df = df.dropna(subset=["adj_close", "sector"])

    # Ensure date is datetime type
    df["date"] = pd.to_datetime(df["date"])

    return df


def add_return_features(df):
    df = df.sort_values(["Ticker", "date"]).copy()

    # Daily percent change for each ticker
    df["daily_return"] = (
        df.groupby("Ticker")["adj_close"].pct_change()
    )

    # Moving average and volatility over five day window
    window = 5
    df["ma_5"] = (
        df.groupby("Ticker")["adj_close"].rolling(window).mean().reset_index(level=0, drop=True)
    )
    df["vol_5"] = (
        df.groupby("Ticker")["daily_return"].rolling(window).std().reset_index(level=0, drop=True)
    )

    # Sector level daily returns: equal weight average across tickers in sector
    sector_daily = (
        df.dropna(subset=["daily_return"])
        .groupby(["date", "sector"])["daily_return"]
        .mean()
        .reset_index()
        .rename(columns={"daily_return": "sector_daily_return"})
    )

    # Merge back
    df = df.merge(
        sector_daily,
        on=["date", "sector"],
        how="left",
    )

    return df


def build_sector_index(df):
    sector_df = (
        df[["date", "sector", "sector_daily_return"]]
        .drop_duplicates()
        .dropna(subset=["sector_daily_return"])
        .sort_values(["sector", "date"])
        .copy()
    )

    def compute_index(group):
        group = group.sort_values("date").copy()
        group["sector_index"] = 100 * (1 + group["sector_daily_return"]).cumprod()
        return group

    sector_df = sector_df.groupby("sector", group_keys=False).apply(compute_index)

    return sector_df


def build_company_summary(df):
    summary = (
        df.dropna(subset=["daily_return"])
        .groupby(["Ticker", "sector"])["daily_return"]
        .agg(
            avg_return="mean",
            volatility="std",
        )
        .reset_index()
    )

    summary["avg_return_pct"] = summary["avg_return"] * 100.0
    summary["volatility_pct"] = summary["volatility"] * 100.0
    return summary


def build_sector_correlation(df):
    sector_series = (
        df[["date", "sector", "sector_daily_return"]]
        .dropna(subset=["sector_daily_return"])
        .drop_duplicates()
        .pivot(index="date", columns="sector", values="sector_daily_return")
    )

    corr = sector_series.corr()

    corr_long = (
        corr.reset_index()
        .melt(
            id_vars="sector",
            var_name="sector_other",
            value_name="correlation",
        )
    )

    return corr_long


def make_chart_normalized_prices(sector_index_df):
    # Define major events
    events = [
        {"date": "2020-03-11", "label": "COVID-19 Pandemic", "color": "red"},
        {"date": "2020-03-20", "label": "Market Crash", "color": "orange"},
        {"date": "2022-01-01", "label": "2022 Recession", "color": "purple"},
    ]
    
    # Create event lines
    event_data = pd.DataFrame(events)
    event_data["date"] = pd.to_datetime(event_data["date"])
    
    event_lines = (
        alt.Chart(event_data)
        .mark_rule(strokeDash=[5, 5], strokeWidth=2)
        .encode(
            x=alt.X("date:T"),
            color=alt.Color("label:N", title="Major Events", scale=alt.Scale(domain=["COVID-19 Pandemic", "Market Crash", "2022 Recession"], 
                                                                           range=["red", "orange", "purple"])),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("label:N", title="Event")],
        )
    )
    
    # Main chart
    chart = (
        alt.Chart(sector_index_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "sector_index:Q",
                title="Normalized sector index (start equals 100)",
            ),
            color=alt.Color("sector:N", title="Sector"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("sector_index:Q", title="Index", format=".1f"),
            ],
        )
        .properties(
            width=700,
            height=400,
            title="Normalized sector price indexes over time",
        )
    )
    
    # Combine chart with event lines
    return chart + event_lines


def make_chart_return_vs_vol(summary_df, prices_df):
    # Create monthly aggregated data for trend visualization
    prices_df["year_month"] = prices_df["date"].dt.to_period("M").dt.to_timestamp()
    monthly_data = (
        prices_df.groupby(["Ticker", "sector", "year_month"])["daily_return"]
        .agg(["mean", "std"])
        .reset_index()
    )
    monthly_data["avg_return_pct"] = monthly_data["mean"] * 100
    monthly_data["volatility_pct"] = monthly_data["std"] * 100
    
    # Create selection for company/sector filtering
    sector_selection = alt.selection_multi(fields=["sector"], bind="legend")
    company_selection = alt.selection_multi(fields=["Ticker"])
    
    # Monthly trend chart
    monthly_chart = (
        alt.Chart(monthly_data)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("year_month:T", title="Month"),
            y=alt.Y("avg_return_pct:Q", title="Average Monthly Return (%)"),
            color=alt.Color("sector:N", title="Sector"),
            tooltip=[
                alt.Tooltip("Ticker:N", title="Ticker"),
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("year_month:T", title="Month"),
                alt.Tooltip("avg_return_pct:Q", title="Avg Return", format=".3f"),
            ],
        )
        .add_selection(sector_selection)
        .transform_filter(sector_selection)
        .properties(
            width=600,
            height=200,
            title="Monthly Return Trends",
        )
    )
    
    # Scatter plot with selection
    scatter = (
        alt.Chart(summary_df)
        .mark_circle(size=100)
        .encode(
            x=alt.X(
                "volatility_pct:Q",
                title="Volatility (daily standard deviation in percent)",
            ),
            y=alt.Y(
                "avg_return_pct:Q",
                title="Average daily return in percent",
            ),
            color=alt.Color("sector:N", title="Sector"),
            size=alt.condition(company_selection, alt.value(200), alt.value(100)),
            opacity=alt.condition(company_selection, alt.value(1), alt.value(0.6)),
            tooltip=[
                alt.Tooltip("Ticker:N", title="Ticker"),
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("avg_return_pct:Q", title="Average return", format=".3f"),
                alt.Tooltip("volatility_pct:Q", title="Volatility", format=".3f"),
            ],
        )
        .add_selection(company_selection)
        .add_selection(sector_selection)
        .transform_filter(sector_selection)
        .properties(
            width=600,
            height=400,
            title="Average daily return vs volatility by company",
        )
    )
    
    return alt.vconcat(monthly_chart, scatter).resolve_scale(color="shared")


def make_sector_comparison_charts(summary_df, prices_df):
    """Create bar and box plots for sector comparison."""
    # Bar chart: Average return by sector
    sector_avg_return = (
        summary_df.groupby("sector")["avg_return_pct"]
        .mean()
        .reset_index()
        .sort_values("avg_return_pct", ascending=False)
    )
    
    bar_return = (
        alt.Chart(sector_avg_return)
        .mark_bar()
        .encode(
            x=alt.X("sector:N", title="Sector", sort="-y", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("avg_return_pct:Q", title="Average Daily Return (%)"),
            color=alt.Color("sector:N", title="Sector", legend=None),
            tooltip=[
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("avg_return_pct:Q", title="Avg Return", format=".3f"),
            ],
        )
        .properties(
            width=300,
            height=300,
            title="Average Return by Sector",
        )
    )
    
    # Bar chart: Average volatility by sector
    sector_avg_vol = (
        summary_df.groupby("sector")["volatility_pct"]
        .mean()
        .reset_index()
        .sort_values("volatility_pct", ascending=False)
    )
    
    bar_vol = (
        alt.Chart(sector_avg_vol)
        .mark_bar()
        .encode(
            x=alt.X("sector:N", title="Sector", sort="-y", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("volatility_pct:Q", title="Average Volatility (%)"),
            color=alt.Color("sector:N", title="Sector", legend=None),
            tooltip=[
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("volatility_pct:Q", title="Avg Volatility", format=".3f"),
            ],
        )
        .properties(
            width=300,
            height=300,
            title="Average Volatility by Sector",
        )
    )
    
    # Box plot: Return distribution by sector
    box_return = (
        alt.Chart(summary_df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("sector:N", title="Sector", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("avg_return_pct:Q", title="Daily Return (%)"),
            color=alt.Color("sector:N", title="Sector", legend=None),
            tooltip=[
                alt.Tooltip("Ticker:N", title="Ticker"),
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("avg_return_pct:Q", title="Return", format=".3f"),
            ],
        )
        .properties(
            width=300,
            height=300,
            title="Return Distribution by Sector",
        )
    )
    
    # Box plot: Volatility distribution by sector
    box_vol = (
        alt.Chart(summary_df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("sector:N", title="Sector", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("volatility_pct:Q", title="Volatility (%)"),
            color=alt.Color("sector:N", title="Sector", legend=None),
            tooltip=[
                alt.Tooltip("Ticker:N", title="Ticker"),
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("volatility_pct:Q", title="Volatility", format=".3f"),
            ],
        )
        .properties(
            width=300,
            height=300,
            title="Volatility Distribution by Sector",
        )
    )
    
    return alt.vconcat(
        alt.hconcat(bar_return, bar_vol),
        alt.hconcat(box_return, box_vol)
    )


def make_chart_correlation_heatmap(corr_long_df, prices_df):
    base = alt.Chart(corr_long_df)

    heatmap = (
        base.mark_rect()
        .encode(
            x=alt.X("sector:N", title="Sector", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("sector_other:N", title="Other Sector"),
            color=alt.Color(
                "correlation:Q",
                title="Correlation",
                scale=alt.Scale(scheme="redblue", domain=(-1, 1)),
            ),
            tooltip=[
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("sector_other:N", title="Other sector"),
                alt.Tooltip("correlation:Q", title="Correlation", format=".2f"),
            ],
        )
        .properties(
            width=450,
            height=450,
            title="Correlation between sector daily returns",
        )
    )

    text = (
        base.mark_text(size=12)
        .encode(
            x="sector:N",
            y="sector_other:N",
            text=alt.Text("correlation:Q", format=".2f"),
            color=alt.condition(
                "datum.correlation > 0.3",
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )
    
    # Create volatility distribution by sector over time
    prices_df["year"] = prices_df["date"].dt.year
    volatility_by_sector = (
        prices_df.dropna(subset=["vol_5"])
        .groupby(["year", "sector"])["vol_5"]
        .mean()
        .reset_index()
    )
    volatility_by_sector["volatility_pct"] = volatility_by_sector["vol_5"] * 100
    
    vol_chart = (
        alt.Chart(volatility_by_sector)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("volatility_pct:Q", title="Average Volatility (%)"),
            color=alt.Color("sector:N", title="Sector"),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("volatility_pct:Q", title="Volatility", format=".3f"),
            ],
        )
        .properties(
            width=450,
            height=300,
            title="Volatility Distribution Over Time by Sector",
        )
    )
    
    # Box plot for sector volatility comparison
    sector_volatility = (
        prices_df.dropna(subset=["vol_5"])
        .groupby(["sector", "date"])["vol_5"]
        .mean()
        .reset_index()
    )
    sector_volatility["volatility_pct"] = sector_volatility["vol_5"] * 100
    
    boxplot = (
        alt.Chart(sector_volatility)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("sector:N", title="Sector", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("volatility_pct:Q", title="Volatility (%)"),
            color=alt.Color("sector:N", title="Sector", legend=None),
            tooltip=[
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("volatility_pct:Q", title="Volatility", format=".3f"),
            ],
        )
        .properties(
            width=450,
            height=300,
            title="Volatility Distribution by Sector",
        )
    )

    return alt.vconcat(
        heatmap + text,
        alt.hconcat(vol_chart, boxplot)
    )


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading price data...")
    prices = download_price_data(TICKERS, START_DATE, END_DATE)
    print(f"Downloaded {len(prices)} rows of price data.")

    print("Adding return features...")
    prices = add_return_features(prices)

    print("Building sector index series...")
    sector_index = build_sector_index(prices)

    print("Building company summary...")
    company_summary = build_company_summary(prices)

    print("Computing sector correlations...")
    corr_long = build_sector_correlation(prices)

    print("Creating Altair charts...")
    make_chart_normalized_prices(sector_index).save(os.path.join(OUTPUT_DIR, "fig_altair_normalized_prices.html"))
    make_chart_return_vs_vol(company_summary, prices).save(os.path.join(OUTPUT_DIR, "fig_altair_return_volatility.html"))
    make_chart_correlation_heatmap(corr_long, prices).save(os.path.join(OUTPUT_DIR, "fig_altair_correlation_heatmap.html"))
    
    # Create additional sector comparison charts
    print("Creating sector comparison charts...")
    sector_comparison = make_sector_comparison_charts(company_summary, prices)
    sector_comparison.save(os.path.join(OUTPUT_DIR, "fig_altair_sector_comparison.html"))

    print("Exporting JSON data...")
    with open(os.path.join(OUTPUT_DIR, "company_summary.json"), "w") as f:
        json.dump(company_summary.to_dict(orient="records"), f, indent=2, default=str)
    
    time_series = prices[["date", "Ticker", "adj_close", "volume", "sector"]].copy()
    time_series["date"] = time_series["date"].dt.strftime("%Y-%m-%d")
    with open(os.path.join(OUTPUT_DIR, "time_series_data.json"), "w") as f:
        json.dump(time_series.to_dict(orient="records"), f, indent=2, default=str)
    
    print("Done.")


if __name__ == "__main__":
    main()
