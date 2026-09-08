import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

# ----- Page configuration
st.set_page_config(
    page_title="Hiking Analysis Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 100%;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ----- Load data
@st.cache_data
def load_data():

    url1 = "https://raw.githubusercontent.com/statzenthusiast921/hike_monitoring/refs/heads/main/data/synthetic_hiking_reviews.csv"
    df = pd.read_csv(url1)

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["date"].dt.year

    exclusions = [
        "bear grass",
        "beargrass",
        "bearings",
        "bearable",
        "bear spray",
        "no bear",
        "see any bears",
        "encounter any bears",
        "thankfully, we didn't spot any"
    ]

    df["bear_flag"] = df["review_text"].str.contains("bear", case=False, na=False)
    df["non_bear_reference"] = df["review_text"].str.contains("|".join(exclusions), case=False, na=False)
    df["actual_bear_sighting"] = (df["bear_flag"] & ~df["non_bear_reference"])

    df = df.sort_values("date").reset_index(drop=True)

    url2 = "https://raw.githubusercontent.com/statzenthusiast921/hike_monitoring/refs/heads/main/data/final_model_results.csv"
    model_data = pd.read_csv(url2)
    model_data = model_data[['trail_name','year_month','avg_monthly_rating','key']]

    url3 = "https://raw.githubusercontent.com/statzenthusiast921/hike_monitoring/refs/heads/main/data/trail_model_estimates.csv"
    model_estimates = pd.read_csv(url3)
    model_estimates = model_estimates[~(model_estimates['term'] == "(Intercept)")]
    model_estimates = model_estimates[['trail_name','term','estimate','p.value']]

    return df, model_data, model_estimates

df, model_data, model_estimates = load_data()

# ----- Main page title
st.title("Hiking Analysis Dashboard")

# ----- Top nav (styled radio buttons — behaves like tabs but keeps state across reruns)
st.markdown(
    """
    <style>
    /* Force the Streamlit radio container and all parent wrappers to full width */
    .st-key-top_nav, 
    .st-key-top_nav > div, 
    .st-key-top_nav [data-testid="stRadio"], 
    .st-key-top_nav [data-testid="stRadio"] > div {
        width: 100% !important;
    }

    .st-key-top_nav div[role="radiogroup"] {
        display: flex !important;
        width: 100% !important;
        gap: 0.5rem;
        border-bottom: 2px solid #e6e6e6;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    .st-key-top_nav div[role="radiogroup"] label {
        flex: 1 1 0% !important;
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        background-color: #f0f2f6;
        padding: 0.6rem 1rem;
        border-radius: 6px 6px 0 0;
        cursor: pointer;
    }

    .st-key-top_nav div[role="radiogroup"] label p {
        color: #262730;
        font-weight: 500;
        text-align: center;
    }

    .st-key-top_nav div[role="radiogroup"] label:has(input:checked) {
        background-color: #ff4b4b;
    }

    .st-key-top_nav div[role="radiogroup"] label:has(input:checked) p {
        color: white;
    }

    /* hide the little circular radio indicator so it reads as a tab, not a radio button */
    .st-key-top_nav div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container(key="top_nav"):
    page = st.radio(
        label="Navigation",
        options=["Welcome", "Hike Insights", "Explore Hikes", "Predictive Model"],
        horizontal=True,
        key="active_page",
        label_visibility="collapsed"
    )

# ----- Welcome
if page == "Welcome":

    st.header("What is this dashboard all about?")

    st.write("""
    This dashboard explores patterns in hiking experiences using a 
    synthetic dataset of hiking reviews collected across a variety 
    of trails and conditions centered around the Portland metro area. 
    It also examines how factors such as trail difficulty,
    elevation gain, weather conditions, wildlife sightings, and other
    characteristics relate to ratings and experiences.
    """)

    st.write("""
    The analysis is divided into several sections. The Hike Insights
    section provides an overview of patterns across the full dataset.
    The Explore Hikes section allows users to examine individual trails
    and hikes in greater detail. Finally, the Predictive Model section
    demonstrates how a machine learning model can be used to predict
    hiking ratings based on available trail and environmental
    characteristics.
    """)

    st.subheader("About the Data")

    st.write("""
    The original goal of this project was to collect real-world hiking
    reviews from AllTrails. I explored automated web scraping as a way
    to gather review data directly from the website, but encountered
    technical and anti-automation measures that prevented me from
    reliably collecting the data needed for this analysis.

    Rather than abandon the project, I pivoted and created a synthetic
    hiking review dataset designed to resemble the types of data that
    could be collected from real hiking reviews. The dataset was
    generated using a combination of Ollama-based large language models
    and randomized data-generation techniques. Randomization was used
    to simulate realistic variation in characteristics such as trail
    difficulty, weather conditions, wildlife sightings, and hiking
    experiences, while Ollama was used to generate natural-language
    review text that aligned to the random variation in trail characteristics.
    """)

# ----- Hike Insights
elif page == "Hike Insights":

    st.header("Hike Insights")

    st.write("""
    Explore the overall hiking review dataset, including the distribution
    of reviews across hikes and the timeline of hiking activity.
    """)
    # ----- Timeline frequency selection
    timeline_frequency = st.radio(
        "Select Timeline Frequency:",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="timeline_frequency"
    )

    # ----- Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Number of Reviews by Hike")

        if timeline_frequency == "Daily":
            avg_hikes_per_trail = (
                df
                .groupby(["trail_name", "date"])
                .size()
                .groupby("trail_name")
                .mean()
                .reset_index(name="avg_hikes")
                .sort_values("avg_hikes", ascending=True)
            )
            avg_hikes_per_trail['avg_hikes'] = avg_hikes_per_trail['avg_hikes'].round(4)

        elif timeline_frequency == "Weekly":
            avg_hikes_per_trail = (
                df
                .groupby(["trail_name", df["date"].dt.to_period("W")])
                .size()
                .groupby("trail_name")
                .mean()
                .reset_index(name="avg_hikes")
                .sort_values("avg_hikes", ascending=True)
            )
            avg_hikes_per_trail['avg_hikes'] = avg_hikes_per_trail['avg_hikes'].round(4)

        else:
            avg_hikes_per_trail = (
                df
                .groupby(["trail_name", df["date"].dt.to_period("M")])
                .size()
                .groupby("trail_name")
                .mean()
                .reset_index(name="avg_hikes")
                .sort_values("avg_hikes", ascending=True)
            )
            avg_hikes_per_trail['avg_hikes'] = avg_hikes_per_trail['avg_hikes'].round(4)

        fig_reviews = px.bar(
            avg_hikes_per_trail,
            x="avg_hikes",
            y="trail_name",
            orientation="h",
            title="Avg Reviews by Hike",
            labels={
                "avg_hikes": "Avg # of Hikes",
                "trail_name": "Hike Name"
            }
        )

        fig_reviews.update_layout(height=400)

        st.plotly_chart(fig_reviews, use_container_width=True, key="reviews_by_hike_chart")

    with col2:
        st.subheader("Hiking Timeline")

        if timeline_frequency == "Daily":

            timeline_data = (
                df
                .groupby("date")
                .size()
                .reset_index(name="hike_count")
            )

            timeline_title = "Number of Hikes by Day"

        elif timeline_frequency == "Weekly":
            timeline_data = (
                    df
                    .set_index("date")
                    .resample("W")
                    .size()
                    .reset_index(name="hike_count")
                )
            timeline_title = "Number of Hikes by Week"

        else:

            timeline_data = (
                df
                .set_index("date")
                .resample("ME")
                .size()
                .reset_index(name="hike_count")
            )

            timeline_title = "Number of Hikes by Month"

        fig_timeline = px.line(
            timeline_data,
            x="date",
            y="hike_count",
            title=timeline_title,
            labels={
                "date": "Date",
                "hike_count": "# of Hikes"
            }
        )

        fig_timeline.update_layout(height=400)

        st.plotly_chart(
            fig_timeline,
            use_container_width=True,
            key="hiking_timeline_chart"
        )


# ----- Explore Hikes
elif page == "Explore Hikes":

    st.header("Explore Hikes")

    st.write("""
    Select an individual hike to explore its characteristics,
    reviews, and ratings in greater detail.
    """)

    # ----- Hike selection
    list_of_hikes = df['trail_name'].unique()
    hike_selection = st.selectbox(
        label="Select Hike:",
        options=list_of_hikes,
        key="hike_selection"
    )

    df_hike = df[df["trail_name"] == hike_selection]
    df_monthly_counts = df_hike.groupby(["date", "rating"]).size().reset_index(name="count")
    df_monthly_counts["month"] = pd.to_datetime(df_monthly_counts["date"]).dt.to_period("M").dt.to_timestamp()
    df_monthly_counts = df_monthly_counts.groupby(["month", "rating"]).size().reset_index(name="count")

    df_monthly_counts["weighted_rating"] = df_monthly_counts["rating"] * df_monthly_counts["count"]
    df_monthly_summary = df_monthly_counts.groupby("month").agg(
        review_count=("count", "sum"),
        total_rating=("weighted_rating", "sum")
    ).reset_index()

    df_monthly_summary["avg_rating"] = (df_monthly_summary["total_rating"] / df_monthly_summary["review_count"]).round(4)

    timeline_chart = px.scatter(
        df_monthly_summary,
        x="month",
        y="review_count",
        color="avg_rating",
        color_continuous_scale="RdYlGn",
        size="review_count",
        labels={
            "review_count": "# Hikes",
            "month": "Year-Month",
            "avg_rating": "Avg Rating"
        }
    )

    year_starts = pd.date_range(
        start=df_monthly_summary["month"].min(),
        end=df_monthly_summary["month"].max(),
        freq="YS"
    )

    for date in year_starts:
        timeline_chart.add_vline(
            x=date,
            line_dash="dash",
            line_color="white",
            opacity=0.6
        )

    timeline_chart.update_layout(
        height=400,
        coloraxis_colorbar=dict(
                title="Avg Monthly Rating",
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                len=0.8
        )
    )

    st.plotly_chart(
        timeline_chart,
        use_container_width=True,
        key="hike_detail_timeline_chart"
    )

    # ----- Create two columns
    col1, col2 = st.columns(2)

    # ----- Word Cloud
    with col1:

        filtered = df[(df['trail_name'] == hike_selection)]
        text = ' '.join(filtered['review_text'].dropna())

        wc = WordCloud(
            # width=1000,
            # height=500,
            background_color="white"
        ).generate(text)

        fig = px.imshow(
            wc.to_array(),
            title = 'Word Cloud for Hikes'
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="wordcloud_chart"
        )

    # ----- Bear chart
    with col2:
        filtered = df[(df['trail_name'] == hike_selection)]
        filtered = filtered[filtered['actual_bear_sighting'] == True]

        monthly_bear = (
            filtered
            .groupby("month")
            .size()
            .reset_index(name="count")
        )

        all_months = pd.date_range(
            df_hike["month"].min(),
            df_hike["month"].max(),
            freq="MS"
        )

        monthly_bear = (
            monthly_bear
            .set_index("month")
            .reindex(all_months, fill_value=0)
            .rename_axis("month")
            .reset_index()
        )
        fig = px.treemap(
            monthly_bear,
            path=[monthly_bear["month"].dt.strftime("%Y-%m")],
            values="count",
            color="count",
            color_continuous_scale="Reds",
            title='Bear Sighting Frequency',
        )

        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Bear Sightings: %{value}<extra></extra>"
        )

        fig.update_layout(coloraxis_colorbar_title="")

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="bear_sighting_chart"
        )

# ----- Predictive Model
elif page == "Predictive Model":

    st.header("Predictive Model")

    st.write("""
    This section presents the results of a predictive model designed
    to estimate hiking ratings based on trail characteristics and
    environmental conditions.

    The chart on the left shows the forecasted average rating per month for each trail.
    The chart on the right shows the p-values for the parameter estimates in the model
    compared against a 0.05 p-value threshold indiciating statistical significance.  Any 
    bars that fall short of the dashed white line indicate the particular parameter is 
    statistically significant.
    """)

    # ----- Hike selection
    list_of_hikes = df['trail_name'].unique()
    hike_selection = st.selectbox(
        label="Select Hike:",
        options=list_of_hikes,
        key="hike_selection"
    )

    # ----- Create two columns
    col1, col2 = st.columns(2)

    # ----- Forecast Chart
    with col1:
        filtered = model_data[(model_data['trail_name'] == hike_selection)]
        filtered['avg_monthly_rating'] = filtered['avg_monthly_rating'].round(4)
        forecast_chart = px.line(
            filtered,
            x="year_month",
            y="avg_monthly_rating",
            color = 'key',
            color_discrete_map={
                "ACTUAL": "#1f77b4",
                "PRED": "#ff7f0e",
            },
            color_discrete_sequence=["#1f77b4", "#ff7f0e"],
            labels={
                "year_month": "Year-Month",
                "avg_monthly_rating": "Avg Monthly Rating",
                "key": "Key"
            },
            title = 'Average Rating Forecast'
        )


        year_starts = pd.date_range(
            start=pd.to_datetime(filtered["year_month"].min()) + pd.DateOffset(years=1),            
            end=filtered["year_month"].max(),
            freq="YS"
        )

        for year_month in year_starts:
            forecast_chart.add_vline(
                x=year_month,
                line_dash="dash",
                line_color="white",
                opacity=0.6
            )
        forecast_chart.update_layout(
                height=400,
                legend_title_text='',
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )
        forecast_chart.update_yaxes(range=[1, 5])

        st.plotly_chart(
            forecast_chart,
            use_container_width=True
        )

    with col2:
        filtered = model_estimates[(model_estimates['trail_name'] == hike_selection)]
        filtered = filtered.sort_values(by = 'p.value', ascending = False)
        filtered['p.value'] = filtered['p.value'].round(4)

        model_estimates_chart = px.bar(
            filtered,
            x='p.value',
            y='term',
            orientation='h',
            color_discrete_sequence=['#800080'],
            labels={
                "term": "Estimate Name",
                "p.value": "P-value"
            },
            title = 'Model Estimates'
        )

        model_estimates_chart.add_vline(
            x=0.05,
            line_dash="dash",
            line_color="#FFFFFF"
        )

        model_estimates_chart.add_scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color='#FFFFFF', dash='dash'),
            name='Statistical Significance Threshold'
        )

        model_estimates_chart.update_layout(
            height=400,
            legend_title_text='',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        model_estimates_chart.update_xaxes(range=[0, 1])

        st.plotly_chart(
            model_estimates_chart,
            use_container_width=True
        )
