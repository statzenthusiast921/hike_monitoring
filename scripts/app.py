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

    url = "https://raw.githubusercontent.com/statzenthusiast921/hike_monitoring/refs/heads/main/data/synthetic_hiking_reviews.csv"
    df = pd.read_csv(url)

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

    return df

df = load_data()

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
    """)

    st.write("""
    The dashboard examines how factors such as trail difficulty,
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
    review text consistent with those characteristics.
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
        "Select timeline frequency:",
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

        fig_reviews = px.bar(
            avg_hikes_per_trail,
            x="avg_hikes",
            y="trail_name",
            orientation="h",
            title="Avg Reviews by Hike",
            labels={
                #"review_count": "Number of Reviews",
                "trail_name": "Hike"
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
                "hike_count": "Number of Hikes"
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
        label="Select hike:",
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

    df_monthly_summary["avg_rating"] = df_monthly_summary["total_rating"] / df_monthly_summary["review_count"]

    timeline_chart = px.scatter(
        df_monthly_summary,
        x="month",
        y="review_count",
        color="avg_rating",
        color_continuous_scale="RdYlGn",
        size="review_count"
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

    timeline_chart.update_layout(height=400)

    st.plotly_chart(
        timeline_chart,
        use_container_width=True,
        key="hike_detail_timeline_chart"
    )

    # ----- Create two columns
    col1, col2 = st.columns(2)

    # Word Cloud
    with col1:

        filtered = df[(df['trail_name'] == hike_selection)]
        text = ' '.join(filtered['review_text'].dropna())

        wc = WordCloud(
            # width=1000,
            # height=500,
            background_color="white"
        ).generate(text)

        fig = px.imshow(wc.to_array())
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="wordcloud_chart"
        )

    # Bear chart
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
            color_continuous_scale="Reds"
        )

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
    """)