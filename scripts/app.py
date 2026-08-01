import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

# ----- Page configuration
st.set_page_config(
    page_title="Hiking Analysis Dashboard",
    layout="wide"
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

# ----- Sidebar navigation
st.sidebar.title("Hiking Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "Welcome",
        "Hike Insights",
        "Explore Hikes",
        "Predictive Model"
    ])

# ----- Main page title
st.title("Hiking Analysis Dashboard")

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
        ["Daily", "Monthly"],
        horizontal=True
    )

    # ----- Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Number of Reviews by Hike")

        reviews_by_hike = (
            df
            .groupby("trail_name")
            .size()
            .reset_index(name="review_count")
            .sort_values("review_count", ascending=True)
        )

        fig_reviews = px.bar(
            reviews_by_hike,
            x="review_count",
            y="trail_name",
            orientation="h",
            title="Reviews by Hike",
            labels={
                "review_count": "Number of Reviews",
                "trail_name": "Hike"
            }
        )

        fig_reviews.update_layout(height=400)

        st.plotly_chart(fig_reviews, use_container_width=True)

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
            use_container_width=True
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
        label = "Select hike:",
        options = list_of_hikes
    )
    df_hike = df[df["trail_name"] == hike_selection]
    metric1 = df_hike.shape[0]
    metric2 = round(df_hike['rating'].mean(),2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Reviews", f"{metric1}")
    with col2:
        st.metric("Average Rating", f"{metric2}/5")
    with col3:
        st.metric("% Good Weather", f"")
    with col4:
        st.metric("Average Suffer Index", f"")

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
        use_container_width=True
    )

    # ----- Word Cloud
    
    filtered = df[(df['trail_name'] == hike_selection)]
    text = ' '.join(filtered['review_text'].dropna())

    wc = WordCloud(
        #width=1000,
        #height=500,
        background_color="white"
    ).generate(text)
    
    fig = px.imshow(wc.to_array())
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(coloraxis_showscale=False)

    st.plotly_chart(fig, use_container_width=True)
    
    
    bear_filtered = df[(df['trail_name'] == hike_selection) ]
    bear_filtered = bear_filtered[bear_filtered['actual_bear_sighting']==True]

    monthly_bear = (
        bear_filtered
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

    fig = px.bar(
        monthly_bear,
        x="month",
        y="count",
        color="count",
        color_continuous_scale="Reds",
        title="Bear Sightings by Month"
    )

    st.plotly_chart(fig, use_container_width=True)

# ----- Predictive Model 
elif page == "Predictive Model":

    st.header("Predictive Model")

    st.write("""
    This section presents the results of a predictive model designed
    to estimate hiking ratings based on trail characteristics and
    environmental conditions.
    """)