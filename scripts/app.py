import streamlit as st
import pandas as pd
import plotly.express as px

# ----- Page configuration
st.set_page_config(
    page_title="Hiking Analysis Dashboard",
    layout="wide"
)


# ----- Load data

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/statzenthusiast921/hike_monitoring/refs/heads/main/data/synthetic_hiking_reviews.csv"
    return pd.read_csv(url)


df = load_data()
df["date"] = pd.to_datetime(df["date"])


# ----- Sidebar navigation
st.sidebar.title("Hiking Analysis")

page = st.sidebar.radio(
    "",
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

        fig_reviews.update_layout(height=600)

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

        fig_timeline.update_layout(height=600)

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

# ----- Predictive Model 
elif page == "Predictive Model":

    st.header("Predictive Model")

    st.write("""
    This section presents the results of a predictive model designed
    to estimate hiking ratings based on trail characteristics and
    environmental conditions.
    """)