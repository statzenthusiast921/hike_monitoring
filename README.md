# Hiking Analysis

## Description

The purpose of this project was to determine:

- how reviews for certain hikes change over time
- if we can see common themes in hiking reviews (eg: bear sightings increasing)
- if we can build a simple model to predict review ratings (1-5)
- build an interactive dashboard using a new framework

## Data
The original goal of this project was to collect real-world hiking reviews from AllTrails. I explored automated web scraping as a way to gather review data directly from the website, but encountered technical and anti-automation measures that prevented me from reliably collecting the data needed for this analysis.

Rather than abandon the project, I pivoted and created a synthetic hiking review dataset designed to resemble the types of data that could be collected from real hiking reviews. The dataset was generated using a combination of [Ollama-based large language models](https://ollama.com/) and randomized data-generation techniques. Randomization was used to simulate realistic variation in characteristics such as trail difficulty, weather conditions, wildlife sightings, and hiking experiences, while Ollama was used to generate natural-language review text that aligned to the random variation in trail characteristics.

### App
Click [here](https://synthetic-hike-monitoring-jz-app.streamlit.app/) to view the app.

## Challenges
All apps/dashboards up to this point in my personal data project history have been built with either Tableau, R/Shiny, or the Dash framework with Python.  I wanted to get experience with a new framework to broaden my skillset so I took on learning [Streamlit](https://streamlit.io/).  While Streamlit has less control over how a dashboard looks and functions vs. Dash, it was much simpler and quicker to build something functional.  Also, the deployment process took just a few tries to succeed vs. the million attempts it usually takes before succeeding with Dash.



