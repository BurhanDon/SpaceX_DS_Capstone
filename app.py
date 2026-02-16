import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import io
import folium
from streamlit_folium import st_folium
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SpaceX Data Science Capstone", page_icon="🚀", layout="wide")

# --- LOAD ASSETS ---
@st.cache_data
def load_data():
    # We load both datasets to show the 'Before' and 'After'
    # Ensure you have both csv files in the folder
    df_raw = pd.read_csv("dataset_part_1.csv") 
    df_clean = pd.read_csv("dataset_part_2.csv")
    return df_raw, df_clean

@st.cache_resource
def load_model():
    return joblib.load("spacex_model.pkl")

try:
    df_raw, df_clean = load_data()
    model = load_model()
except FileNotFoundError:
    st.error("Error: Ensure 'dataset_part_1.csv', 'dataset_part_2.csv', and 'spacex_model.pkl' are in the directory.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Project Roadmap")
section = st.sidebar.radio("Navigate Project Steps:", 
    ["1. Project Overview", 
     "2. Data Collection (Scraping)", 
     "3. Data Wrangling & SQL", 
     "4. Visual Analytics", 
     "5. Predictive Model"])

st.sidebar.markdown(
    "Developed by: [Burhan Siraj](https://www.linkedin.com/in/burhan-siraj/)",
    unsafe_allow_html=True
)

# --- SECTION: PROJECT OVERVIEW ---
if section == "1. Project Overview":
    st.title("SpaceX Falcon 9 First Stage Landing Prediction")
    
    st.markdown("### Executive Summary")
    st.write("""
    The commercial space industry has been revolutionized by SpaceX's ability to reuse the first stage of the Falcon 9 rocket. 
    A standard Falcon 9 launch costs approximately $62 million, significantly less than the $165 million average of traditional competitors. 
    This price advantage is largely derived from the successful recovery and refurbishment of the first-stage boosters.
    
    This project focuses on developing a machine learning pipeline to predict the landing outcome of the Falcon 9 first stage. 
    Accurate predictions of landing success enable competitors and stakeholders to estimate the true cost of a launch and assess the feasibility of bidding against SpaceX for government and commercial contracts.
    """)

    st.divider()

    # --- TWO COLUMN LAYOUT: Problem & Objective ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Problem Statement")
        st.write("""
        SpaceX does not publicly disclose the exact cost savings for every launch. 
        External stakeholders (such as competing aerospace firms or insurance providers) lack a deterministic method to calculate the probability of a successful recovery.
        Without this probability, it is difficult to accurately model the risk-adjusted cost of a launch.
        """)

    with col2:
        st.subheader("Project Objective")
        st.write("""
        The primary objective is to build a binary classification model that predicts whether the first stage will land successfully (1) or fail (0).
        
        **Key Goals:**
        * Extract and clean launch data from multiple sources.
        * Perform exploratory data analysis (EDA) to identify success factors.
        * Develop an interactive dashboard for decision-makers.
        * Train and optimize a predictive model with >80% accuracy.
        """)

    st.divider()

    # --- METHODOLOGY ---
    st.subheader("Methodology")
    st.write("""
    This project follows a standard Data Science lifecycle, divided into four distinct phases:
    """)
    
    with st.container(border=True):
        st.markdown("""
        **1. Data Collection:** Gathering raw data via the SpaceX REST API and web scraping historical records from Wikipedia.
        **2. Data Wrangling:** Cleaning the dataset, handling missing values, and performing One-Hot Encoding on categorical features.
        **3. Exploratory Analysis:** Utilizing SQL for database querying and Visual Analytics (Folium/Plotly) to detect geographical and payload patterns.
        **4. Predictive Modeling:** Training, testing, and tuning four classification algorithms (Logistic Regression, SVM, Decision Tree, KNN) to identify the optimal model.
        """)

    # --- TECHNOLOGIES & SKILLS ---
    st.divider()
    st.subheader("Technologies & Tools")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    with tech_col1:
        st.markdown("""
        * Python
        * Pandas & NumPy
        * Scikit-learn & Scipy
        * BeautifulSoup & Selenium (Scraping)
        """)
        
    with tech_col2:
        st.markdown("""
        * Streamlit (UI)
        * Matplotlib & Seaborn
        * Plotly Dash & Folium (Geospatial)
        * Data Visualization & Statistical Analysis
        """)
        
    with tech_col3:
        st.markdown("""
        * Jupyter Notebooks
        * SQL (SQLite/Db2)
        * Git & GitHub
        * REST APIs
        """)

    # --- CERTIFICATION ---
    st.divider()
    st.info("""
    **Project Context & Personal Journey:**
    This capstone project stands as a testament to my individual dedication and rigorous self-study. 
    While the IBM Data Science Professional Certificate provided the conceptual framework, the execution represents countless hours of independent coding, debugging, and analytical reasoning. 
    
    I developed every stage of this pipeline—from the complexity of web scraping to the precision of machine learning tuning—to not only demonstrate my current proficiency in Python and SQL but also to showcase my relentless drive to explore, learn, and master the depths of the Data Science domain.""")
    
# --- SECTION 2: WEB SCRAPING ---
elif section == "2. Data Collection (Scraping)":
    st.title("Phase 1: Data Collection Pipeline")
    
    st.markdown("""
    ### Methodology
    To ensure a robust dataset, I employed a dual-source strategy. Data was gathered from the **SpaceX REST API** to obtain standardized launch metrics, and **Web Scraping** was utilized to extract historical launch records directly from Wikipedia. This approach ensures comprehensive coverage of mission outcomes, payload details, and booster landing statuses.
    """)

    # --- Placeholders: Launch Visuals ---
    col_media1, col_media2 = st.columns(2)
    with col_media1:
        st.markdown("**Figure 1: Falcon 9 First Stage Landing**")
        st.image(
            "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/landing_1.gif",
            caption="Falcon 9 first-stage booster performing vertical landing"
        )
    
    with col_media2:
        st.markdown("**Figure 2: Falcon 9 Liftoff Sequence**")
        st.image(
            "https://media.giphy.com/media/3og0IvAdrwbryA5sLm/giphy.gif",
            caption="Falcon 9 launch and ascent"
        )

    st.divider()

    # --- PART A: API INTERACTION ---
    st.subheader("A. SpaceX API Extraction")
    st.markdown("""
    I developed a Python script to interface with the SpaceX REST API. The script iterates through the launch endpoints, filtering for Falcon 9 boosters and extracting critical telemetry data including Core Serial Numbers, Flight Numbers, and Landing Outcomes.
    """)
    
    with st.expander("View API Extraction Code", expanded=False):
        st.code("""
# Requesting data from SpaceX API
spacex_url = "https://api.spacexdata.com/v4/launches/past"
response = requests.get(spacex_url)
data = pd.json_normalize(response.json())

# Filtering for Falcon 9 only
data = data[data['rocket'] == 'Falcon 9']

# Extracting Core and Payload details
for core in data['cores']:
    if core['core'] != None:
        response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
        BoosterVersion.append(response['serial'])
        Outcome.append(str(core['landing_success']) + ' ' + str(core['landing_type']))
        """, language="python")

    # --- Placeholders: In-Flight Visuals ---
    col_media3, col_media4 = st.columns(2)
    with col_media3:
        st.markdown("**Figure 3: Booster Recovery Attempt**")
        st.image(
            "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/crash.gif",
            caption="Falcon 9 booster recovery scenario"
        )
    
    with col_media4:
        st.markdown("**Figure 4: Falcon 9 Vehicle Family**")
        st.image(
            "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_1_L2/images/Falcon9_rocket_family.svg",
            caption="Falcon 9 rocket variants and configurations"
        )
    st.divider()

    # --- PART B: WEB SCRAPING ---
    st.subheader("B. Web Scraping Implementation")
    st.markdown("""
    Complementing the API data, I built a custom web scraper using `BeautifulSoup`. This script parses the HTML structure of the Falcon 9 Wikipedia page to extract tabular data for historical cross-referencing. It specifically targets the `wikitable plainrowheaders` class to isolate launch records.
    """)

    st.code("""
# Parsing Wikipedia HTML
soup = BeautifulSoup(response.content, 'html.parser')

# Iterating through HTML tables
for table in soup.find_all('table', "wikitable plainrowheaders"):
    for rows in table.find_all("tr"):
        if rows.th:
            if rows.th.string:
                flight_number = rows.th.string.strip()
                flag = flight_number.isdigit()
        else:
            flag = False
        
        # Extracting Table Cells
        if flag:
            row = rows.find_all('td')
            launch_dict['Flight No.'].append(flight_number)
            launch_dict['Launch site'].append(row[2].a.string)
            launch_dict['Payload'].append(row[3].a.string)
            launch_dict['Orbit'].append(row[5].a.string)
    """, language="python")

    # --- Placeholders: Landing/Crash Visuals ---
    st.markdown("**Figure 5: Wikipedia Launch Data Table**")
    st.image(
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_1_L2/images/falcon9-launches-wiki.png",
        caption="Wikipedia Falcon 9 launch history table",
        use_container_width=True
    )

    # --- PART C: RAW DATASET PREVIEW ---
    st.subheader("C. Data Collection Output")
    st.markdown("""
    The resulting dataset aggregates all Falcon 9 launches. This raw CSV file serves as the foundation for the subsequent data wrangling and machine learning phases.
    """)
    
    # Displaying the dataframe
    st.dataframe(df_raw.head(10))
    
    # Download Button for the REAL file
    with open("spacex_web_scraped.csv", "rb") as file:
        btn = st.download_button(
            label="Download Raw Scraped Data (CSV)",
            data=file,
            file_name="spacex_web_scraped.csv",
            mime="text/csv"
        )

# --- SECTION: WRANGLING & EDA ---
elif section == "3. Data Wrangling & SQL":
    st.title("Phase 2: Data Wrangling & Exploratory Analysis")
    
    st.markdown("""
    ### Methodology
    Raw data requires rigorous cleaning and interrogation before modeling. I utilized **SQL** for querying the database to answer specific business questions and **Pandas** to refine the dataset for the machine learning pipeline.
    """)

    # --- PART A: SQL EXPLORATION ---
    st.header("A. Database Querying (SQL)")
    st.markdown("""
    I established a connection to a **SQLite** database to execute complex queries. This allowed me to derive immediate insights regarding launch sites and payload capacities without loading the entire dataset into memory.
    """)
    with open("my_data1.db", "rb") as file:
        st.download_button(
            label="Download SQL Dataset (my_data1.db)",
            data=file,
            file_name="my_data1.db",
            mime="application/octet-stream"
        )
    # --- SQL Task Showcase ---
    col_sql1, col_sql2 = st.columns(2)
    
    with col_sql1:
        st.subheader("Task: Identify High-Traffic Launch Sites")
        st.write("Query to find unique launch sites in the mission registry.")
        st.code("""
SELECT DISTINCT Launch_Site 
FROM SPACEXTABLE;
        """, language="sql")
        st.markdown("**Result:** identified 4 primary sites including *CCAFS LC-40* and *KSC LC-39A*.")

    with col_sql2:
        st.subheader("Task: Calculate NASA's Total Payload")
        st.write("Query to sum the total payload mass transported for NASA (CRS).")
        st.code("""
SELECT SUM(PAYLOAD_MASS__KG_) 
FROM SPACEXTABLE 
WHERE Customer = 'NASA (CRS)';
        """, language="sql")
        st.markdown("**Result:** NASA (CRS) has transported **45,596 kg** of payload.")

    st.markdown("---")
    
    st.subheader("Advanced SQL: Successful Drone Ship Landings")
    st.write("Filtering for boosters that landed successfully on drone ships with a payload between 4,000kg and 6,000kg.")
    st.code("""
SELECT Booster_Version 
FROM SPACEXTABLE 
WHERE Landing_Outcome = 'Success (drone ship)' 
AND PAYLOAD_MASS__KG_ BETWEEN 4000 AND 6000;
    """, language="sql")
    
    # Placeholder for SQL Notebook Screenshot
    st.image("assets/Screenshot_1.png", caption="[SQL Notebook Execution]")
    
    # Download Button for SQL Data
    with open("Spacex.csv", "rb") as file:
        st.download_button(
            label="Download SQL Dataset (Spacex.csv)",
            data=file,
            file_name="Spacex.csv",
            mime="text/csv"
        )

    st.divider()
    # --- PART B: DETAILED VISUAL ANALYTICS ---
    st.header("B. Pandas Visual Analytics")
    st.markdown("Uncovering patterns using Python visualization libraries.")

    # We use tabs to keep the UI clean while showing many charts
    tab1, tab2, tab3 = st.tabs(["Launch Sites", "Orbits", "Yearly Trend"])

    with tab1:
        st.subheader("1. Flight Number vs. Launch Site")
        st.markdown("Do specific launch sites get used more as the flight number increases?")
        fig1 = px.scatter(df_clean, x="FlightNumber", y="LaunchSite", color="Class",
                          title="Flight Number vs Launch Site (0=Fail, 1=Success)",
                          color_discrete_map={0: 'red', 1: 'green'})
        st.plotly_chart(fig1, use_container_width=True)
        st.image("assets/FlightNumber-vs-LaunchSite.png", caption="[Flight Number Vs Launch Site]")
        st.image("assets/FlightNumber-vs-PayloadMass.png", caption="[Flight Number Vs Pay Load Mass]")

        st.subheader("2. Payload Mass vs. Launch Site")
        st.markdown("Are there heavy payloads launched from specific sites?")
        fig2 = px.scatter(df_clean, x="PayloadMass", y="LaunchSite", color="Class",
                          title="Payload Mass vs Launch Site",
                          color_discrete_map={0: 'red', 1: 'green'})
        st.plotly_chart(fig2, use_container_width=True)
        st.image("assets/PayloadMass-vs-LaunchSite.png", caption="[Pay Load Mass Vs Launch Site]")

    with tab2:
        st.subheader("3. Success Rate by Orbit Type")
        st.markdown("Which orbits have the highest success rate? (ES-L1, GEO, HEO, SSO have high success rates)")
        orbit_success = df_clean.groupby('Orbit')['Class'].mean().reset_index()
        fig3 = px.bar(orbit_success, x='Orbit', y='Class', color='Class',
                      title="Success Rate per Orbit Type")
        st.plotly_chart(fig3, use_container_width=True)
        st.image("assets/Success Rate by Orbit Type.png", caption="[Success Rate by Orbit Type]")

        st.subheader("4. Flight Number vs. Orbit Type")
        st.markdown("How has the target orbit changed over time?")
        fig4 = px.scatter(df_clean, x="FlightNumber", y="Orbit", color="Class",
                          title="Flight Number vs Orbit Type",
                          color_discrete_map={0: 'red', 1: 'green'})
        st.plotly_chart(fig4, use_container_width=True)
        st.image("assets/Relation ship bw Flight number and Orbit type.png", caption="[elation ship bw Flight number and Orbit type]")
        
        st.subheader("5. Payload Mass vs. Orbit Type")
        st.markdown("Heavy payloads (Polar, ISS) vs Lighter payloads (LEO).")
        fig5 = px.scatter(df_clean, x="PayloadMass", y="Orbit", color="Class",
                          title="Payload Mass vs Orbit Type",
                          color_discrete_map={0: 'red', 1: 'green'})
        st.plotly_chart(fig5, use_container_width=True)
        st.image("assets/Relation ship bw PayloadMass and Orbit type.png", caption="[Relation ship bw PayloadMass and Orbit type]")

    with tab3:
        st.subheader("6. Launch Success Yearly Trend")
        # Extract Year if not already in df
        df_trend = df_clean.copy()
        # Ensure Date is datetime
        # If your csv has 'Date', we extract year. If not, skip or use 'Year' column if exists
        try:
            df_trend['Year'] = pd.to_datetime(df_clean['Date']).dt.year
            yearly_success = df_trend.groupby('Year')['Class'].mean().reset_index()
            fig6 = px.line(yearly_success, x='Year', y='Class', title="Success Rate Trend (2010-2020)")
            st.plotly_chart(fig6, use_container_width=True)
            st.image("assets/Payloadmass_X_Orbit_type_year_trend.png", caption="[Payloadmass X Orbit type year Trend]")
        except Exception as e:
            st.warning("Date column not found for Yearly Trend analysis.")
            # st.error(e)

    st.divider()

    # --- PART C: FEATURE ENGINEERING ---
    st.header("C. Feature Engineering")
    st.markdown("""
    Machine Learning algorithms cannot understand text. I used **One-Hot Encoding** to convert categorical variables 
    (like `Orbit` and `LaunchSite`) into numerical columns.
    """)

    with st.expander("View One-Hot Encoding Code"):
        st.code("""
# Selecting features
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]

# Applying One-Hot Encoding
features_one_hot = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])

# Casting to float64
features_one_hot.astype('float64')
        """, language="python")

    st.write(f"**Result:** Dataset dimensions expanded to {df_clean.shape} to capture all categorical variances.")

    # Download Button for the processed data
    with open("dataset_part_3.csv", "rb") as file:
        st.download_button(
            label="Download Feature Engineered Data (dataset_part_3.csv)",
            data=file,
            file_name="dataset_part_3.csv",
            mime="text/csv"
        )

# --- SECTION 4: VISUAL ANALYTICS ---
# --- SECTION: INTERACTIVE VISUAL ANALYTICS ---
elif section == "4. Visual Analytics":
    st.title("Phase 3: Interactive Visual Analytics")
    
    st.markdown("""
    ### Objective
    This module integrates geospatial analysis and interactive dashboards to visualize launch outcomes. 
    I developed two key components:
    1.  **Geospatial Analysis (Folium):** To analyze the proximity of launch sites to critical infrastructure (coastlines, railways).
    2.  **Interactive Dashboard (Plotly):** To allow stakeholders to filter data by Launch Site and Payload Mass.
    """)

    # --- PART A: GEOSPATIAL ANALYSIS (FOLIUM) ---
    st.header("A. Geospatial Analysis")
    st.markdown("""
    Using **Folium**, I mapped all launch sites and added marker clusters to visualize Success/Failure density. 
    I also calculated distances to coastlines to understand safety protocols.
    """)

    # 1. Create the Map
    # Filter for necessary columns
    spacex_df = df_clean  # Using the cleaned dataframe loaded earlier
    
    # Map Coordinates for Launch Sites
    site_coords = {
        'CCAFS LC-40': [28.562302, -80.577356],
        'CCAFS SLC-40': [28.563197, -80.576820],
        'KSC LC-39A': [28.573255, -80.646895],
        'VAFB SLC-4E': [34.632834, -120.610745]
    }

    # Initialize Map centered on CCAFS
    m = folium.Map()

    # Add Launch Sites
    for site, coords in site_coords.items():
        folium.Circle(
            location=coords,
            radius=1000,
            color='#000000',
            fill=True
        ).add_child(folium.Popup(site)).add_to(m)
        
        folium.Marker(
            location=coords,
            icon=folium.DivIcon(html=f"""<div style="font-family: courier new; color: blue">{site}</div>""")
        ).add_to(m)

    # Add Marker Clusters for Success/Fail
    from folium.plugins import MarkerCluster
    marker_cluster = MarkerCluster().add_to(m)

    for index, row in spacex_df.iterrows():
        # Define color: Green for Success (1), Red for Failure (0)
        marker_color = 'green' if row['Class'] == 1 else 'red'
        
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],  # Ensure these columns exist in your CSV
            icon=folium.Icon(color='white', icon_color=marker_color),
            popup=f"Payload: {row['PayloadMass']} kg"
        ).add_to(marker_cluster)
    m.fit_bounds(m.get_bounds())
    # Render Map in Streamlit
    st.subheader("Interactive Launch Map")
    st.caption("Green markers indicate success; Red markers indicate failure. Click clusters to zoom.")
    st_folium(m, width=700, height=500)

    st.info("Analysis: Most launch sites are located near coastlines to minimize risk to populated areas in case of launch failure.")

    st.divider()

    # --- PART B: PLOTLY DASHBOARD (Recreated from Dash) ---
    st.header("B. Dynamic Launch Dashboard")
    st.markdown("This section replicates the functionality of the **Plotly Dash** application I built, allowing for real-time data filtering.")

    # Load Dashboard Specific Data (if different, otherwise use df_clean)
    # If you have 'spacex_launch_dash.csv', uncomment the next line:
    # dash_df = pd.read_csv("spacex_launch_dash.csv") 
    dash_df = df_clean # Fallback to using the main cleaned dataset

    # --- INPUT WIDGETS ---
    col_controls1, col_controls2 = st.columns([1, 2])
    
    with col_controls1:
        # TASK 1: Launch Site Dropdown
        site_list = dash_df['LaunchSite'].unique().tolist()
        site_list.insert(0, 'All Sites')
        
        selected_site = st.selectbox(
            "Select Launch Site:", 
            options=site_list,
            index=0
        )

    with col_controls2:
        # TASK 3: Payload Slider
        min_payload = int(dash_df['PayloadMass'].min())
        max_payload = int(dash_df['PayloadMass'].max())
        
        payload_range = st.slider(
            "Select Payload Range (Kg):",
            min_value=0,
            max_value=10000,
            value=(min_payload, max_payload)
        )

    # --- FILTERING LOGIC ---
    # Filter by Payload
    filtered_df = dash_df[
        (dash_df['PayloadMass'] >= payload_range[0]) & 
        (dash_df['PayloadMass'] <= payload_range[1])
    ]

    # Filter by Site (for Pie Chart logic)
    if selected_site == 'All Sites':
        pie_data = filtered_df
        pie_fig = px.pie(
            pie_data, 
            names='LaunchSite', 
            values='Class', # Counting successes
            title='Total Successful Launches by Site'
        )
        
        scatter_data = filtered_df
        scatter_title = 'Correlation between Payload and Success for All Sites'
    else:
        # Specific Site Logic
        pie_data = filtered_df[filtered_df['LaunchSite'] == selected_site]
        # For specific site, we show Success vs Failure counts
        site_counts = pie_data['Class'].value_counts().reset_index()
        site_counts.columns = ['Class', 'Count']
        site_counts['Outcome'] = site_counts['Class'].map({1: 'Success', 0: 'Failure'})
        
        pie_fig = px.pie(
            site_counts,
            names='Outcome',
            values='Count',
            title=f'Success vs. Failed Launches for {selected_site}',
            color='Outcome',
            color_discrete_map={'Success': 'green', 'Failure': 'red'}
        )
        
        scatter_data = pie_data
        scatter_title = f'Correlation between Payload and Success for {selected_site}'

    # --- CHARTS ---
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # TASK 2: Pie Chart Display
        st.plotly_chart(pie_fig, use_container_width=True)
        
    with col_chart2:
        # TASK 4: Scatter Chart Display
        scatter_fig = px.scatter(
            scatter_data,
            x='PayloadMass', 
            y='Class',
            color='BoosterVersion', # Ensure this column exists in your CSV
            title=scatter_title,
            hover_data=['LaunchSite']
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    # Download Button for Dashboard Data
    with open("spacex_launch_dash.csv", "rb") as file:
        st.download_button(
            label="Download Dashboard Data (csv)",
            data=file,
            file_name="spacex_launch_dash.csv",
            mime="text/csv"
        )



# --- SECTION: MACHINE LEARNING ---
elif section == "5. Predictive Model":
    st.title("Phase 4: Predictive Machine Learning")
    
    st.markdown("""
    ### Objective
    The final goal was to build a binary classification model to predict the landing outcome (1=Success, 0=Failure).
    I implemented and compared four standard classification algorithms, optimizing each using **GridSearchCV** to find the best hyperparameters.
    """)

    # --- PART A: MODEL EVALUATION & COMPARISON ---
    st.header("A. Model Performance Comparison")
    st.markdown("""
    After standardizing the data (StandardScaler) and splitting it into Training/Test sets, I trained the following models. 
    The bar chart below compares their accuracy on the Test set.
    """)

    # UPDATED SCORES BASED ON YOUR LATEST RESULTS
    model_performance = {
        'Model': ['Logistic Regression', 'Support Vector Machine (SVM)', 'Decision Tree', 'K-Nearest Neighbors (KNN)'],
        'Accuracy': [0.833, 0.833, 0.778, 0.833] 
    }
    df_perf = pd.DataFrame(model_performance)

    # 1. Comparison Bar Chart
    fig_perf = px.bar(
        df_perf, 
        x='Model', 
        y='Accuracy', 
        color='Accuracy',
        title="Model Accuracy Comparison (Test Set)",
        range_y=[0.6, 1.0], 
        text_auto=True,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    st.info("""
    **Insight:** Logistic Regression, SVM, and KNN tied for the highest performance with **83.3%** accuracy. 
    The **Decision Tree** achieved the highest Training accuracy (87.7%) but dropped to **77.8%** on the Test set, indicating it suffered from **overfitting** (memorizing the training data rather than generalizing).
    """)

    # 2. Explanations (Text Only)
    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        st.subheader("Top Performer: Logistic Regression")
        st.markdown("""
        Since Logistic Regression, SVM, and KNN performed equally well, **Logistic Regression** is selected as the ideal model for production. It offers the same high accuracy (83.3%) but with faster execution time and simpler interpretability than SVM or KNN.
        
        **Best Hyperparameters found:**
        * C: `0.01`
        * Solver: `lbfgs`
        * Penalty: `l2`
        """)
    
    with col_eval2:
        st.subheader("Confusion Matrix Analysis")
        st.markdown("""
        The Confusion Matrix allows us to see where the model makes mistakes.
        * **True Positives (TP):** Correctly predicted landings.
        * **True Negatives (TN):** Correctly predicted crashes.
        * **False Positives (FP):** Model predicted success, but it crashed (High Risk).
        * **False Negatives (FN):** Model predicted crash, but it landed (Missed Opportunity).
        """)

    # 3. Full Width Confusion Matrix Grid (2x2)
    st.markdown("---")
    st.subheader("Confusion Matrix Comparison")
    
    # Row 1
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.image("assets/Logistic Regression Confusion Matrix.png", caption="Logistic Regression (83.3%)", use_container_width=True)
    with row1_col2:
        st.image("assets/Support Vector Machine Confusion Matrix.png", caption="SVM (83.3%)", use_container_width=True)

    # Row 2
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.image("assets/Decision Tree Confusion Matrix.png", caption="Decision Tree (77.8%)", use_container_width=True)
    with row2_col2:
        st.image("assets/KNN Confusion Matrix.png", caption="KNN (83.3%)", use_container_width=True)

    st.divider()

    # --- PART B: THE "BLACK BOX" CODE ---
    st.header("B. Hyperparameter Tuning Code")
    st.markdown("I used `GridSearchCV` to exhaustively search for the best parameters. This ensures the model is mathematically optimized, not just guessed.")
    
    with st.expander("View GridSearchCV Logic"):
        st.code("""
# Parameter Grid for Logistic Regression
parameters = {
    'C': [0.01, 0.1, 1],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

# Grid Search Execution
lr = LogisticRegression()
gscv = GridSearchCV(lr, parameters, scoring='accuracy', cv=10)
logreg_cv = gscv.fit(X_train, Y_train)

print("Tuned Hyperparameters :", logreg_cv.best_params_)
print("Accuracy :", logreg_cv.best_score_)
        """, language="python")

    st.divider()

    # --- PART C: LIVE PREDICTION DEMO ---
    st.header("C. Live Landing Prediction")
    st.markdown("""
    **Test the Model:** Configure a hypothetical Falcon 9 launch below. The model will process your inputs in real-time and predict the landing outcome.
    """)

    # Container for inputs
    with st.container(border=True):
        col_input1, col_input2, col_input3 = st.columns(3)
        
        with col_input1:
            st.markdown("**1. Payload & Orbit**")
            payload = st.number_input("Payload Mass (kg)", min_value=0, max_value=20000, value=6104, step=100)
            orbit = st.selectbox("Orbit Type", ['LEO', 'ISS', 'PO', 'GTO', 'ES-L1', 'SSO', 'HEO', 'MEO', 'VLEO', 'SO', 'GEO'])
            
        with col_input2:
            st.markdown("**2. Launch Site & Flights**")
            site = st.selectbox("Launch Site", ['CCAFS LC-40', 'VAFB SLC-4E', 'KSC LC-39A', 'CCAFS SLC-40'])
            flights = st.number_input("Flight Number", min_value=1, max_value=100, value=1)
            reused_count = st.number_input("Times Booster Reused", min_value=0, max_value=20, value=0)

        with col_input3:
            st.markdown("**3. Booster Configuration**")
            grid_fins = st.radio("Grid Fins?", [True, False], index=0)
            legs = st.radio("Landing Legs?", [True, False], index=0)
            reused = st.radio("Reused Booster?", [True, False], index=1) # Default False
            block = st.slider("Block Version", 1.0, 5.0, 5.0, 1.0)

        # Predict Button
        predict_btn = st.button("🚀 Run Prediction Sequence", type="primary", use_container_width=True)

    # Result Display
    if predict_btn:
        # Create input dataframe matching training features
        input_data = pd.DataFrame({
            'PayloadMass': [payload],
            'Orbit': [orbit],
            'LaunchSite': [site],
            'Flights': [flights],
            'GridFins': [grid_fins],
            'Reused': [reused],
            'Legs': [legs],
            'Block': [block],
            'ReusedCount': [reused_count]
        })
        
        # Run Prediction (Using the loaded model, which we assume is the best performer)
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] # Probability of Success
        
        st.markdown("---")
        st.subheader("Prediction Report")
        
        if prediction == 1:
            st.success(f"## SUCCESSFUL LANDING PREDICTED")
            st.metric(label="Confidence Level", value=f"{prob:.1%}")
            st.write(f"The model is **{prob:.1%}** sure this mission will result in a successful recovery.")
        else:
            st.error(f"## CRASH LANDING PREDICTED")
            st.metric(label="Confidence Level", value=f"{1-prob:.1%}")

            st.write(f"The model predicts a failure with **{1-prob:.1%}** confidence. Re-evaluate mission parameters.")

