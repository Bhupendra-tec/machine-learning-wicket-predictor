from pathlib import Path
import glob

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="NPL Cricket Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "NPL_final.csv"

MODEL_DIR = BASE_DIR / "models" / "decision_tree"

MODEL_PATH = MODEL_DIR / "decision_tree_wicket_model.pkl"

COMPARISON_PATH = (
    BASE_DIR
    / "models"
    / "comparison"
    / "model_comparison.csv"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    div[data-testid="stMetric"] {
        background-color: #172338;
        border: 1px solid #293b55;
        padding: 18px;
        border-radius: 12px;
    }

    .stButton button,
    .stFormSubmitButton button {
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


NUMERICAL_FEATURES = [
    "inning",
    "over_number",
    "balls_before",
    "runs_before",
    "wickets_before",
    "previous_ball_runs",
    "previous_ball_wicket"
]

CATEGORICAL_FEATURES = [
    "batting_team",
    "bowling_team",
    "batsman",
    "bowler",
    "batting_style",
    "bowling_style",
    "bowling_arm"
]

FEATURE_COLUMNS = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES
)


@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "match_id",
        "batting_team",
        "bowling_team",
        "batsman",
        "bowler",
        "batting_style",
        "bowling_style",
        "bowling_arm",
        "dismissal_kind",
        "total_runs"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    # Create target if it does not exist
    if "is_wicket" not in data.columns:

        data["is_wicket"] = (
            data["dismissal_kind"].notna()
            & data["dismissal_kind"].ne("Retired Hurt")
        ).astype(int)

    return data


@st.cache_resource
def load_model():

    # First try the expected model filename
    if MODEL_PATH.exists():

        selected_path = MODEL_PATH

    else:

        # Fallback: find available pickle files
        model_files = glob.glob(
            str(MODEL_DIR / "*.pkl")
        )

        if not model_files:

            raise FileNotFoundError(
                f"No trained model found in {MODEL_DIR}"
            )

        selected_path = max(
            model_files,
            key=lambda path: Path(path).stat().st_mtime
        )

    trained_model = joblib.load(selected_path)

    return trained_model, selected_path



try:

    df = load_dataset()

    model, loaded_model_path = load_model()

except Exception as error:

    st.error("Application could not start.")

    st.exception(error)

    st.stop()



if not hasattr(model, "predict_proba"):

    st.error(
        "Your saved model does not support predict_proba(). "
        "Please load the complete Decision Tree Pipeline."
    )

    st.stop()



if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []

if "latest_prediction" not in st.session_state:

    st.session_state.latest_prediction = None


def get_options(column):

    if column not in df.columns:

        return []

    values = (
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values)


def make_prediction(input_data):

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS
    )

    prediction = int(
        model.predict(input_df)[0]
    )

    probabilities = model.predict_proba(
        input_df
    )[0]

    classes = list(model.classes_)

    if 1 not in classes:

        raise ValueError(
            "The trained model does not contain the wicket class."
        )

    wicket_index = classes.index(1)

    wicket_probability = float(
        probabilities[wicket_index]
    )

    return prediction, wicket_probability


def display_metric(title, value):

    st.metric(
        label=title,
        value=value
    )


with st.sidebar:

    st.title("🏏 NPL Intelligence")

    st.caption(
        "Experimental ML Prediction System"
    )

    st.divider()

    st.subheader("NAVIGATION")

    page = st.radio(
        "Choose a page",
        [
            "Home",
            "Wicket Predictor",
            "Analytics",
            "Prediction History"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("Model Information")

    st.success("Decision Tree Model")

    st.caption(
        "Trained using NPL ball-by-ball data."
    )

    st.caption(
        f"Dataset: {len(df):,} deliveries"
    )

    st.caption(
        f"Model: {Path(loaded_model_path).name}"
    )

    st.divider()

    st.caption(
        "Nepal Premier League | Ball-by-Ball Analytics"
    )


if page == "Home":

    st.caption("NEPAL PREMIER LEAGUE")

    st.title("🏏 NPL Cricket Intelligence")

    st.markdown(
        """
        Explore cricket statistics and estimate wicket
        probabilities using machine learning.

        **Nepal Premier League | Ball-by-Ball Analytics**
        """
    )

    st.success(
        "EXPERIMENTAL MACHINE LEARNING PREDICTION SYSTEM"
    )

    st.divider()

    # Dataset statistics
    total_deliveries = len(df)

    total_wickets = int(
        df["is_wicket"].sum()
    )

    total_matches = df["match_id"].nunique()

    total_players = pd.concat(
        [
            df["batsman"],
            df["bowler"]
        ]
    ).dropna().nunique()

    # Metric cards
    st.subheader("NPL Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        display_metric(
            "Total Deliveries",
            f"{total_deliveries:,}"
        )

    with col2:

        display_metric(
            "Total Wickets",
            f"{total_wickets:,}"
        )

    with col3:

        display_metric(
            "Total Matches",
            f"{total_matches:,}"
        )

    with col4:

        display_metric(
            "Total Players",
            f"{total_players:,}"
        )

    st.divider()

    # Platform information
    st.subheader("Explore the Platform")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🎯 Wicket Predictor")

        st.write(
            """
            Enter a cricket match situation, select batting
            and bowling players, and estimate the probability
            of a wicket on the next delivery.
            """
        )

    with col2:

        st.markdown("### 📊 Cricket Analytics")

        st.write(
            """
            Explore team statistics, wicket distribution,
            batting performance, and machine learning
            model comparisons.
            """
        )

    st.divider()

    # Wicket distribution
    st.subheader("Delivery Outcome Distribution")

    wicket_counts = (
        df["is_wicket"]
        .map({
            0: "No Wicket",
            1: "Wicket"
        })
        .value_counts()
        .reset_index()
    )

    wicket_counts.columns = [
        "Result",
        "Deliveries"
    ]

    fig = px.pie(
        wicket_counts,
        names="Result",
        values="Deliveries",
        hole=0.55,
        title="Wicket vs No Wicket"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


elif page == "Wicket Predictor":

    st.title("🎯 Wicket Probability Predictor")

    st.write(
        """
        Enter the current match situation and get an estimated
        probability of a wicket on the next delivery.
        """
    )

    st.warning(
        "This is an experimental machine learning estimate, "
        "not a guaranteed prediction."
    )

    st.divider()

    # Dropdown options
    batting_teams = get_options("batting_team")

    bowling_teams = get_options("bowling_team")

    batsmen = get_options("batsman")

    bowlers = get_options("bowler")

    batting_styles = get_options("batting_style")

    bowling_styles = get_options("bowling_style")

    bowling_arms = get_options("bowling_arm")

    required_options = [
        batting_teams,
        bowling_teams,
        batsmen,
        bowlers,
        batting_styles,
        bowling_styles,
        bowling_arms
    ]

    if not all(required_options):

        st.error(
            "One or more required dataset columns have no options."
        )

        st.stop()


    with st.form("prediction_form"):

        st.subheader("1. Match Information")

        col1, col2 = st.columns(2)

        with col1:

            batting_team = st.selectbox(
                "Batting Team",
                batting_teams
            )

            inning = st.selectbox(
                "Innings",
                [1, 2]
            )

            current_over = st.number_input(
                "Current Over",
                min_value=0,
                max_value=19,
                value=0,
                step=1
            )

            ball_number = st.selectbox(
                "Current Ball",
                [1, 2, 3, 4, 5, 6]
            )

        with col2:

            bowling_team = st.selectbox(
                "Bowling Team",
                bowling_teams
            )

            current_score = st.number_input(
                "Current Team Score",
                min_value=0,
                max_value=500,
                value=0,
                step=1
            )

            wickets_lost = st.number_input(
                "Wickets Already Lost",
                min_value=0,
                max_value=9,
                value=0,
                step=1
            )

        st.divider()

        st.subheader("2. Player Information")

        col1, col2 = st.columns(2)

        with col1:

            batsman = st.selectbox(
                "Current Batsman",
                batsmen
            )

            batting_style = st.selectbox(
                "Batting Style",
                batting_styles
            )

        with col2:

            bowler = st.selectbox(
                "Current Bowler",
                bowlers
            )

            bowling_style = st.selectbox(
                "Bowling Style",
                bowling_styles
            )

            bowling_arm = st.selectbox(
                "Bowling Arm",
                bowling_arms
            )

        st.divider()

        st.subheader("3. Previous Delivery")

        col1, col2 = st.columns(2)

        with col1:

            previous_ball_runs = st.number_input(
                "Previous Ball Runs",
                min_value=0,
                max_value=7,
                value=0
            )

        with col2:

            previous_ball_wicket = st.checkbox(
                "Was there a wicket on the previous ball?"
            )

        st.divider()

        st.subheader("4. Advanced Settings")

        manual_balls = st.checkbox(
            "Manually enter legal balls already bowled"
        )

        estimated_balls = (
            int(current_over) * 6
            + int(ball_number) - 1
        )

        if manual_balls:

            balls_before = st.number_input(
                "Legal Balls Already Bowled",
                min_value=0,
                max_value=120,
                value=estimated_balls,
                step=1
            )

        else:

            balls_before = estimated_balls

            st.info(
                f"Estimated legal balls already bowled: "
                f"{balls_before}"
            )

        st.caption(
            "The automatic estimate assumes six legal balls per over. "
            "Wides and no-balls can change the actual count."
        )

        st.divider()

        st.subheader("5. Verify Information")

        confirm = st.checkbox(
            "I have checked all the match and player information."
        )

        submit = st.form_submit_button(
            "🏏 PREDICT WICKET",
            use_container_width=True
        )


    if submit:

        if batting_team == bowling_team:

            st.error(
                "Batting team and bowling team cannot be the same."
            )

        elif not confirm:

            st.error(
                "Please confirm that you have checked the information."
            )

        else:

            # Prepare input in exact training feature order
            input_data = {

                "inning": int(inning),

                "over_number": int(current_over),

                "balls_before": int(balls_before),

                "runs_before": int(current_score),

                "wickets_before": int(wickets_lost),

                "previous_ball_runs": int(previous_ball_runs),

                "previous_ball_wicket": int(
                    previous_ball_wicket
                ),

                "batting_team": batting_team,

                "bowling_team": bowling_team,

                "batsman": batsman,

                "bowler": bowler,

                "batting_style": batting_style,

                "bowling_style": bowling_style,

                "bowling_arm": bowling_arm

            }

            try:

                prediction, probability = make_prediction(
                    input_data
                )

                # Save latest prediction
                st.session_state.latest_prediction = {

                    "prediction": prediction,

                    "probability": probability,

                    "input": input_data

                }

                # Save history
                st.session_state.prediction_history.append({

                    "Batting Team": batting_team,

                    "Bowling Team": bowling_team,

                    "Batsman": batsman,

                    "Bowler": bowler,

                    "Innings": inning,

                    "Over": current_over,

                    "Ball": ball_number,

                    "Score": current_score,

                    "Wickets Lost": wickets_lost,

                    "Wicket Probability (%)": round(
                        probability * 100, 2
                    ),

                    "Prediction": (
                        "Wicket"
                        if prediction == 1
                        else "No Wicket"
                    )

                })

                st.rerun()

            except Exception as error:

                st.error(
                    f"Prediction error: {error}"
                )


    result = st.session_state.latest_prediction

    if result is not None:

        st.divider()

        st.subheader("📈 Prediction Result")

        probability = result["probability"]

        prediction = result["prediction"]

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Estimated Wicket Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )

        with col2:

            if prediction == 1:

                st.error(
                    "⚠️ Model Prediction: WICKET"
                )

            else:

                st.success(
                    "🟢 Model Prediction: NO WICKET"
                )

        st.caption(
            "The model's estimated probability is experimental. "
            "It is not a calibrated or guaranteed real-world forecast."
        )

        with st.expander("Review Prediction Inputs"):

            st.json(
                result["input"]
            )


elif page == "Analytics":

    st.title("📊 NPL Cricket Analytics")

    st.write(
        "Explore historical NPL ball-by-ball statistics."
    )

    st.divider()


    st.subheader("Wickets by Batting Team")

    team_wickets = (
        df.groupby("batting_team")["is_wicket"]
        .sum()
        .reset_index()
        .sort_values(
            "is_wicket",
            ascending=False
        )
    )

    fig = px.bar(
        team_wickets,
        x="batting_team",
        y="is_wicket",
        text="is_wicket",
        title="Wickets by Batting Team",
        labels={
            "batting_team": "Batting Team",
            "is_wicket": "Total Wickets"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()


    st.subheader("Runs by Batting Team")

    team_runs = (
        df.groupby("batting_team")["total_runs"]
        .sum()
        .reset_index()
        .sort_values(
            "total_runs",
            ascending=False
        )
    )

    fig = px.bar(
        team_runs,
        x="batting_team",
        y="total_runs",
        text="total_runs",
        title="Total Runs by Batting Team"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()
    

    # MODEL COMPARISON
    st.subheader("🤖 Machine Learning Model Comparison")

    if COMPARISON_PATH.exists():

        comparison_df = pd.read_csv(
            COMPARISON_PATH
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # Find PR-AUC column
        pr_columns = [
            column
            for column in comparison_df.columns
            if "pr_auc" in column.lower()
        ]

        if pr_columns:

            model_column = comparison_df.columns[0]

            fig = px.bar(
                comparison_df,
                x=model_column,
                y=pr_columns[0],
                title="Model Comparison - PR-AUC",
                text=pr_columns[0]
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "Model comparison CSV not found."
        )

        st.caption(
            "Run your model comparison script to generate it."
        )


elif page == "Prediction History":

    st.title("📜 Prediction History")

    st.write(
        "Review all predictions made during this session."
    )

    st.divider()

    history = st.session_state.prediction_history

    if len(history) > 0:

        history_df = pd.DataFrame(history)

        st.metric(
            "Total Predictions",
            len(history_df)
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # CSV download
        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction History",
            data=csv_data,
            file_name="npl_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):

            st.session_state.prediction_history = []

            st.session_state.latest_prediction = None

            st.rerun()

    else:

        st.info(
            "No predictions available yet. "
            "Visit Wicket Predictor to make your first prediction."
        )




st.divider()

st.caption(
    "🏏 NPL CRICKET INTELLIGENCE | "
    "Experimental Machine Learning Prediction System"
)

st.caption(
    "Nepal Premier League | Ball-by-Ball Analytics"
)