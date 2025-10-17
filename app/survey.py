"""
User Survey Module

Handles user feedback collection and storage for the AI Professor Platform.
Saves survey results to Google Sheets for easy access and analysis.
"""

import csv
import logging
import streamlit as st
from pathlib import Path
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

logger = logging.getLogger(__name__)


def render_user_survey():
    """Render user feedback survey in sidebar"""
    st.markdown("---")
    st.markdown("### 📊 Quick Feedback")

    # Initialize survey responses in session state if not present
    if "survey_speed_up" not in st.session_state:
        st.session_state.survey_speed_up = 50
    if "survey_recommend" not in st.session_state:
        st.session_state.survey_recommend = "Yes"
    if "survey_submitted" not in st.session_state:
        st.session_state.survey_submitted = False

    # Slider: How much does this speed you up
    speed_up = st.slider(
        "How much does this speed up your studying?",
        min_value=0,
        max_value=100,
        value=st.session_state.survey_speed_up,
        format="%d%%",
        help="Estimate the percentage improvement in your study efficiency"
    )
    st.session_state.survey_speed_up = speed_up

    # Selectbox: Would you recommend
    recommend = st.selectbox(
        "Would you recommend this tool to other students?",
        options=["Yes", "No"],
        index=0 if st.session_state.survey_recommend == "Yes" else 1,
        help="Let us know if you'd recommend this to your classmates"
    )
    st.session_state.survey_recommend = recommend

    # Submit button
    if st.button("Submit Feedback", type="primary", use_container_width=True):
        save_survey_results(speed_up, recommend)
        st.session_state.survey_submitted = True
        st.success("Thank you for your feedback!")

    # Show submission status
    if st.session_state.survey_submitted:
        st.caption(" Feedback submitted")


def _get_google_sheets_client():
    """Initialize Google Sheets client using service account credentials"""
    try:
        # Get credentials from Streamlit secrets
        credentials_dict = st.secrets.get("gcp_service_account", None)

        if not credentials_dict:
            logger.error("No Google service account credentials found in secrets")
            return None

        # Define the scopes
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        # Create credentials from the service account info
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=scopes
        )

        # Authorize and return the client
        client = gspread.authorize(credentials)
        return client

    except Exception as e:
        logger.error(f"Failed to initialize Google Sheets client: {e}")
        return None


def save_survey_results(speed_up: int, recommend: str):
    """Save survey results to Google Sheets

    Args:
        speed_up: Percentage improvement in study efficiency (0-100)
        recommend: Whether user would recommend the tool ("Yes" or "No")
    """
    # Prepare feedback data
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "course": st.session_state.get("selected_course", "unknown"),
        "lesson": st.session_state.get("selected_lesson", "unknown"),
        "speed_up_percentage": speed_up,
        "would_recommend": recommend,
        "messages_count": len(st.session_state.get("messages", []))
    }

    # Try Google Sheets first
    try:
        client = _get_google_sheets_client()

        if client:
            # Get the spreadsheet (create if doesn't exist)
            spreadsheet_name = "AI Professor Survey Results"

            try:
                spreadsheet = client.open(spreadsheet_name)
            except gspread.SpreadsheetNotFound:
                # Create new spreadsheet if it doesn't exist
                spreadsheet = client.create(spreadsheet_name)
                # Share with your email for access
                # spreadsheet.share('your-email@example.com', perm_type='user', role='writer')
                logger.info(f"Created new spreadsheet: {spreadsheet_name}")

            # Get the first worksheet (or create it)
            try:
                worksheet = spreadsheet.sheet1
            except Exception:
                worksheet = spreadsheet.add_worksheet(title="Survey Responses", rows=1000, cols=10)

            # Check if headers exist, if not add them
            if worksheet.row_count == 0 or not worksheet.row_values(1):
                headers = list(feedback_data.keys())
                worksheet.append_row(headers)
                logger.info("Added headers to Google Sheet")

            # Append the data
            row_data = list(feedback_data.values())
            worksheet.append_row(row_data)
            logger.info(f"Survey feedback saved to Google Sheets: {feedback_data}")
            return True

    except Exception as e:
        logger.error(f"Failed to save to Google Sheets: {e}")
        logger.info("Falling back to local CSV storage")

    # Fallback to CSV if Google Sheets fails
    try:
        results_dir = Path(__file__).parent / "survey_results"
        results_dir.mkdir(exist_ok=True)

        csv_file = results_dir / "user_feedback.csv"
        file_exists = csv_file.exists()

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=feedback_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(feedback_data)
        logger.info(f"Survey feedback saved to CSV (fallback): {feedback_data}")
        return True

    except Exception as e:
        logger.error(f"Failed to save survey results to both Google Sheets and CSV: {e}")
        return False
