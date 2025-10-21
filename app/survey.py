"""
User Survey Module

Handles user feedback collection and storage for the AI Professor Platform.
Saves survey results to Google Sheets for easy access and analysis.
"""

import logging
import streamlit as st
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
    """Initialize Google Sheets client using service account credentials from Streamlit secrets"""
    try:
        # Get credentials from Streamlit secrets
        credentials_dict = st.secrets.get("gcp_service_account")

        if not credentials_dict:
            logger.warning("No Google service account credentials found in secrets")
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
        logger.info("Successfully initialized Google Sheets client")
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

    try:
        client = _get_google_sheets_client()

        if not client:
            logger.error("Google Sheets client not initialized")
            return False

        # Open the existing shared spreadsheet
        spreadsheet_name = "AI Professor Survey Results"
        spreadsheet = client.open(spreadsheet_name)
        worksheet = spreadsheet.sheet1

        # Check if headers exist, if not add them
        existing_headers = worksheet.row_values(1) if worksheet.row_count > 0 else []
        if not existing_headers:
            headers = list(feedback_data.keys())
            worksheet.append_row(headers)
            logger.info("Added headers to Google Sheet")

        # Append the data
        row_data = list(feedback_data.values())
        worksheet.append_row(row_data)
        logger.info("Survey feedback saved to Google Sheets successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to save survey results to Google Sheets: {e}")
        return False
