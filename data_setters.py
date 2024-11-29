import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
from dbmanager import SqliteDBManager


class GoogleAuth:
    def __init__(self):
        self.key: str = "sports-data-project-google-service-acct.json"
        self.auth: gspread.auth = self.get_auth()

    def get_auth(self) -> gspread.auth:
        """Get google auth object for use in google api's and services.

        Returns:
            auth (gspread.auth): Authorization via google service account
        """
        return gspread.service_account(filename=self.key)


class GoogleSheetsSetter(GoogleAuth):
    def open_gsheet_by_url(self, url: str) -> gspread.worksheet:
        """Open a google sheet with a provided url."""

        if "https://docs.google.com/spreadsheets/" not in url:
            raise ValueError(f"""Provided destination url {url} does not appear
                             to be a valid google sheets URL""")

        return self.auth.open_by_url(url)

    def add_sheet_if_not_exists(
        self, spreadsheet, sheet_name: str, rows: int = 1000, cols: int = 26
    ):
        """Add a sheet to the Google Spreadsheet if it does not already exist.

        Args:
            spreadsheet: The Google Spreadsheet object.
            sheet_name (str): The name of the sheet to add if not found.
            rows (int, optional): Number of rows for the new sheet. Defaults to 1000.
            cols (int, optional): Number of columns for the new sheet. Defaults to 26.

        Returns:
            The worksheet object for the newly created or existing sheet.
        """
        try:
            # Try to get the worksheet by name
            return spreadsheet.worksheet(sheet_name)
        except Exception:
            # If not found, add a new sheet
            try:
                return spreadsheet.add_worksheet(title=sheet_name, rows=rows, cols=cols)
            except Exception as e:
                raise Exception(f"Failed to create a new sheet '{sheet_name}': {e}")

    def df_to_gsheet_via_url(
        self, df: pd.DataFrame, dest_url: str, sheet_name: str
    ) -> None:
        """Send a dataframe to a specified google sheet using the workbook URL.

        Args:
            df (pd.DataFrame): The provided dataframe intended to go into the sheet.
            dest_url (str): Provided destination url
            sheet_name (str): The name of the sheet you wish to port the df into.

        Raises:
            ValueError: If provided df is empty will throw a value error.
        """
        if df.empty:
            raise ValueError("Dataframe cannot be empty.")

        try:
            # Open the Google Sheet by URL
            spreadsheet = self.open_gsheet_by_url(dest_url)
        except Exception as e:
            raise Exception(f"Failed to open Google Sheet at {dest_url}: {e}")

        try:
            # Access the specified worksheet, create one if it doesn't exist.
            worksheet = self.add_sheet_if_not_exists(spreadsheet, sheet_name)
        except Exception as e:
            raise KeyError(f"Sheet '{sheet_name}' not found in the Google Sheet: {e}")

        try:
            # Write the dataframe to the worksheet
            set_with_dataframe(worksheet, df)
        except Exception as e:
            raise Exception(
                f"Failed to write dataframe to the sheet '{sheet_name}': {e}"
            )
