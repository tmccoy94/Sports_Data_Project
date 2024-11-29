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


class GoogleSheetsSetter:
    def open_gsheet_by_url(self, url: str) -> gspread.worksheet:
        return self.auth.open_by_url(url)

    def df_to_gsheet(self, df: pd.DataFrame, dest_url: str, sheet_name: str) -> None:
        spreadsheet = self.open_gsheet_by_url(dest_url)

        worksheet = spreadsheet.worksheet(sheet_name)

        set_with_dataframe(worksheet, df)
