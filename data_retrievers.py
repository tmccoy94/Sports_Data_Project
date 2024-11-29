import pandas as pd
from dbmanager import SqliteDBManager


class Data_Retriever:
    def __init__(self, league_name: str, db_name: str):
        """A retriever designed to do specific retrieval tasks within
        the specified database.

        Args:
            league_name (int): The name of league you want info for
            db_name (str): The name of the database you are pulling from.
        """
        self.db_manager = SqliteDBManager(db_name)
        self.league_name = league_name
        self.league_serial = self._get_league_serial(self.league_name)

    def _get_league_serial(self, league: str) -> int:
        """
        Retrieve odds api sport name for the api call from the database.

        Args:
          League (str): The name of the league you are querying from the DB.
        """
        serial: list[tuple] = self.db_manager.fetch_records(
            f"SELECT SERIAL FROM LEAGUES WHERE NAME ='{league}'", fetchstyle="one"
        )
        return serial[0][0]  # unpack tuple

    def get_predicted_info_by_team(self) -> pd.DataFrame:
        f"""Returns a dataframe with predicted outcomes for future games for
        {self.league_name}"""

        return self.db_manager.dataframe_query(
            f"""
        SELECT
            HOME_TEAM.TEAM_NAME AS HOME_TEAM,
            AWAY_TEAM.TEAM_NAME AS AWAY_TEAM,
            GOO.PREDICTED_HOME_POINTS AS PREDICTED_HOME_SCORE,
            GOO.PREDICTED_AWAY_POINTS AS PREDICTED_AWAY_SCORE,
            GOO.SPREAD AS HOME_SPREAD,
            GOO.TOTAL AS PREDICTED_TOTAL,
            GAMES.GAME_DATE
        FROM
            GAME_OWN_ODDS AS GOO JOIN GAMES AS GAMES ON GOO.GAME_SERIAL = GAMES.SERIAL
            JOIN TEAMS AS HOME_TEAM ON HOME_TEAM.SERIAL = GAMES.HOME_TEAM_SERIAL
            JOIN TEAMS AS AWAY_TEAM ON AWAY_TEAM.SERIAL = GAMES.AWAY_TEAM_SERIAL
        WHERE
            GAMES.LEAGUE_SERIAL = {self.league_serial}
        """
        )
