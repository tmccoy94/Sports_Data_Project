import pandas as pd
from dbmanager import SqliteDBManager
import data_packers


class NFL_Data_Retriever(data_packers.NFL_Data_Packer):
    def get_predicted_info_by_team(self):
        f"""Returns a dataframe with predicted outcomes for future games for
        {self.league}"""

        return self.db_manager.dataframe_query(
            """
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
        """
        )
