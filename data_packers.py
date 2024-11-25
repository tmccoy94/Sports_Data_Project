import requests
import pandas as pd
from restricted import odds_api_key
from dbmanager import SqliteDBManager
from dataclasses import dataclass
from datetime import datetime, timedelta
import pytz
from typing import Any
import time, copy

LOCAL_TIMEZONE = 'America/New_York'

@dataclass
class Team_Games_Scraper:
    """This dataclass is to ensure type safety for info used to scrape teams"""
    team_name: str
    tr_url_name: str


class Sports_DB_Packer:
    """
    This is a class buit to make getting data for a sports league from teamrankings.com, and the odds api
    and packing it into an SQLite database. In this case using the other class, SqliteDBManager.

    This is made specifically for the SportsData db created for the sports data app, but could be easily changed. 
    """
    def __init__(self):
        self.api_key: str = odds_api_key
        self.regions: str = 'us'
        self.markets: str = 'spreads,totals,h2h'
        self.odds_format: str = 'decimal'
        self.odds_date_format: str = 'iso'
        self.db_date_format: str = r"%Y-%m-%d %H:%M:%S" # for use in formatting dates on way to db
        self.sport: str = None # defined in sublcasses
        self.league_serial: int = None # defined in sublcasses
        self.requests_used: int = None # Requests for odds API
        self.requests_remaining: int = None # Requests for odds API
        self.odds_called: bool = False # change if odds api is called
        self.odds_df: pd.DataFrame = None # info pulled from odds api
        self.combined_market_odds_df: pd.DataFrame = None # Combine odds_df and games_table info
        self.future_games_checked: bool = False # check if future games have been accounted for yet
        self.manager: SqliteDBManager = SqliteDBManager('SportsData.db')
        self.tables: list[str] = self.manager.check_table_names_in_db()
        self.db_last_updated_date: str = self._query_last_db_updated_date()        
        self.teams_df : pd.DataFrame = self._query_teams_table()        
        self.team_serial_ref_dict: dict[str, str] = self._build_team_serial_ref()

    def _query_last_db_updated_date(self) -> str:
        """Get date the sports db was last updated"""
        date = self.manager.fetch_records(f"""SELECT MAX(LAST_UPDATED_DATE) FROM DB_UPDATE_RECORDS
                                          WHERE LEAGUE_SERIAL = {self.league_serial}""",fetchstyle='one')
        return date[0][0]

        
    def _get_sport(self, league: str) -> str:
        """
        Retrieve odds api sport name for the api call from the database.

        Args:
          League (str): The name of the league you are querying from the DB.
        """
        sport: str = self.manager.fetch_records(f"SELECT ODDS_URL_NAME FROM LEAGUES WHERE NAME ='{league}'",
                                   fetchstyle='one')
        return sport[0][0] # unpack tuple list to get val
    
    def _get_league_serial(self, league: str) -> int:
        """
        Retrieve odds api sport name for the api call from the database.

        Args:
          League (str): The name of the league you are querying from the DB.
        """
        serial: int = self.manager.fetch_records(f"SELECT SERIAL FROM LEAGUES WHERE NAME ='{league}'",
                                   fetchstyle='one')
        return serial[0][0] # unpack tuple list to get val
    
    # Query and create everything needed for all functionality here
    def _query_teams_table(self) -> pd.DataFrame:
        """Queries the TEAMS table in the db as a df for use in gathering teams data."""
        return self.manager.dataframe_query("select * from TEAMS")
    
    def _build_team_serial_ref(self) -> dict[str, int]:
        """For referencing the team name to the team serial for all teams in db for this league"""
        return {team_name:serial for team_name,serial in zip(self.teams_df['TEAM_NAME'],self.teams_df['SERIAL'])}
    
    def insert_new_db_updated_date(self) -> None:
        """Insert a new updated date into the db after updates are done"""
        now =  datetime.now().strftime(self.db_date_format)

        self.manager.insert_table_records("DB_UPDATE_RECORDS", [(now,self.league_serial)])
    
    def call_odds_api(self) -> dict:
        """
        Call the odds api and store the response as a json.
        """
        odds_response = requests.get(
            f'https://api.the-odds-api.com/v4/sports/{self.sport}/odds',
            params={
                'api_key': self.api_key,
                'regions': self.regions,
                'markets': self.markets,
                'oddsFormat': self.odds_format,
                'dateFormat': self.odds_date_format
            }
        )

        if odds_response.status_code != 200:
            print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

        else:
            # Jsonify the results
            odds_json = odds_response.json()

            # Check the usage quota
            self.requests_remaining = odds_response.headers['x-requests-remaining']
            self.requests_used = odds_response.headers['x-requests-used']

            return odds_json
        
    def get_market_info(self) -> list[tuple[int,str]]:
        """Queries MARKETS table and returns a tuple for each market: (market serial, market name)"""

        markets_table = self.manager.dataframe_query('SELECT * FROM MARKETS')

        markets = []

        for item in markets_table.itertuples():
            markets.append(item[1:]) #drop index from db
        
        return markets
        
    def parse_market_info(self, market_name: str, market_serial: int, response_json: list[dict]) -> pd.DataFrame:
        entries = []

        for response in response_json:
            entry = {
                'Home_Team': response['home_team'],
                'Away_Team': response['away_team'],
                'Date': response['commence_time'],
                'MARKET_SERIAL': market_serial,
                'H2H_HOME': None, 
                'H2H_AWAY': None, 
                'SPREAD_HOME': None, 
                'TOTAL': None
            }

            for bookmaker in response['bookmakers']:
                if bookmaker['title'] == market_name:
                    for market in bookmaker['markets']:
                        if market['key'] == 'h2h':
                            entry['H2H_HOME'] = market['outcomes'][0]['price']
                            entry['H2H_AWAY'] = market['outcomes'][1]['price']
                        elif market['key'] == 'spreads':
                            entry['SPREAD_HOME'] = market['outcomes'][0]['point']
                        elif market['key'] == 'totals':
                            entry['TOTAL'] = market['outcomes'][0]['point']

            entries.append(entry)

            df = pd.DataFrame(entries)
        
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_convert(LOCAL_TIMEZONE)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return df
    
    def get_odds_df(self) -> None:
        markets: list[tuple[int,str]] = self.get_market_info()
        response_json: list[dict] = self.call_odds_api()

        market_dfs = []

        for serial, market_name in markets:
            market_dfs.append(self.parse_market_info(market_name, serial, response_json))

        
        self.odds_df = pd.concat(market_dfs, axis=0)
        self.odds_df['home_serial'] = self.odds_df['Home_Team'].apply(self.get_team_serial)
        self.odds_df['away_serial'] = self.odds_df['Away_Team'].apply(self.get_team_serial)
        self.odds_df = self.odds_df[['Home_Team','home_serial','Away_Team', 'away_serial','MARKET_SERIAL', 'SPREAD_HOME', 
                      'TOTAL', 'H2H_HOME', 'H2H_AWAY','Date']]
        
        self.odds_called = True

    def show_api_requests_used(self) -> None:
        print(f"You have used {self.requests_used} calls and have {self.requests_remaining} remaining.")    

    def get_team_serial(self, team_name: str):
        return self.team_serial_ref_dict[team_name]

    def populate_future_game_data(self, debug: bool = False) -> None:
        """
        Check most recently entered game and grab any game data from odds data which surpasses that date.
        """
        if not self.league_serial:
            raise AttributeError("self.league_serial must be populated to use this function.")
        if self.odds_df.empty:
            raise AttributeError("You haven't yet called the odds api info to compare this with.")
        
        most_recent_game: str = self.manager.fetch_records("SELECT MAX(GAME_DATE) FROM GAMES",fetchstyle='one')[0][0]

        odds_df_copy: pd.DataFrame = copy.deepcopy(self.odds_df)

        #use only one market id for the game set
        market_id = odds_df_copy['MARKET_SERIAL'].unique()[0]

        # Check the games from the odds API and compare to what is currently in the db by date
        # Eliminate duplicate games by only grabbing games from one market
        try:
            unmade_games = odds_df_copy[(odds_df_copy['Date'] > most_recent_game) &
                                    odds_df_copy['MARKET_SERIAL'] == market_id]
        except KeyError as e:
            raise KeyError(f"""Looks like when you called the odds api the columns in the df did not populate
                           correctly. See specific error: {e}. See self.get_odds_df for more.""")
        # Add the league serial so it can be added to the games table
        unmade_games['league_serial'] = self.league_serial

        # Drop all columns except for what is needed for the games table
        unmade_games = unmade_games[['home_serial', 'away_serial', 'Date', 'league_serial']]

        self.df_to_db('GAMES', unmade_games, debug=debug)

        if not debug:
            self.future_games_checked = True

    def _combine_odds_api_and_games_dfs(self) -> pd.DataFrame:
        """Check that the correct columns are grabbed and combine the games table and odds df
        info so that they are ready to go into the market odds table."""

        if not getattr(self, 'odds_called', True):
            try:
                self.get_odds_df()
            except Exception as e:
                raise AttributeError(f"Odds api df not yet created. Use self.get_odds_df to fill. Error: {e}")
            
        else:
            changed_odds_check: pd.DataFrame = copy.deepcopy(self.odds_df)

            changed_odds_check['UID'] = changed_odds_check['UID'] = (
                changed_odds_check['home_serial'].astype(str) + ' ' +
                changed_odds_check['away_serial'].astype(str) + ' ' +
                changed_odds_check['Date'].astype(str)
            )

            games_df = self.manager.dataframe_query('select * from GAMES')
            games_df['UID'] = games_df['UID'] = (
                games_df['HOME_TEAM_SERIAL'].astype(str) + ' ' +
                games_df['AWAY_TEAM_SERIAL'].astype(str) + ' ' +
                games_df['GAME_DATE'].astype(str)
            )

            # Join the team df to the current db games data using the UID 
            joined_df = pd.merge(changed_odds_check,games_df, on='UID', how='inner')
            joined_df = joined_df.drop_duplicates(subset=['UID','MARKET_SERIAL'])
            joined_df = joined_df[['SERIAL','MARKET_SERIAL','SPREAD_HOME','TOTAL','H2H_HOME','H2H_AWAY', 'Date']]

            self.combined_market_odds_df = joined_df
    
    def get_existing_market_odds_records_for_called_games(self) -> pd.DataFrame:
        """Pull all existing market odds for games that were pulled from the api call."""
        if not hasattr(self, 'combined_market_odds_df'):
            self._combine_odds_api_and_games_dfs()
        elif self.combined_market_odds_df.empty:
            raise ValueError("Your combined odds df has no values")
        else:
            most_recent_record: str = self.manager.fetch_records("SELECT MAX(UPDATED_DATE) FROM GAME_MARKET_ODDS",fetchstyle='one')[0][0]
             # Retrieve game serials to query for
            game_serials: pd.Series[int] = self.combined_market_odds_df['SERIAL'].unique()   
            existing_game_data = self.manager.dataframe_query(f"""SELECT * FROM GAME_MARKET_ODDS 
                                                              WHERE GAME_SERIAL IN 
                                                              {tuple(str(serial) for serial in game_serials)}""")

            return existing_game_data
        
    def market_odds_have_changed(self, api_row: pd.Series, existing_row: tuple) -> bool:
        """Compare values in the pulled api results vs the existing rows we have."""
        return (
        api_row['SPREAD_HOME'] != existing_row.SPREAD or
        api_row['TOTAL'] != existing_row.TOTAL or
        api_row['H2H_HOME'] != existing_row.H2H_HOME or
        api_row['H2H_AWAY'] != existing_row.H2H_AWAY
        )
    
    def get_changed_market_odds(self, existing_data: pd.DataFrame, api_data: pd.DataFrame) -> pd.DataFrame:
        new_rows = []

        for row in existing_data.itertuples():

            # Find matching rows in api_data on game & market serial
            api_data_match = api_data.loc[
                (api_data['SERIAL'] == int(row.GAME_SERIAL)) &
                (api_data['MARKET_SERIAL'] == int(row.MARKET_SERIAL))
            ]

            if api_data_match.empty:
                continue # skip if no rows match

            # Handle single or multiple matches
            for _, api_row in api_data_match.iterrows():
                if self.market_odds_have_changed(api_row, row):
                    # Append tuple or other logic for changed rows
                    new_rows.append({
                        'GAME_SERIAL': api_data['SERIAL'],
                        'MARKET_SERIAL': api_data['MARKET_SERIAL'],
                        'SPREAD_HOME': api_row['SPREAD_HOME'],
                        'TOTAL': api_row['TOTAL'],
                        'H2H_HOME': api_row['H2H_HOME'],
                        'H2H_AWAY': api_row['H2H_AWAY']
                    })

        new_entries = pd.DataFrame(new_rows)

        return new_entries

    def refresh_market_odds_table(self, debug: bool = False) -> None:
        """Get the existing data and compare it to the newly pulled api market odds data"""
        if getattr(self, 'odds_called', False):
            try:
                self.get_odds_df()
            except Exception as e:
                raise Exception(f"Could not get odds_df. Troubleshoot starting there. Error was: {e}")
        if not hasattr(self, 'combined_market_odds_df'):
            try:
                self._combine_odds_api_and_games_dfs()
            except Exception as e:
                raise Exception(f"Could not get combined_market_odds_df. Troubleshoot starting there. Error was: {e}")
        else:
            existing_mkt_odds: pd.DataFrame = self.get_existing_market_odds_records_for_called_games()
            api_mkt_odds: pd.DataFrame = copy.deepcopy(self.combined_market_odds_df)
            api_mkt_odds = api_mkt_odds.drop('Date', axis=1) # drop date column for comparison

            new_entries = self.get_changed_market_odds(existing_mkt_odds,api_mkt_odds)


            if new_entries.empty:
                return ("No game odds that exist have changed.")
            else:
                self.df_to_db("GAME_MARKET_ODDS",new_entries,debug=debug)

    def pack_odds_table(self, debug: bool = False) -> None:
        """Compare the most recently added game odds to what is in the api data, then
        add the data to the game_market_odds table."""
        
        # Get the most recent game date from DB
        most_recent_game_date_has_odds = self.manager.fetch_records("""
                        SELECT 
                            G.GAME_DATE AS GAME_DATE
                        FROM 
                            GAMES AS G JOIN GAME_MARKET_ODDS AS GMO ON G.SERIAL = GMO.GAME_SERIAL
                        WHERE 
                            G.GAME_DATE = (SELECT MAX(G.GAME_DATE) FROM 
                                            GAMES AS G JOIN GAME_MARKET_ODDS AS GMO 
                                            ON G.SERIAL = GMO.GAME_SERIAL)
                        """, fetchstyle='one')[0][0]
        
        # Filter market odds by the game date
        odds_to_add: pd.DataFrame = self.combined_market_odds_df[(self.combined_market_odds_df['Date'] > 
                                                                  most_recent_game_date_has_odds)]

        if odds_to_add.empty:
            return("No odds to add right now.")
        
        # Drop game date so df can go in market odds table
        odds_to_add = odds_to_add.drop('Date', axis=1)

        if debug:
            print(odds_to_add)
        
        self.df_to_db('GAME_MARKET_ODDS', odds_to_add, debug=debug)
    
    def full_odds_run(self):
        # Run the full odds api process and insert the records.
        self.get_odds_df()
        self.populate_future_game_data()
        self._combine_odds_api_and_games_dfs()
        self.refresh_market_odds_table()
        self.pack_odds_table()


    # move to dbmanager later *****
    def df_to_db(self, table_name: str, df: pd.DataFrame, debug: bool = False):
        f"""
        This function is intended only to take in a df with matching cols to
        the table to which you are inserting.

        Args:
            table_name: THe name of the table that you are trying to insert records into.
            (table options: {self.tables})
            df: The dataframe you are trying to send information from (make sure cols match
            table)
            debug: Boolean, turn on to see your query printed out. it will not yet send it
            in until you set to False.
        """
        if table_name not in self.tables:
            raise ValueError(f"""The table name you are using is not in the Sports Data DB.
            Here are the list of tables: {self.tables}""")
        row_tuples = [tuple(row) for row in df.itertuples(index=False, name=None)]
    
        self.manager.insert_table_records(table_name, records=row_tuples, debug=debug)


# --------------------------------------- NFL DATA PACKER ---------------------------------------------

class NFL_Data_Packer(Sports_DB_Packer):
    """
    This object is intended to get NFL data and send it into the Sports Data DB.

    If a func name has tr, it means it has to do with Team Rankings data. If it has
    odds, it has to do with the odds api.
    """

    def __init__(self, debug = False):
        """
        Here we query the Sports DB for the league that is chosen in the data packer.

        We also get all the team data for that league and create keys for later data
        translation from the different data sources so they match db info.

        We also defin
        """
        super().__init__()   
        self.debug = debug
        self.league: str = 'NFL'
        self.sport : str = self._get_sport(self.league)
        self.league_serial: int = self._get_league_serial(self.league)
        self.team_tr_table_key: dict[str, str] = self._build_team_tr_table_key() 
        self.url_ref_dict: dict[str, str] = self._build_tr_url_ref_dict()
        self.opponent_map = self._build_opponent_map()
        self.odds_df: pd.DataFrame = None
        self.all_games_data: dict[str,pd.DataFrame] = None
        self.game_data_packed: bool = False

    

    def _build_team_tr_table_key(self) -> dict[str, str]:
        """
        This grabs the names that teamranks uses in their tables that we scrape with this object
        then matches them with the team serials that exist in the Sports Data database so that
        they can be sent into the STATS tables that the TEAMS table is connected to.
        """
        return {x:y for x,y in zip(self.teams_df['TR_TABLE_NAME'], self.teams_df['SERIAL'])}
    
    def _build_tr_url_ref_dict(self) -> dict[str, str]:
        """
        This uses the teamranks URL column in the teams table so that you can scrape the data for
        each team from the database. For use in the team scrape functionality.
        """
        return {team_name: url_name for team_name, url_name in zip(self.teams_df['TEAM_NAME'], 
                                                                   self.teams_df['TR_URL_NAME'])}
    
    def _build_opponent_map(self) -> dict[str, str]:
        """Returns a dict used to map the names of opponents on team schedule to names that will
        match what the names are in the Sports DB."""
        return {team_tr_name: url_name for team_tr_name, url_name in zip(self.teams_df['TR_TABLE_NAME'], 
                                                                         self.teams_df['TEAM_NAME'])}
    
    # This section is for OPPG and PPG stats calling and packing
    
    def _call_tr_ppg_and_oppg(self) -> list[pd.DataFrame]:
        """
        Call the info from the teamrankings site for points per game and opponents points per game.
        """
            
        league: str = self.manager.fetch_records(f"SELECT TR_URL_NAME FROM LEAGUES WHERE NAME = '{self.league.upper()}'", 'one')
        league = league[0][0] #unpack tuple
        ppg = pd.read_html(f"https://www.teamrankings.com/{league.lower()}/stat/points-per-game")
        oppg = pd.read_html(f"https://www.teamrankings.com/{league.lower()}/stat/opponent-points-per-game")

        ppg = ppg[0]
        oppg = oppg[0]
        return [ppg, oppg]
    
    def _reorder_team_data_ppg_oppg(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        This function reorders and renames the default df cols from the dfs

        Args:
          df (pd.DataFrame): A dataframe from team rankings that inlcludese PPG data from the website.

        Returns: 
          The same df ordered so that it can be inserted into the GAME PPG tables in the SportsData.db
        """
        df.columns = ['RANK', 'TEAM', 'THIS_YEAR_PPG','LAST_3','LAST_1','HOME','AWAY', 'LAST_YEAR_PPG', 'TEAM_SERIAL']
        df = df.drop('TEAM', axis=1)
        df = df[['TEAM_SERIAL', 'RANK', 'THIS_YEAR_PPG','LAST_3','LAST_1','HOME','AWAY', 'LAST_YEAR_PPG']]
        return df
    
    def pack_ppg_oppg(self, debug: bool = False) -> None:
        """
        Send OPPG and PPG stats into the database.

        Args:
          debug: Bool, turn on to see your query printed before you execute it
          by setting this arg to False(default val).
        """
        # Get the data
        ppg, oppg = self._call_tr_ppg_and_oppg()
        # Get cols in line with DB formatting
        ppg['TEAM_SERIAL'] = [self.team_tr_table_key[key] for key in ppg['Team']]
        oppg['TEAM_SERIAL'] = [self.team_tr_table_key[key] for key in oppg['Team']]

        ppg = self._reorder_team_data_ppg_oppg(ppg)
        oppg = self._reorder_team_data_ppg_oppg(oppg)
        # Send data into database
        self.df_to_db('TEAM_TR_FOOTBALL_PPG', ppg, debug=debug)
        self.df_to_db('TEAM_TR_FOOTBALL_OPPG', oppg, debug=debug)

    # This section is for team games data calling and packing

    def build_scrapers(self) -> list[Team_Games_Scraper]:
        """Create scrapers for each team in the chosen league"""
        scrapers = []

        for name, url_name in self.url_ref_dict.items():
            scrapers.append(Team_Games_Scraper(name,url_name))

        return scrapers
    
    def get_proper_team_name(self, tr_table_name: str):
        return self.opponent_map[tr_table_name]
    
    def format_team_game_history(self, df: pd.DataFrame):

        # Use regex groups to capture each part of the 'Result' column
        df[['Result', 'Points', 'Opp Points']] = df['Result'].str.extract(r'(\w+)\s+(\d+)-(\d+)')

        # Convert 'Points' and 'Opp Points' columns to integers
        df['Points'] = df['Points'].astype(int)
        df['Opp Points'] = df['Opp Points'].astype(int)
        df['Actual Spread'] = df['Opp Points'] - df['Points']
        df['Total Points'] = df['Opp Points'] + df['Points']
        df['Opponent'] =df['Opponent'].apply(self.get_proper_team_name)

        # Get the current year
        now = datetime.now()
        current_year = now.year

        # Convert Date column to include the current year and current time
        df['Date'] = pd.to_datetime(df['Date'] + f'/{current_year} ').dt.strftime(self.db_date_format)

        return df
    
    def get_all_games_data(self) -> None:
        """
        Scrape the team rankings website for the queried teams from the database using the teams_df.

        Debug mode will only grab the first 4 teams.

        In the processing we grab only the home games data, format the data, check if each game scraped
        is after the last time the database was updated (need to do after formatting so that the date
        formatting will match and the check will work), skip if no games are needing updated, assign
        the appropriate team and league serials if we are adding the data.
        """
        games_data_by_team = {}

        games_scrapers: list[Team_Games_Scraper] = self.build_scrapers() # uses teams_df data

        if self.debug:
            games_scrapers = games_scrapers[:4]

        for scraper in games_scrapers:
            # retrieve raw data
            nfl_team_data = pd.read_html(f"https://www.teamrankings.com/nfl/team/{scraper.tr_url_name}/")
            games_data = nfl_team_data[1].dropna() # Only games which have been played

            # Process data
            games_data = games_data[games_data['Location'] == 'Home']
            games_data = self.format_team_game_history(games_data)
            games_data = games_data[games_data['Date'] >= self.db_last_updated_date]

            # Check if DataFrame is empty
            if games_data.empty:
                continue  # Skip this team if no data matches the criteria

            # Add metadata
            games_data['League Serial'] = self.league_serial
            games_data['home_serial'] = self.get_team_serial(scraper.team_name)
            games_data['away_serial'] = games_data['Opponent'].apply(self.get_team_serial)

            # Assign processed data to the dictionary
            games_data_by_team[scraper.team_name] = games_data

            # Respect crawl delay
            time.sleep(10)

        self.all_games_data = games_data_by_team

    def pack_games_data(self, acknowdedged: bool = False) -> None:
        """This function is setup to pack all the games data into the GAMES table in the db.
        This should only have to be run to catch up if the scripts haven't run in a while.
        
        The odds api provides future games that get packed in, so running this function should
        be rare. As a result you have to acknowledge its usage."""
        if not self.all_games_data:
            raise ValueError("All games data is not populated, run self.get_all_games_data().")
        if not acknowdedged:
            raise ValueError("""This function is setup to pack all the games data into the GAMES table in the db.
        This should only have to be run to catch up if the scripts haven't run in a while.
        
        The odds api provides future games that get packed in to the db, so running this function should
        be rare. As a result you have to acknowledge its usage.
                             
        Rerun with acknowledged as True to run the function.""")
        games_table_data = copy.deepcopy(self.all_games_data)

        most_recent_game_date = self.manager.fetch_records("""
                                SELECT
                                    GAME_DATE
                                FROM
                                    GAMES
                                ORDER BY 
                                    GAME_DATE DESC
                                LIMIT 1""",fetchstyle="one")[0][0] # unpack
        
        games_table_data = games_table_data[games_table_data['Date'] >= most_recent_game_date]

        for team, df in games_table_data.items():
            df = df[['home_serial', 'away_serial', 'Date', 'League Serial']]
            self.df_to_db('GAMES', df)

        self.game_data_packed = True

    def pack_game_outcomes_data(self, debug: bool= False) -> None:


        if not self.game_data_packed:
            raise ValueError("Game data not yet packed, run self.pack_games_data().")
        # Create a copy of the retrieved all games data dict
        game_outcomes_table_data = copy.deepcopy(self.all_games_data)

        # Call all current game data in the db
        games_df = self.manager.dataframe_query('select * from GAMES')
        games_df['UID'] = games_df['UID'] = (
            games_df['HOME_TEAM_SERIAL'].astype(str) + ' ' +
            games_df['AWAY_TEAM_SERIAL'].astype(str) + ' ' +
            games_df['GAME_DATE'].astype(str)
        )
        
        for team, df in game_outcomes_table_data.items():
            # Process each df and add UID
            df = df[['Date', 'home_serial','away_serial','Result','Points','Opp Points']]
            df['UID'] = (
                df['home_serial'].astype(str) + ' ' +
                df['away_serial'].astype(str) + ' ' +
                df['Date'].astype(str)
                )   
            # Join the team df to the current db games data using the UID 
            joined_df = pd.merge(df,games_df, on='UID', how='inner')
            # Identify the winner for each game by team serial
            joined_df['winner_serial'] = joined_df.apply(
                lambda row : row['home_serial'] if row['Result'] == 'W' else row['away_serial'],
                axis= 1
            )
            # Drop unecessary data and insert into game outcomes table
            joined_df = joined_df[['SERIAL', 'winner_serial', 'Points', 'Opp Points']]
            self.df_to_db('GAME_OUTCOMES', joined_df, debug=debug)

    def pack_all_games_data(self) -> None:
        """Combining all funcs needed to get all the prior games and game outcomes for the
        entire season. This shouldn't be called every time."""
        self.get_all_games_data()
        self.pack_games_data()
        self.pack_game_outcomes_data()

    def pack_game_outcomes_data_only(self) -> None:
        self.get_all_games_data()
        self.pack_game_outcomes_data()

    # Put it all together

    def is_new_week(self) -> bool:
        """Use datetime to check if the db_last_updated date is in the past week.
        
        Returns:
            bool: True if the most recent Sunday has passed since the last_updated_date, 
                indicating a new calendar week. False otherwise.
        """

        last_updated_date = self.db_last_updated_date
        date_format = self.db_date_format

        # Parse the last updated date
        try:
            last_updated = datetime.strptime(last_updated_date, date_format)
        except ValueError as e:
            raise ValueError(f"Invalid date format for last_updated_date: {last_updated_date}. Expected format: {date_format}") from e

        # Calculate today's date
        today = datetime.now()

        # Find the most recent Sunday (relative to today)
        # If today is Sunday, it will return today's date; otherwise, the last Sunday.
        days_since_sunday = today.weekday() + 1  # weekday() returns 0 for Monday, so add 1
        most_recent_sunday = today - timedelta(days=days_since_sunday)

        # Check if the most recent Sunday has passed since the last updated date
        # AND the last updated date is before the most recent Sunday
        return (last_updated < most_recent_sunday)

        pass

    def full_pack(self, pack_all_games_data: bool = False) -> None:
        f"""This func is designed to run a standard data gathering for the {self.league}.
        
        A standard run will fill future game data and grab odds for any games from the odds
        api which have not already been populated and check to see if any of the odds have
        changed.
        
        It will also updated OPPG and PPG stats for all the teams.
        
        Weekly, it will grab all the games for all the teams and update the outcomes table.
        
        Args:
            pack_all_games_data (bool): If true, this will get all the 
            game data for the season and put it."""
        
        # get PPG OPPG stats
        self.pack_ppg_oppg()

        # Get odds data
        self.full_odds_run()

        # Pack all the games data if option is selected - NOT standard run.
        if pack_all_games_data:
            self.pack_all_games_data()
            return None

        # Check if it's a new week yet and get game outcomes data if so.
        if self.is_new_week():
            self.pack_game_outcomes_data_only()

        self.insert_new_db_updated_date() # Insert new updated record into the db

        return "Packed all data." # Need to write a logger so we get more than this lol

