import requests
import pandas as pd
from restricted import odds_api_key
from dbmanager import SqliteDBManager
from dataclasses import dataclass
from datetime import datetime, timedelta
import pytz
import time
import copy

LOCAL_TIMEZONE = 'America/New_York'

@dataclass
class Team_Games_Scraper:
    """This dataclass is to ensure type safety for info used to scrape teams"""
    team_name: str
    tr_url_name: str

    # maybe build out some funcs here that ensure that the team_name and tr_url_name exist in the db

@dataclass
class GameOwnOddsRow:
    game_serial: int
    predicted_home_points: float
    predicted_away_points: float

    @property
    def spread(self) -> float:
        return round(self.predicted_home_points - self.predicted_away_points,2)
    
    @property
    def total(self) -> float:
        return round(self.predicted_home_points + self.predicted_away_points,2)
    
    def to_dict(self):
        """
        Convert the values in the object to a dictionary, and enforce type security
        for the database
        """
        return {
            "game_serial": self.game_serial,
            "predicted_home_points": float(self.predicted_home_points),
            "predicted_away_points": float(self.predicted_away_points),
            "spread": float(self.spread),
            "total": float(self.total),
        }

class OddsApiCallerMixin:
    """This class is intended to be used for calling the odds api information and returning it.
    Args:
          sport (str): The sport you are looking for. Reference odds api docs.
          regions (str): The regions you are using. Reference odds api docs.
          markets (str): The markets you are using. Reference odds api docs.
          odds_format (str): The fomat you want the odds for the games to pull in. 
          Reference odds api docs.
          odds_date_format(str): The format you want the odds api dates to pull in. 
          Reference odds api docs."""
    def __init__(self, regions: str = 'us', markets: str = 'spreads,totals,h2h', odds_format: str = 'decimal',
                 odds_date_format: str = 'iso', api_key: str = odds_api_key, time_zone: str = 'America/New_York'):
        self.api_key: str = api_key
        self.regions: str = regions
        self.markets: str = markets
        self.odds_format: str = odds_format
        self.odds_date_format: str = odds_date_format
        self.time_zone: str = time_zone
        self.requests_remaining: int = None
        self.requests_used: int = None
        self.odds_called: bool = False # changed after odds api is called
        self.sport: str = None # defined in subclasses
        self.documentation = 'https://the-odds-api.com/liveapi/guides/v4/'

    def call_odds_api(self, retries: int = 3, delay: int = 5) -> dict:
        """
        Call the odds api and store the response as a json.

        This object has a self.documentation attr that has the odds api docs link in it
        if you want to reference those docs.        

        Returns:
          JSON file from odds api call. Reference odds api docs for more info.
        """
        for attempt in range(retries):
            try:
                response = requests.get(
                    f'https://api.the-odds-api.com/v4/sports/{self.sport}/odds',
                    params={
                        'api_key': self.api_key,
                        'regions': self.regions,
                        'markets': self.markets,
                        'oddsFormat': self.odds_format,
                        'dateFormat': self.odds_date_format
                    }
                )
                response.raise_for_status()
                # Jsonify the results
                odds_json = response.json()

                # Check the usage quota
                self.requests_remaining = response.headers['x-requests-remaining']
                self.requests_used = response.headers['x-requests-used']

                self.odds_called = True

                return odds_json
            except requests.exceptions.RequestException as e:
                if attempt < retries -1:
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Failed after {retries} attempts: {e}")
                

        if response.status_code != 200:
            print(f'Failed to get odds: status_code {response.status_code}, response body {response.text}')            
        
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
    
    def show_api_requests_used(self) -> None:
        print(f"You have used {self.requests_used} calls and have {self.requests_remaining} remaining.")  

class TeamRankingsScraperMixin:

    def set_teams_df(self, teams_df: pd.DataFrame):
        self.teams_df: pd.DataFrame = teams_df
        self.team_tr_table_key: dict[str, str] = self._build_team_tr_table_key() 
        self.url_ref_dict: dict[str, str] = self._build_tr_url_ref_dict()
        self.opponent_map: dict[str, str] = self._build_opponent_map()


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
        """Returns:
         A dict used to map the names of opponents on team schedule to names that will
        match what the names are in the Sports DB."""
        return {team_tr_name: url_name for team_tr_name, url_name in zip(self.teams_df['TR_TABLE_NAME'], 
                                                                         self.teams_df['TEAM_NAME'])}
    
    # This section is for OPPG and PPG stats calling and packing
    
    def _call_tr_ppg_and_oppg(self, league) -> list[pd.DataFrame]:
        """
        Call the info from the teamrankings site for points per game and opponents points per game.
        """
        
        ppg = pd.read_html(f"https://www.teamrankings.com/{league.lower()}/stat/points-per-game")
        oppg = pd.read_html(f"https://www.teamrankings.com/{league.lower()}/stat/opponent-points-per-game")

        ppg = ppg[0]
        oppg = oppg[0]
        return [ppg, oppg]

# Build an object that checks the tables in the db via a dataclass? ***
class Sports_Odds_DB_Packer(OddsApiCallerMixin, TeamRankingsScraperMixin):
    """
    This is a class buit to make getting data for a sports league from teamrankings.com, and the odds api
    and packing it into an SQLite database. In this case using another class, SqliteDBManager. This should
    be in a separate module included with this program.

    This is made specifically for the SportsData db created for the sports data app, but could be easily changed.
    I have tried to write it to be as extensible as possible. 
    """
    def __init__(self, db_name):
        """
        Define all the object you'll need to pack the odds api information not done in the odds api mixin object.

        Args:
          db_name (str): The name of the database you want to either create or refer to 
          with this object.
        """
        # Theses are the default args for the OddsApiCallerMixin, rewritten here for clarity.
        super().__init__(regions = 'us', markets = 'spreads,totals,h2h', odds_format = 'decimal',
                         odds_date_format = 'iso', time_zone = 'America/New_York')
        self.db_manager: SqliteDBManager = SqliteDBManager(db_name) # define the database you're calling from/to
        self.teams_df : pd.DataFrame = self._query_teams_table()
        self.set_teams_df(self.teams_df)        
        self.sport: str = None # defined in sublcasses
        self.db_last_updated_date: str = None # defined in sublcasses
        self.most_recent_game_date: str = None # defined in sublcasses 
        self.league_serial: int = None # defined in sublcasses
        self.odds_df: pd.DataFrame = None # info pulled from odds api
        self.combined_market_odds_df: pd.DataFrame = None # Combine odds_df and games_table info
        self.future_games_checked: bool = False # check if future games have been accounted for yet
        
         # League specified in subclasses    
        self.team_serial_ref_dict: dict[str, str] = self._build_team_serial_ref() # League specified in subclasses

    def _query_last_db_updated_date(self) -> str:
        """Get date the sports db was last updated.

        You can only call this function in a subclass to Sports_Odds_DB_Packer that has defined what league it is using.

        Returns:
          date (str): The last date the db was updated in %Y-%m-%d %H:%M:%S"""
        db_last_updated_date: str = self.db_manager.fetch_records(f"""SELECT MAX(LAST_UPDATED_DATE) FROM DB_UPDATE_RECORDS
                                          WHERE LEAGUE_SERIAL = {self.league_serial}""",fetchstyle='one')[0][0]
        return db_last_updated_date 
    
    def _query_most_recent_game_date(self) -> str:
        """Get date the sports db contains for a game.

        You can only call this function in a subclass to Sports_Odds_DB_Packer that has defined what league it is using.

        Returns:
          date (str): The last date the db was updated in %Y-%m-%d %H:%M:%S"""
        most_recent_game_date = self.db_manager.fetch_records(f"""SELECT MAX(GAME_DATE) FROM GAMES
                                               WHERE LEAGUE_SERIAL = {self.league_serial}""",fetchstyle='one')[0][0]
        return most_recent_game_date
        
    def _get_sport(self, league: str) -> str:
        """
        Retrieve odds api sport name for the api call from the database.
        This type of name is intended to go into a URL for the odds api.

        Args:
          League (str): The name of the league you are querying from the DB.
        """
        sport: str = self.db_manager.fetch_records(f"SELECT ODDS_URL_NAME FROM LEAGUES WHERE NAME ='{league}'",
                                   fetchstyle='one')
        return sport[0][0] # unpack tuple list to get val
    
    def _get_league_serial(self, league: str) -> int:
        """
        Retrieve odds api sport name for the api call from the database.

        Args:
          League (str): The name of the league you are querying from the DB.
        """
        serial: int = self.db_manager.fetch_records(f"SELECT SERIAL FROM LEAGUES WHERE NAME ='{league}'",
                                   fetchstyle='one')
        return serial[0][0] # unpack tuple list to get val
    
    # Query and create everything needed for all functionality here
    def _query_teams_table(self) -> pd.DataFrame:
        """Queries the TEAMS table in the db as a df for use in gathering teams data."""
        return self.db_manager.dataframe_query("select * from TEAMS")
    
    def _build_team_serial_ref(self) -> dict[str, int]:
        """For referencing the team name to the team serial for all teams in db for this league"""
        return {team_name:serial for team_name,serial in zip(self.teams_df['TEAM_NAME'],self.teams_df['SERIAL'])}
    
    def insert_new_db_updated_date(self) -> None:
        """Insert a new updated date into the db after updates are done"""
        now: str =  datetime.now().strftime(self.db_manager.db_date_format)

        self.db_manager.insert_table_records("DB_UPDATE_RECORDS", [(now,self.league_serial)])

    def is_new_week(self) -> bool:
        """Use datetime to check if the db_last_updated date is in the past week.
        
        Returns:
            bool: True if the most recent Sunday has passed since the last_updated_date, 
                indicating a new calendar week. False otherwise.
        """
        if not self.db_last_updated_date:
            raise AttributeError("""This object does not have a last updated date to check if 
                                 it's a new week.""")

        last_updated_date: str = self.db_last_updated_date
        date_format: str = self.db_manager.db_date_format

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
        
    def get_market_info(self) -> list[tuple[int,str]]:
        """Queries MARKETS table and returns a tuple for each market: (market serial, market name)"""

        markets_table = self.db_manager.dataframe_query('SELECT * FROM MARKETS')

        markets = []

        for item in markets_table.itertuples():
            markets.append(item[1:]) #drop index from db
        
        return markets
    
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

        odds_df_copy: pd.DataFrame = copy.deepcopy(self.odds_df)

        #use only one market id for the game set
        market_id = odds_df_copy['MARKET_SERIAL'].unique()[0]

        # Check the games from the odds API and compare to what is currently in the db by date
        # Eliminate duplicate games by only grabbing games from one market
        try:
            if self.most_recent_game_date:
                unmade_games: pd.DataFrame = odds_df_copy[(odds_df_copy['Date'] > self.most_recent_game_date) &
                                    odds_df_copy['MARKET_SERIAL'] == market_id]
            else:
                unmade_games: pd.DataFrame = odds_df_copy[odds_df_copy['MARKET_SERIAL'] == market_id]
        except KeyError as e:
            raise KeyError(f"""Looks like when you called the odds api the columns in the df did not populate
                           correctly. See specific error: {e}. See self.get_odds_df for more.""")
        # Add the league serial so it can be added to the games table
        unmade_games['league_serial'] = self.league_serial

        # Drop all columns except for what is needed for the games table
        unmade_games = unmade_games[['home_serial', 'away_serial', 'Date', 'league_serial']]

        if unmade_games.empty:
            print("No unmade games to add to the db")
            self.future_games_checked = True
            return None

        self.db_manager.df_to_db('GAMES', unmade_games, debug=debug)

        if not debug:
            self.future_games_checked = True

    def get_uid(self, df: pd.DataFrame, home_serial_col_name:str, away_serial_col_name:str, 
                date_col_name: str, debug: bool = False) -> pd.DataFrame:
        """Takes in a df along with the col names needed in order to form a new df that
        only uses the day of the date (formatted %Y:%m:%d %H:%M:%S) to create a join between the dfs instead of
        the full date time that is stored in this database.

        It will combine the home team serial, away team serial, and day of the game into one string creating
        a unique id for each game.
        
        It will add a day & time column along with the original date column, it won't eliminate
        the date column.
        
        Args:
          df (pd.DataFrame): original dataframe
          home_serial_col_name (str): The name of the column with the home serial
          away_serial_col_name (str): The name of the column with the away serial
          date_col_name (str): The name of the column with the date (formatted %Y:%m:%d %H:%M:%S)
          debug (bool): A boolean you can set to true to help debug the function
          
        Returns:
          df (pd.DataFrame): Same df you put in PLUS three cols: day, time, and UID."""
        
        if df.empty:
            raise ValueError ("Provided df is empty. Need values for this work")
        # Validate date column format
        try:
            if not debug:
                df[date_col_name] = pd.to_datetime(df[date_col_name], format='%Y-%m-%d %H:%M:%S')
            else:
                copied_df = copy.deepcopy(df)
                copied_df[date_col_name] = pd.to_datetime(copied_df[date_col_name], format='%Y-%m-%d %H:%M:%S')
                copied_df['day'] = copied_df[date_col_name].dt.date.astype(str)  # Extracts the day (YYYY-MM-DD)
                print(f"Day format of col: {copied_df['day'].values[0]}")
                copied_df['time'] = copied_df[date_col_name].dt.time.astype(str)  # Extracts the time (HH:MM:SS)
                print(f"Time format of col: {copied_df['time'].values[0]}")
                return "Run again without debug to create the df with the day, time, UID cols."
        except ValueError as e:
            raise ValueError(f"Date column '{date_col_name}' contains invalid formats. Expected format is '%Y-%m-%d %H:%M:%S'. Error: {e}")           
        
        # Extract day and time
        df['day'] = df[date_col_name].dt.date.astype(str)  # Extracts the day (YYYY-MM-DD)
        df['time'] = df[date_col_name].dt.time.astype(str)  # Extracts the time (HH:MM:SS)

              
        df['UID'] = df['UID'] = (
                df[home_serial_col_name].astype(str) + ' ' +
                df[away_serial_col_name].astype(str) + ' ' +
                df['day'].astype(str)
            )
        
        return df

    def _combine_odds_api_and_games_dfs(self) -> pd.DataFrame:
        """Check that the correct columns are grabbed and combine the games table and odds df
        info so that they are ready to go into the market odds table."""

        if not getattr(self, 'odds_called', True):
            try:
                self.get_odds_df()
            except Exception as e:
                raise AttributeError(f"Odds api df not yet created. Use self.get_odds_df to fill. Error: {e}")
            
        else:
            copied_odds_df: pd.DataFrame = copy.deepcopy(self.odds_df)

            copied_odds_df = self.get_uid(copied_odds_df, 'home_serial', 'away_serial', 'Date') 
            
            games_df = self.db_manager.dataframe_query('select * from GAMES')
            games_df = self.get_uid(games_df,'HOME_TEAM_SERIAL','AWAY_TEAM_SERIAL','GAME_DATE')
                
            # Join the team df to the current db games data using the UID 
            joined_df = pd.merge(copied_odds_df,games_df, on='UID', how='inner')
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
            
             # Retrieve game serials to query for
            game_serials: list[str]= self.combined_market_odds_df['SERIAL'].unique()   
            existing_game_data = self.db_manager.dataframe_query(f"""SELECT * FROM GAME_MARKET_ODDS 
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
                        'GAME_SERIAL': api_row['SERIAL'],
                        'MARKET_SERIAL': api_row['MARKET_SERIAL'],
                        'SPREAD_HOME': api_row['SPREAD_HOME'],
                        'TOTAL': api_row['TOTAL'],
                        'H2H_HOME': api_row['H2H_HOME'],
                        'H2H_AWAY': api_row['H2H_AWAY']
                    })

        new_entries = pd.DataFrame(new_rows)

        return new_entries

    def refresh_market_odds_table(self, debug: bool = False) -> None:
        """Get the existing data and compare it to the newly pulled api market odds data.
        Then insert any data that has changed on the new odds api pull."""
        # Need to have called the odds api for this to work
        if getattr(self, 'odds_called', False):
            try:
                self.get_odds_df()
            except Exception as e:
                raise Exception(f"Could not get odds_df. Troubleshoot starting there. Error was: {e}")
        # Need to have combined the odds api and games table info for this to work
        if not hasattr(self, 'combined_market_odds_df'):
            try:
                self._combine_odds_api_and_games_dfs()
            except Exception as e:
                raise Exception(f"Could not get combined_market_odds_df. Troubleshoot starting there. Error was: {e}")
        # Execute assuming that all the correct info is in place to refresh any changed odds data.
        else:
            existing_mkt_odds: pd.DataFrame = self.get_existing_market_odds_records_for_called_games()
            api_mkt_odds: pd.DataFrame = copy.deepcopy(self.combined_market_odds_df)
            api_mkt_odds = api_mkt_odds.drop('Date', axis=1) # drop date column for comparison

            new_entries = self.get_changed_market_odds(existing_data=existing_mkt_odds,api_data=api_mkt_odds)


            if new_entries.empty:
                print("No game odds that exist have changed.")
                return None
            else:
                try:
                    self.db_manager.df_to_db("GAME_MARKET_ODDS",new_entries,debug=debug)
                except Exception as e:
                    raise RuntimeError(f"Failed to insert records for market odds tabel refresh function:\n {e}")

    def pack_odds_table(self, debug: bool = False) -> None:
        """Compare the most recently added game odds to what is in the api data, then
        add the data to the game_market_odds table."""
        
        # Get the most recent game date from DB
        most_recent_game_date_has_odds: list[tuple] = self.db_manager.fetch_records("""
                        SELECT 
                            G.GAME_DATE AS GAME_DATE
                        FROM 
                            GAMES AS G JOIN GAME_MARKET_ODDS AS GMO ON G.SERIAL = GMO.GAME_SERIAL
                        WHERE 
                            G.GAME_DATE = (SELECT MAX(G.GAME_DATE) FROM 
                                            GAMES AS G JOIN GAME_MARKET_ODDS AS GMO 
                                            ON G.SERIAL = GMO.GAME_SERIAL)
                        """, fetchstyle='one')
        
        if most_recent_game_date_has_odds:
            most_recent_game_date_has_odds = most_recent_game_date_has_odds[0][0]# unpack val from list of tuples    
            # Filter market odds by the game date
            odds_to_add: pd.DataFrame = self.combined_market_odds_df[(self.combined_market_odds_df['Date'] > 
                                                                    most_recent_game_date_has_odds)]

        if odds_to_add.empty:
            print("No odds to add right now.")
            return None
        
        # Drop game date so df can go in market odds table
        odds_to_add = odds_to_add.drop('Date', axis=1)

        if debug:
            print(odds_to_add)
        try:
            self.db_manager.df_to_db('GAME_MARKET_ODDS', odds_to_add, debug=debug)
        except Exception as e:
                    raise RuntimeError(f"Failed to insert records for market odds pack odds table function:\n {e}")
    
    def full_odds_run(self):
        # Run the full odds api process and insert the records.
        self.get_odds_df()
        self.populate_future_game_data()
        self._combine_odds_api_and_games_dfs()
        self.refresh_market_odds_table()
        self.pack_odds_table()


# --------------------------------------- NFL DATA PACKER ---------------------------------------------

class NFL_Data_Packer(Sports_Odds_DB_Packer):
    """
    This object is intended to get NFL data and send it into the Sports Data DB.

    If a func name has tr, it means it has to do with Team Rankings data. If it has
    odds, it has to do with the odds api.
    """

    def __init__(self, db_name: str, league_name: str = 'NFL', debug = False):
        """
        Here we query the Sports DB for the league that is chosen in the data packer,
        get the odds info from the inhereted 

        We also get all the team data for that league and create keys for later data
        translation from the different data sources so they match db info.

        Args:
          db_name (str): The name of the database you want to either create or refer to 
          with this object.
        """
        super().__init__(db_name)   
        self.debug = debug
        self.league: str = league_name
        self.sport : str = self._get_sport(self.league)
        self.league_serial: int = self._get_league_serial(self.league)        
        self.db_last_updated_date: str = self._query_last_db_updated_date()
        self.most_recent_game_date: str = self._query_most_recent_game_date() 
        self.odds_df: pd.DataFrame = None
        self.all_games_data: dict[str,pd.DataFrame] = None
        self.games_data_retrieved: bool = False
    
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
        ppg, oppg = self._call_tr_ppg_and_oppg(self.league)
        # Get cols in line with DB formatting
        ppg['TEAM_SERIAL'] = [self.team_tr_table_key[key] for key in ppg['Team']]
        oppg['TEAM_SERIAL'] = [self.team_tr_table_key[key] for key in oppg['Team']]

        ppg = self._reorder_team_data_ppg_oppg(ppg)
        oppg = self._reorder_team_data_ppg_oppg(oppg)
        # Send data into database
        self.db_manager.df_to_db('TEAM_TR_FOOTBALL_PPG', ppg, debug=debug)
        self.db_manager.df_to_db('TEAM_TR_FOOTBALL_OPPG', oppg, debug=debug)

    # This section is for making our own predictions
    def get_team_ppg_data(self):
        return self.db_manager.dataframe_query(
        f"""
        SELECT
            TEAMS.SERIAL,
            PPG.THIS_YEAR_PPG AS PPG_2024,
            OPPG.THIS_YEAR_PPG AS OPPG_2024,
            PPG.LAST_3 AS PPG_LAST_3,
            OPPG.LAST_3 AS OPPG_LAST_3,
            PPG.HOME AS PPG_HOME,
            OPPG.HOME AS OPPG_HOME,
            PPG.AWAY AS PPG_AWAY,
            OPPG.AWAY AS OPPG_AWAY
        FROM
            TEAMS AS TEAMS
            JOIN TEAM_TR_FOOTBALL_PPG AS PPG ON TEAMS.SERIAL = PPG.TEAM_SERIAL
            JOIN TEAM_TR_FOOTBALL_OPPG AS OPPG ON TEAMS.SERIAL = OPPG.TEAM_SERIAL
        WHERE
            TEAMS.LEAGUE_SERIAL = {self.league_serial} AND
            PPG.UPDATED_DATE = (SELECT MAX(UPDATED_DATE) FROM TEAM_TR_FOOTBALL_PPG)
            AND OPPG.UPDATED_DATE = (SELECT MAX(UPDATED_DATE) FROM TEAM_TR_FOOTBALL_OPPG)
        ORDER BY
            TEAMS.TEAM_NAME
        """)

    def pack_own_odds_predictions(self) -> None:
        """
        This will be a function that packs the own_odds table for each team in games
        upcoming.
        """
        pass
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
        df['Date'] = pd.to_datetime(df['Date'] + f'/{current_year} ').dt.strftime(self.db_manager.db_date_format)

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


            # Get most recent game with outcome data
            most_recent_game_w_outcome = self.db_manager.fetch_records("""
                    SELECT MAX(G.GAME_DATE)
                    FROM GAMES AS G JOIN GAME_OUTCOMES AS GO ON G.SERIAL = GO.GAME_SERIAL                      
                    """, fetchstyle='one')[0][0]
            # Process data
            games_data = games_data[games_data['Location'] == 'Home']
            games_data = self.format_team_game_history(games_data)
            if most_recent_game_w_outcome:
                games_data = games_data[games_data['Date'] >= most_recent_game_w_outcome]

            # Check if DataFrame is empty
            if games_data.empty:
                print(f"{scraper.team_name} has no game updates since {most_recent_game_w_outcome}")
                time.sleep(10)
                continue  # Skip this team if no data matches the criteria

            # Add metadata
            games_data['League Serial'] = self.league_serial
            games_data['home_serial'] = self.get_team_serial(scraper.team_name)
            games_data['away_serial'] = games_data['Opponent'].apply(self.get_team_serial)

            # Assign processed data to the dictionary
            games_data_by_team[scraper.team_name] = games_data
            print(f"{scraper.team_name} added {len(games_data)} games since {most_recent_game_w_outcome}.")
            # Respect crawl delay
            time.sleep(10)

        self.all_games_data = games_data_by_team
        self.games_data_retrieved = True

    def pack_games_data(self, acknowledged: bool = False) -> None:
        """This function is setup to pack all the games data into the GAMES table in the db.
        This should only have to be run to catch up if the scripts haven't run in a while.
        
        The odds api provides future games that get packed in, so running this function should
        be rare. As a result you have to acknowledge its usage."""
        if not self.games_data_retrieved:
            try:
                self.get_all_games_data()
            except Exception:
                raise ValueError("Game data not yet packed, run self.pack_games_data().")
        if not acknowledged:
            raise ValueError("""This function is setup to pack all the games data into the GAMES table in the db.
                                This should only have to be run to catch up if the scripts haven't run in a while.
                                
                                The odds api provides future games that get packed in to the db, so running this function should
                                be rare. As a result you have to acknowledge its usage.
                                                    
                                Rerun with acknowledged as True to run the function.""")

        games_table_data = copy.deepcopy(self.all_games_data)
        if self.most_recent_game_date:
            games_table_data = games_table_data[games_table_data['Date'] >= self.most_recent_game_date]

        for team, df in games_table_data.items():
            df = df[['home_serial', 'away_serial', 'Date', 'League Serial']]
            self.db_manager.df_to_db('GAMES', df)

    def pack_game_outcomes_data(self, debug: bool= False) -> None:

        if not self.games_data_retrieved:
            try:
                self.get_all_games_data()
            except Exception as e:
                raise ValueError(f"""Game data not yet retrieved, run self.get_all_games_data(). 
                                 The program did try to run that func, here's what we got: 
                                 {e}""")
        # Create a copy of the retrieved all games data dict
        game_outcomes_table_data = copy.deepcopy(self.all_games_data)

        # Call all current game data in the db
        games_df = self.db_manager.dataframe_query('select * from GAMES')
        games_df = self.get_uid(games_df,'HOME_TEAM_SERIAL','AWAY_TEAM_SERIAL','GAME_DATE')
        
        for team, df in game_outcomes_table_data.items():
            # Process each df and add UID
            df = df[['Date', 'home_serial','away_serial','Result','Points','Opp Points']]
            df = self.get_uid(df, 'home_serial', 'away_serial', 'Date')  
            # Join the team df to the current db games data using the UID 
            joined_df = pd.merge(df,games_df, on='UID', how='inner')
            # Identify the winner for each game by team serial
            joined_df['winner_serial'] = joined_df.apply(
                lambda row : row['home_serial'] if row['Result'] == 'W' else row['away_serial'],
                axis= 1
            )
            # Drop unecessary data and insert into game outcomes table
            joined_df = joined_df[['SERIAL', 'winner_serial', 'Points', 'Opp Points']]
            self.db_manager.df_to_db('GAME_OUTCOMES', joined_df, debug=debug)

        print(f"Game outcomes data added, added {len(joined_df)} records.")

    def pack_all_games_data(self, acknowledged: bool = False) -> None:
        """
        Combining all funcs needed to get all the prior games and game outcomes for the
        entire season. This shouldn't be called every time.
        Args:
            acknowledged (bool): Acknowledge your understanding of the risks of doubling 
            the data
        """                
        self.pack_games_data(acknowledged=acknowledged)
        self.pack_game_outcomes_data()
    # Put it all together

    def full_pack(self, pack_all_games_data: bool = False, acknowledged: bool = False) -> None:
        f"""This func is designed to run a standard data gathering for the {self.league}.
        
        A standard run will fill future game data and grab odds for any games from the odds
        api which have not already been populated and check to see if any of the odds have
        changed.
        
        It will also updated OPPG and PPG stats for all the teams.
        
        Weekly, it will grab all the games for all the teams and update the outcomes table.
        
        Args:
            pack_all_games_data (bool): If true, this will get all the 
            game data for the season and put it in the games table.
            acknowledged (bool): acknowledging that you know the risks that come
            with duplicating a ton of data if you pack the games data for the
            whole season. """
        
        # get PPG OPPG stats
        print("Fetching PPG and OPPG Data")
        self.pack_ppg_oppg()

        # Get odds data
        if not pack_all_games_data:
            print("Fetching odds data")   
            self.full_odds_run()
            # Check if it's a new week yet and get game outcomes data if so.
            self.pack_own_odds_predictions()
            if self.is_new_week():
                print("Fetching game outcomes data.")
                self.pack_game_outcomes_data()

        # Pack all the games data if option is selected - NOT standard run.
        # Use in case of a fresh db.
        if pack_all_games_data:
            print("Fetching all games data.")
            self.pack_all_games_data(acknowledged=acknowledged)
            print("Fetching odds data")          
            self.full_odds_run()
                

        self.insert_new_db_updated_date() # Insert new updated record into the db
        print('Packed all data.') # Need to write a logger so we get more than this lol

