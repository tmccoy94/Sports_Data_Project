import pandas as pd
import sqlite3
import datetime as dt

# add funcs for checking the columns and the data types later.*


class SqliteDBManager:
    """
    A class intended to a be a flexible Sqlite DB manager that can fairly simply craft
    databases and retrieve information from them. All basic types of queries are present
    for creating and altering tables, as well as inserting records into them.

    Use fetch records to retrieve records directly and use the dataframe query to retrieve your
    query results as a dataframe.

    All basic error handling is included through the use of these funcs.
    """
    def __init__(self, db_name):
        """
        Initialize the SportsDBManager with a default database name.
        """
        self.db_name = db_name #'SportsData.db'
        self.conn = None

    def connect(self):
        """Establish a connection to the database if not already connected."""
        if not self.conn:
            self.conn = sqlite3.connect(self.db_name)
            self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self):
        """Close the database connection if open."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def check_table_names_in_db(self) -> list[str]:
        """Check for tables in the database."""
        stmt = "SELECT name FROM sqlite_master WHERE type='table';"
        result =self.fetch_records(stmt, fetchstyle='all')
        ls = []
        for name in result:
            ls.append(name[0])
        return ls
        

    def check_table_info(self, table_name, help: bool = False):
        """
        Check the structure of a table in the database.

        Args:
            table_name (str): Name of the table to check.
            help (bool): If True, print help information.
        """
        if help:
                print("The tuples in the result mean the  following:\ncolumn_id: 0 for first column, 1 for second column, etc.\nname: Column name\ndata_type: Data type of the column\nnotnull: 0 for can be NULL, 1 for cannot be NULL\ndefault_value: None for no default value\nprimary_key: 1 if this column is part of the primary key, 0 otherwise\nhidden: 0 for visible, 1 for hidden")
        
        cols = []
        for col in self.fetch_records(f"PRAGMA table_xinfo({table_name})"):
            cols.append(col)

        return cols


    def _read_query(self, query: str, params: tuple = None, ) -> None:
        """
        Execute a database query with error handling.

        Args:
            query (str): The SQL query to execute.
            params (tuple, optional): Parameters to include in the query.
        """
        try:
            self.connect()
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
                result = cursor.fetchall()
                print(result)
            else:
                cursor.execute(query)
                result = cursor.fetchall()
                print(result)
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            self.close()

    def _execute_query(self, query: str, params: tuple = None, commit: bool = False) -> None:
        """
        Execute a database query with error handling.

        Args:
            query (str): The SQL query to execute.
            params (tuple, optional): Parameters to include in the query.
            commit (bool): Whether to commit the transaction (useful for INSERT, UPDATE, DELETE).
        """
        try:
            self.connect()
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            if commit:
                self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise
        finally:
            self.close()

    def fetch_records(self, query: str, fetchstyle: str, size: int = None) -> list[tuple]:
        """
        Fetch records from the database based on the specified retrieval style.

        Args:
            query (str): The SQL query to execute.
            fetchstyle (str): The fetch style to use; must be one of 'one', 'many', or 'all'.
                            - 'one': Returns a single record as a tuple.
                            - 'many': Returns a limited number of records as a list of tuples.
                            - 'all': Returns all matching records as a list of tuples.

        Example usage:
            # To fetch a single record
            self.fetch_records("SELECT * FROM TEAMS WHERE SERIAL = 1", fetchstyle="one")

            # To fetch all records
            self.fetch_records("SELECT * FROM TEAMS", fetchstyle="all")
        """
        # Validate inputs
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if fetchstyle not in ('one', 'many', 'all'):
            raise ValueError("fetchstyle must be 'one', 'many', or 'all'")
        if fetchstyle == 'many' and size == None:
            raise ValueError("Using fetchmany() you must specify the number of rows.")
        if size and not isinstance(size, int):
            raise TypeError("Size must be an integer")

        try:
            # Connect to the database and execute the query
            self.connect()
            cursor = self.conn.cursor()
            cursor.execute(query)

            # Fetch results based on the specified fetch style
            if fetchstyle == 'one':
                result = cursor.fetchone()
                return [result] if result else []
            elif fetchstyle == 'many':
                result = cursor.fetchmany(size)
                return result if result else []
            else:  # 'all'
                result = cursor.fetchall()
                return result if result else []

        except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise

        finally:
            # Ensure the database connection is closed
            self.close()


    def dataframe_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a pandas DataFrame.
        Args:
            query (str): The SQL query to execute.
        Returns:
            pd.DataFrame: The result of the query as a pandas DataFrame.
        """
        try:
            self.connect()
            df = pd.read_sql_query(query, self.conn)
            return df
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            self.close()

    def create_table(self, table_name: str, columns: list[tuple] = [], debug: bool = False) -> None:
        """
        Create a table with the specified name and columns, including support for foreign keys.

        Args:
            table_name (str): Name of the table.
            columns (list of tuples): Each tuple should contain (column name, data type, additional_constraint).
                                    additional_constraint can be 'PRIMARY KEY' or a foreign key definition 
                                    like ('LEAGUE_SERIAL', 'INTEGER', 'REFERENCES LEAGUES(SERIAL)').
            debug (bool): If True, print debug information.

        Example usage:
        To create a table with a foreign key:
            self.create_table('TEAMS', [
                ('SERIAL', 'INTEGER', 'PRIMARY KEY'),
                ('LEAGUE_SERIAL', 'INTEGER', 'REFERENCES LEAGUES(SERIAL)'),
                ('TEAM_NAME', 'TEXT')
            ], debug=True)
        """
        if not columns:
          raise TypeError("columns must be a non-empty list")
        if not isinstance(columns[0], tuple):
          raise TypeError("columns must be a list of tuples")
        if not isinstance(table_name, str):
          raise TypeError("table_name must be a string")
        if not isinstance(debug, bool):
          raise TypeError("debug must be a boolean")
        # Build the column definitions from the list of tuples
        column_defs = []
        for col in columns:
            if 'TIMESTAMP' in col:
                column_defs.append(col)
                continue
            col_def = f"{col[0]} {col[1]}"
            if len(col) == 3:
                constraint = col[2].upper()
                if constraint == "PRIMARY KEY":
                    col_def += " PRIMARY KEY"
                elif "FOREIGN KEY" in constraint:
                    col_def = f"{constraint}"  # Use the full foreign key constraint as is
            column_defs.append(col_def)

        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_defs)})"
        # Debug mode check
        if debug:
            print(f"create_table_query: {create_table_query}")
            return ("Run without debug to commit")

        # Execute the create table query
        self._execute_query(create_table_query, commit=True)
        print(f"Table '{table_name}' created successfully.")

    def insert_table_records(self, table_name: str, records: list[tuple] = [], debug: bool = False) -> None:
      """
      Insert multiple records into a table.

      Args:
          table_name (str): Name of the table.
          records (list of tuples): Each tuple represents a record to be inserted.
          debug (bool): If True, print debug information.
      """
      # Check all basic issues with function inputs
      if not records:
        raise TypeError("records must be a non-empty list")
      if not isinstance(records[0], tuple):
        raise TypeError("records must be a list of tuples")
      if not isinstance(table_name, str):
        raise TypeError("table_name must be a string")
      if not isinstance(debug, bool):
        raise TypeError("debug must be a boolean")

      try:
        # connect to db
        self.connect()
        cur = self.conn.cursor()
        # get column names of table
        cur.execute(f"PRAGMA table_xinfo({table_name})")
        # Get columns, remove any primary keys or timestamps (they autofill the vals)
        columns = [row[1] for row in cur.fetchall() if row[2] != 'TIMESTAMP' and row[5] != 1]
        # check if number of columns in table matches number of columns in records
        if len(columns) != len(records[0]):
          raise ValueError("Number of columns in table and length of the first record tuple do not match")
        column_names = f"{', '.join(columns)}"
        marks = ', '.join(['?'] * len(columns))
        # write query
        insert_query = f"""INSERT INTO {table_name} ({column_names})
        VALUES ({marks})
        """
        if debug:
          print(f"""insert query: {insert_query}
          1st record: {records[0]}
          """)
          return ("Run without debug to commit")
        # execute & commit
        cur.executemany(insert_query,records)
        self.conn.commit()
        print(f"Records inserted successfully into {table_name}")
      except sqlite3.Error as e:
        print(f"Database error: {e}")
      except TypeError as e:
        print(f"Type error test: {e}")
      finally:
        self.close()

    def update_table_records(self, table_name: str, values_to_update: dict, conditions: dict, debug: bool = False) -> None:
      """
      Update selected records in a table based on conditions.

      Args:
          table_name (str): Name of the table.
          values_to_update (dict): Dictionary of columns and values to update.
          conditions (dict): Dictionary of columns and values to match for selecting records to update.
          debug (bool): If True, print debug information.

      Example of updating records
        values_to_update = {'TEAM_NAME': 'New Team Name', 'SPORT': 'Basketball'}
        conditions = {'SERIAL': 1}

      Run the update
        self.update_table_records('TEAMS', values_to_update, conditions, debug=False)
      """
      # Validate inputs
      if not values_to_update:
          raise ValueError("values_to_update must be a non-empty dictionary")
      if not conditions:
          raise ValueError("conditions must be a non-empty dictionary")
      if not isinstance(table_name, str):
          raise TypeError("table_name must be a string")
      if not isinstance(debug, bool):
          raise TypeError("debug must be a boolean")
      # Construct the SET part of the SQL statement
      set_clause = ', '.join([f"{col} = ?" for col in values_to_update.keys()])
      set_values = list(values_to_update.values())

        # Construct the WHERE part
      where_clause = ' AND '.join(
            [f"{col} {cond}" if cond in ["IS NULL", "IS NOT NULL"] else f"{col} = ?" 
            for col, cond in conditions.items()]
        )
      where_values = [val for val in conditions.values() if val not in ["IS NULL", "IS NOT NULL"]]

        # Combine SET and WHERE parts into full query
      query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
      values = set_values + where_values

      # Debug print
      if debug:
          print(f"Update query: {query}")
          print(f"Query parameters: {values}")
          return "Run without debug to commit the changes."

      # Execute and commit
      
      self._execute_query(query,values,commit=True)
      print("Records updated successfully.")

    def delete_table_records(self, table_name: str, conditions: dict, debug: bool = False) -> None:
      """
      Delete selected records from a table based on conditions.

      Args:
          table_name (str): Name of the table.
          conditions (dict): Dictionary of columns and values to match for selecting records to delete.
          debug (bool): If True, print debug information.

      Example of deleting records
        conditions = {'SERIAL': 1, 'TEAM_NAME': 'Old Team Name'}

      Run the delete
        self.delete_table_records('TEAMS', conditions, debug=True)
      """
      # Validate inputs
      if not conditions:
          raise ValueError("conditions must be a non-empty dictionary")
      if not isinstance(table_name, str):
          raise TypeError("table_name must be a string")
      if not isinstance(debug, bool):
          raise TypeError("debug must be a boolean")

      
      # Prepare the delete query
      where_clause = ' AND '.join([f"{col} = ?" for col in conditions.keys()])
      delete_query = f"DELETE FROM {table_name} WHERE {where_clause}"
      
      # Collect condition values for the query
      query_params = tuple(conditions.values())

      # Debug print
      if debug:
          print(f"Delete query: {delete_query}")
          print(f"Query parameters: {query_params}")
          return "Run without debug to commit the changes."

      # Execute and commit
      self._execute_query(delete_query,query_params,commit=True)
      print("Records deleted successfully.")

    def truncate_table(self, table_name: str, debug: bool = False) -> None:
      """
      Remove all records from a table, effectively resetting it.

      Args:
          table_name (str): Name of the table to truncate.
          debug (bool): If True, print debug information without executing.

      Example usage:
        To truncate the TEAMS table:
            self.truncate_table('TEAMS', debug=True)
      """
      # Validate inputs
      if not isinstance(table_name, str):
          raise TypeError("table_name must be a string")
      if not isinstance(debug, bool):
          raise TypeError("debug must be a boolean")

      # Prepare the truncate query (equivalent to TRUNCATE)
      truncate_query = f"DELETE FROM {table_name}"

      # Debug print
      if debug:
          print(f"Truncate query: {truncate_query}")
          return "Run without debug to commit the changes."

      # Execute and commit
      self._execute_query(truncate_query, commit=True)
      print(f"Table '{table_name}' truncated successfully.")

    def copy_table(self, new_table_name: str, old_table_name: str, debug: bool = False) -> None:
      """
      Copy records from one table into another new table, with optional alterations

      Args:
          new_table_name (str): Name of the new table to reference.
          old_table_name (str): Name of the old table to reference.
          debug (bool): If True, print debug information without executing.

      Example usage:
        To truncate the TEAMS table:
            self.truncate_table('TEAMS', debug=True)
      """
      # Validate inputs
      if not isinstance(new_table_name, str):
          raise TypeError("new_table_name must be a string")
      if not isinstance(old_table_name, str):
          raise TypeError("old_table_name must be a string")
      if not isinstance(debug, bool):
          raise TypeError("debug must be a boolean")
      
      old_table_info = self.fetch_records(F"PRAGMA table_xinfo({old_table_name})",fetchstyle='all')
      new_table_info = self.fetch_records(F"PRAGMA table_xinfo({new_table_name})",fetchstyle='all')

      old_table_cols = [row[1] for row in old_table_info]
      new_table_cols = [row[1] for row in new_table_info]

      if old_table_cols != new_table_cols:
          raise ValueError(f"The columns from {old_table_name} and {new_table_name} do not match.")
      
      cols_for_query = ', '.join(old_table_cols)

      # Prepare the truncate query (equivalent to TRUNCATE)
      copy_query = f"""INSERT INTO {new_table_name} ({cols_for_query})
                        SELECT {cols_for_query}
                        FROM {old_table_name};"""

      # Debug print
      if debug:
          print(f"Truncate query: {copy_query}")
          return "Run without debug to commit the changes."

      # Execute and commit
      self._execute_query(copy_query, commit=True)
      print(f"Table '{old_table_name}' copied into '{new_table_name}' successfully.")

    def drop_table(self, table_name: str, debug: bool = False) -> None:
      """
      Delete a table completely from the database.

      Args:
          table_name (str): Name of the table to delete.
          debug (bool): If True, print debug information without executing.

      Example usage:
        To delete the TEAMS table:
            self.drop_table('TEAMS', debug=True)
      """
      # Validate inputs
      if not isinstance(table_name, str):
          raise TypeError("table_name must be a string")
      if not isinstance(debug, bool):
          raise TypeError("debug must be a boolean")

      # Prepare the drop table query
      drop_query = f"DROP TABLE IF EXISTS {table_name}"

      # Debug print
      if debug:
          print(f"Drop table query: {drop_query}")
          return "Run without debug to commit the changes."

      # Execute and commit
      self._execute_query(drop_query, commit=True)
      print(f"Table '{table_name}' deleted successfully.")

    def alter_table(self, table_name: str, operation: str, column_name: str = None, new_name: str = None, 
                    data_type: str = None, debug: bool = False) -> None:
      """
      Alter a table to add a column, rename a column, or rename the table.

      Special note:
        This func can handle timestamps if you put dtype as timestamp. Will create that
        col with a defuault current_timestamp.

      Args:
          table_name (str): Name of the table to alter.
          operation (str): Type of alteration ('add_column', 'rename_column', 'rename_table').
          column_name (str, optional): The name of the column to add or rename (required for column operations).
          new_name (str, optional): The new name for a column or table (required for renaming).
          data_type (str, optional): Data type for the new column if adding (required for 'add_column').
          debug (bool): If True, print debug information without executing.

      Example usage:
        To add a column to the TEAMS table:
            self.alter_table('TEAMS', operation='add_column', column_name='CITY', data_type='TEXT', debug=True)

        To rename a column in the TEAMS table:
            self.alter_table('TEAMS', operation='rename_column', column_name='OLD_NAME', new_name='NEW_NAME', debug=True)

        To rename the table:
            self.alter_table('TEAMS', operation='rename_table', new_name='NEW_TEAMS', debug=True)
      """
      # Validate inputs
      if not isinstance(table_name, str):
          raise TypeError("table_name must be a string")
      if not isinstance(operation, str) or operation not in ['add_column', 'rename_column', 'rename_table']:
          raise ValueError("operation must be one of 'add_column', 'rename_column', or 'rename_table'")
      if not isinstance(debug, bool):
          raise TypeError("debug must be a boolean")

      # Prepare the alteration query based on the operation type
      if operation == 'add_column':
          if not column_name or not data_type:
            raise ValueError("For 'add_column', both 'column_name' and 'data_type' must be provided.")
          if data_type == 'TIMESTAMP':
            alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
          else:
            alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}"

      elif operation == 'rename_column':
          if not column_name or not new_name:
              raise ValueError("For 'rename_column', both 'column_name' and 'new_name' must be provided.")
          alter_query = f"ALTER TABLE {table_name} RENAME COLUMN {column_name} TO {new_name}"

      elif operation == 'rename_table':
          if not new_name:
              raise ValueError("For 'rename_table', 'new_name' must be provided.")
          alter_query = f"ALTER TABLE {table_name} RENAME TO {new_name}"

      # Debug print
      if debug:
          print(f"Alter table query: {alter_query}")
          return "Run without debug to commit the changes."

      # Execute and commit
      self._execute_query(alter_query, commit=True)
      print(f"Table '{table_name}' altered successfully with operation '{operation}'.")

