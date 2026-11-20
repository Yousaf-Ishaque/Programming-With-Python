
"""
database.py


Generates and handles the SQLite database utilizing SQLAlchemy.
Stores Pandas DataFrames in the form of database tables and
enables methods for examining the stored tables.
"""

from pathlib import Path
import re

import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError


class DatabaseManager:
    """Gnerates, handles and examines SQLite database."""

    def __init__(self, database_path=None):
        """Generates the database engine and database path."""

        project_dir = Path(__file__).resolve().parent

        if database_path is None:
            output_dir = project_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            database_file = output_dir / "assignment.db"
            database_path = f"sqlite:///{database_file.as_posix()}"

        self.database_path = database_path
        self.engine = create_engine(self.database_path)

    def create_database(self):
        """Generates the database connection and validates the availability."""

        try:
            with self.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")

            print("Database created successfully!")

        except SQLAlchemyError as error:
            raise RuntimeError(
                f"Unable to create or connect to database: {error}"
            ) from error

    def get_engine(self):
        """Returns the SQLAlchemy database engine"""

        return self.engine

    def save_dataframe(self, dataframe, table_name):
        """Saves a Pandas DataFrame in the form of a database table."""

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")

        if (
            not isinstance(table_name, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name)
        ):
            raise ValueError("Invalid database table name.")

        if len(dataframe.columns) == 0:
            raise ValueError(
                "DataFrame must contain at least one column."
            )

        try:
            # Use a context-managed connection so it is closed
            # after the DataFrame has been saved.
            with self.engine.begin() as connection:
                dataframe.to_sql(
                    name=table_name,
                    con=connection,
                    if_exists="replace",
                    index=False
                )

            print(
                f"Table '{table_name}' saved successfully. "
                f"Records: {len(dataframe)}"
            )

        except SQLAlchemyError as error:
            raise RuntimeError(
                f"Unable to save table '{table_name}': {error}"
            ) from error

    def get_table_names(self):
        """Returns the titles of all tables in the database."""

        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()

        except SQLAlchemyError as error:
            raise RuntimeError(
                f"Unable to retrieve database tables: {error}"
            ) from error

    def get_table_columns(self, table_name):
        """Returns the columns names of a given database table."""

        try:
            inspector = inspect(self.engine)

            if not inspector.has_table(table_name):
                raise ValueError(
                    f"Table '{table_name}' does not exist."
                )

            columns = inspector.get_columns(table_name)

            return [column["name"] for column in columns]

        except SQLAlchemyError as error:
            raise RuntimeError(
                f"Unable to inspect table '{table_name}': {error}"
            ) from error

    def close(self):
        """Close the SQLAlchemy database engine to save resources."""

        self.engine.dispose()


if __name__ == "__main__":

    database = DatabaseManager()

    try:
        database.create_database()

        print("\nExisting Database Tables:")

        tables = database.get_table_names()

        if tables:
            for table in tables:
                print(table)
                print(
                    "Columns:",
                    database.get_table_columns(table)
                )
        else:
            print("No tables have been stored yet.")

    finally:
        database.close()
