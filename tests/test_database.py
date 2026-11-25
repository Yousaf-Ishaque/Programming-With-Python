
"""
test_database.py

Unit tests for validating database initialization,
engine creation, and DataFrame storage functionality.
"""

import unittest
import pandas as pd

from database import DatabaseManager
from sqlalchemy.engine import Engine


class TestDatabaseManager(unittest.TestCase):
    """Tests the functionality of database_manager."""

    def setUp(self):
        """Initialization of database manager prior to each test."""
        self.database = DatabaseManager()

    def tearDown(self):
        """Enables the disposal of database engine after every test."""
        self.database.close()

    def test_database_manager_created(self):
        """Validation that the database manager is initialized."""
        self.assertIsNotNone(self.database)

    def test_database_path(self):
        """Validates that the configured SQLite database path is valid."""
        self.assertTrue(
            self.database.database_path.endswith("output/assignment.db")
        )

    def test_engine_created(self):
        """Validates that the SQLAlchemy engine is generated."""
        self.assertIsNotNone(self.database.get_engine())

    def test_get_engine_returns_engine(self):
        """Validates that get_engine returns the configured engine."""
        self.assertIsInstance(self.database.get_engine(), Engine)

    def test_save_dataframe_none(self):
        """Validates that saving None raises a ValueError."""
        with self.assertRaises(ValueError):
            self.database.save_dataframe(None, "training_data")

    def test_save_dataframe(self):
        """Validates that a DataFrame is saved as a database table."""
        dataframe = pd.DataFrame({
            "x": [1, 2],
            "y": [10, 20]
        })

        self.database.save_dataframe(dataframe, "sample_table")

        table_names = self.database.get_table_names()

        self.assertIn("sample_table", table_names)

    def test_create_database(self):
        """Validates that the database connection can be generated."""
        self.database.create_database()

        with self.database.get_engine().connect() as connection:
            result = connection.exec_driver_sql("SELECT 1").scalar()

        self.assertEqual(result, 1)

    def test_engine_type(self):
        """Validates that the SQLAlchemy engine contains the expected type."""
        self.assertIsInstance(self.database.get_engine(), Engine)


if __name__ == "__main__":
    unittest.main()