import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch database credentials from environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

# Create the MySQL connection URL using PyMySQL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

def get_db_engine():
    """
    Creates and returns a SQLAlchemy engine instance.
    """
    try:
        # Create database engine
        engine = create_engine(DATABASE_URL)
        
        # Test the connection
        with engine.connect() as connection:
            print("Success: Connected to the MySQL database successfully!")
            
        return engine
        
    except SQLAlchemyError as e:
        print(f"Error: Failed to connect to the database. \nDetails: {e}")
        return None

# Execute connection test when running this file directly
if __name__ == "__main__":
    get_db_engine()