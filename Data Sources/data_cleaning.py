import pandas as pd
import numpy as np
import os
from datetime import datetime
import shutil
import argparse
import sys
import traceback


class FileNotFoundError(Exception):
    """Custom exception for file not found"""

    pass


def validate_file(filename):
    """
    Validate if the file exists and is a CSV file
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: The file '{filename}' does not exist.")

    if not filename.lower().endswith(".csv"):
        raise ValueError(f"Error: '{filename}' is not a CSV file.")


def fill_missing_values(df):
    """
    Fill missing values using appropriate methods for each column
    """
    # For Volume: fill with rolling average of past 5 days
    df["Volume"] = df["Volume"].fillna(
        df["Volume"].rolling(window=5, min_periods=1).mean()
    )

    # For price columns (Close, Open, High, Low)
    price_columns = ["Close", "Open", "High", "Low"]
    for col in price_columns:
        if df[col].isnull().any():
            # Fill with previous day's value first
            df[col] = df[col].fillna(method="ffill")
            # If any remaining NaN (at the beginning), fill with next day's value
            df[col] = df[col].fillna(method="bfill")

    return df


def clean_data(filename):
    """
    Clean the data from the specified CSV file
    """
    try:
        # Validate file first
        validate_file(filename)

        # Create archive directory if it doesn't exist
        archive_dir = "archive"
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)

        # Archive the original file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        archive_file = os.path.join(
            archive_dir, f"{os.path.splitext(filename)[0]}_{timestamp}.csv"
        )
        shutil.copy2(filename, archive_file)

        # Read and clean the data
        try:
            data = pd.read_csv(filename)
        except pd.errors.EmptyDataError:
            raise ValueError(f"Error: '{filename}' is empty.")
        except pd.errors.ParserError:
            raise ValueError(f"Error: '{filename}' is not properly formatted CSV.")

        df = data.copy()

        # Print initial data types for debugging
        print("\nInitial column dtypes:")
        print(df.dtypes)
        print("\nSample of data before cleaning:")
        print(df.head())

        # Convert Date to datetime with correct format (MM/DD/YY)
        try:
            df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
        except KeyError:
            raise ValueError("Error: CSV file must contain a 'Date' column.")
        except ValueError:
            raise ValueError(
                "Error: Dates in CSV are not in the expected format (MM/DD/YY)."
            )

        # Sort values
        df = df.sort_values("Date", ascending=True)

        # Verify required columns exist
        required_columns = ["Price", "Open", "High", "Low", "Vol."]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Error: Missing required columns: {', '.join(missing_columns)}"
            )

        # Convert price columns to numeric (removing commas)
        price_columns = ["Price", "Open", "High", "Low"]
        for col in price_columns:
            # Check if column contains string values before using str accessor
            if df[col].dtype == "object":
                df[col] = df[col].str.replace(",", "").astype(float)
            else:
                # If already numeric, just ensure it's float type
                df[col] = df[col].astype(float)

        # Rename columns
        df = df.rename(columns={"Price": "Close", "Vol.": "Volume"})

        # Clean Volume column - handle both K and M
        def convert_volume(vol_str):
            if isinstance(vol_str, str):
                if "K" in vol_str:
                    return float(vol_str.replace("K", "")) * 1000
                elif "M" in vol_str:
                    return float(vol_str.replace("M", "")) * 1000000
                else:
                    return float(vol_str)
            return vol_str

        df["Volume"] = df["Volume"].apply(convert_volume)

        # Drop Change % column if it exists
        if "Change %" in df.columns:
            df = df.drop("Change %", axis=1)

        # Calculate daily returns
        df["Daily_Return"] = df["Close"].pct_change()

        # Set Date as index
        df = df.set_index("Date")

        # Check and display missing values before filling
        print("\nMissing values before filling:")
        print(df.isnull().sum())

        # Fill missing values
        df = fill_missing_values(df)

        # Check and display missing values after filling
        print("\nMissing values after filling:")
        print(df.isnull().sum())

        # Display descriptive statistics of the cleaned data
        print("\nDescriptive statistics of the cleaned data:")
        print(df.describe())

        # Save the cleaned data to the original location
        df.to_csv(filename)

        print(f"\nOriginal file archived as: {archive_file}")
        print(f"Cleaned data saved to: {filename}")

        # Display info about the cleaned dataset
        print("\nCleaned dataset info:")
        print(df.info())

        print("\nFirst few rows of cleaned dataset:")
        print(df.head())

        # Show examples of filled values
        print("\nRows where Volume was originally missing and filled:")
        print("Format: Date | Filled Volume | Calculated From")
        print("-" * 60)

        # Get original missing volume rows
        missing_volume_mask = data["Vol."].isna()
        if missing_volume_mask.any():
            missing_dates = data[missing_volume_mask]["Date"].tolist()
            for date in missing_dates:
                dt = pd.to_datetime(date, format="%m/%d/%y")
                filled_value = df.loc[dt, "Volume"]
                # Get the 5 previous days' values used for the average
                prev_5_days = df.loc[:dt].iloc[-6:-1]["Volume"].values
                print(
                    f"{dt.strftime('%Y-%m-%d')}: {filled_value:10.2f} | Avg of previous values: {', '.join([f'{v:.2f}' for v in prev_5_days if not np.isnan(v)])}"
                )
        else:
            print("No missing Volume values found in the dataset.")

    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print("An unexpected error occurred!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Clean financial data from CSV file.")
    parser.add_argument(
        "--filename", type=str, required=True, help="Name of the CSV file to clean"
    )

    # Parse arguments
    args = parser.parse_args()

    # Clean the data
    clean_data(args.filename)


if __name__ == "__main__":
    main()
