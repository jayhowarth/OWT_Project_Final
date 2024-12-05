import pandas as pd
import numpy as np
import os
import glob

data_dir = '/Users/jayhowarth/Exeter Uni/MSc Project/Data/' 

def get_file_list(data_directory):
    """
    Retrieves a sorted list of all .txt files in the specified directory.

    Args:
        data_dir (str): Path to the directory containing data files.

    Returns:
        list: Sorted list of file paths.
    """
    file_list = glob.glob(os.path.join(data_directory, '*.txt'))
    if not file_list:
        raise ValueError(f"No txt files found in directory {data_directory}")
    file_list.sort()  # Ensure consistent order
    print(f"Found {len(file_list)} files.")
    return file_list


def convert_to_csv_by_day():
    output_folder = 'csvs'
    os.makedirs(output_folder, exist_ok=True)
    file_list = get_file_list(data_dir)
    for file in file_list:
        df = pd.read_csv(file)
        df['Timestamp'] = pd.to_datetime(df['Date'])

        df['Day'] = df['Timestamp'].dt.date
        df['Timex'] = df['Timestamp'].dt.time

        # Split the DataFrame by day and save each as a separate CSV file
        for date, day_df in df.groupby('Day'):
            day_df = day_df.sort_values('Date')
            time = str(day_df['Timex'].iloc[0])[:-3].replace(":", "-")
            output_path = os.path.join(output_folder, f"{date}_{time}.csv")
            day_df.drop(columns=['Day'], inplace=True)  # Drop the extra Date column
            day_df.drop(columns=['Timex'], inplace=True)
            day_df.to_csv(output_path, index=False)
            print(f"Saved {output_path}")


def change_in_place_timestamp():
    output_folder = 'data_ts'
    os.makedirs(output_folder, exist_ok=True)
    file_list = get_file_list(data_dir)
    for idx, file in enumerate(file_list):
        print(file)
        df = pd.read_csv(file)
        # Parse 'Date' and 'Time(ms)' to create 'Timestamp'
        df['DateTime'] = pd.to_datetime(df['Date'], format='%d %B %Y %H:%M', errors='coerce')
        df['Timestamp'] = pd.to_timedelta(df['Time(ms)'], unit='ms')
        df['Timestamp'] = df['DateTime'] + df['Timestamp']

        # Drop intermediate columns
        df.drop(columns=['DateTime', 'Date', 'Time', 'Time(ms)'], inplace=True)
        df.to_csv(f'./{output_folder}/all_data_0{idx}.txt', header=True, index=False)

def downsample(downsample_rate):
    output_folder = f'data_downsample_{downsample_rate}'
    os.makedirs(output_folder, exist_ok=True)
    file_list = get_file_list(data_dir)
    for idx, file in enumerate(file_list):
        print(file)
        df = pd.read_csv(file)
        # Parse 'Date' and 'Time(ms)' to create 'Timestamp'
        df['DateTime'] = pd.to_datetime(df['Date'], format='%d %B %Y %H:%M', errors='coerce')
        df['Timestamp'] = pd.to_timedelta(df['Time(ms)'], unit='ms')
        df['Timestamp'] = df['DateTime'] + df['Timestamp']

        # Drop intermediate columns
        df.drop(columns=['DateTime', 'Date', 'Time', 'Time(ms)'], inplace=True)

        df = df.set_index('Timestamp')
        # Resample the data to the specified interval
        df = df.resample(f'{downsample_rate}ms').mean()
        # Drop any rows with NaN values resulting from resampling
        df = df.dropna()

        # Reset index
        df = df.reset_index()
        
        df.to_csv(f'./{output_folder}/resample_{downsample_rate}ms_0{idx+1}.txt', header=True, index=False)

downsample(50)