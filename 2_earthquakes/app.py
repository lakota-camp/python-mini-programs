import requests
import csv 
import xmltodict
from dotenv import load_dotenv
import os
import json
import datetime

"""
This program fetches hourly earthquake data from the USGS Earthquake API and processes it to include human-readable
time and location information. The program performs the following steps:
1. Fetches earthquake data from the USGS Earthquake API.
2. Converts the coordinates of each earthquake to a human-readable location using the OpenCage Geocoding API.
3. Converts the Unix timestamp of each earthquake to a human-readable date and time.
4. Compiles the processed data into a list of dictionaries.
5. Exports the compiled data to a CSV file.
6. Reads data from CSV file to ensure data is correctly formatted.
The program is designed to be run as a standalone script and requires an API key for the OpenCage Geocoding API,
which should be stored in a .env file.
"""

# Load API Key
load_dotenv()
open_cage_api_key = os.getenv("OPEN_CAGE_API_KEY")

def get_earthquake_data() -> list:
    """Makes GET request to the earthquake.gov API.
    Returns:
        list: [mag, coordinate, time]
    """
    try: 
        url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson'
        response = requests.get(url)
        
        if response:
            print("\nSuccessful connection!")
            print(f'Status code:  {response.status_code}\n')
            earthquake_data = []
            data = json.loads(response.text)
    
            data = data["features"]
            
            for item in data:
                mag = item["properties"]["mag"]
                coordinates = item["geometry"]["coordinates"]
                time = item["properties"]["time"]
                
                entry = {
                    'mag': mag,
                    'coordinates': coordinates,
                    'time': time
                }
                
                earthquake_data.append(entry)
            
            return earthquake_data
        
            
        else:
            print("Connection error. Request failed")
            print(f'Status code: {response.status_code}')
            print("Double check the URL and try again.")
            
    except Exception as e:
        print(f'An unexpected error has ocurred: {e}')

def convert_coordinates_to_location(long: float, lat: float) -> list:
    """Converts the give coordinates (longitude and latitude)
    to a human-readable location.
    Args:
        long (float): longitude
        lat (float): latitude

    Returns:
        list: ['County', 'State']
    """
    try:
        url = f'https://api.opencagedata.com/geocode/v1/xml?q={lat}+{long}&key={open_cage_api_key}'
        response = requests.get(url)
        
        location = []
        
        if response:
            data = xmltodict.parse(response.text)

            county = data["response"]["results"]["result"]["components"]["county"]
            state = data["response"]["results"]["result"]["components"]["state"]
                
            location.append(county)
            location.append(state)
            
            return location
            
        else:
            print("Connection error. Request failed")
            print(f'Status code: {response.status_code}')
            return
    except Exception as e:
        # If error occurs when finding location, N/A assigned to both 'county' and 'state'
        location.append("N/A")
        location.append("N/A")
        return location

def convert_time(unix_time: float) -> str:
    """Converts Unix time to a human readable format. 
    E.g. “September 01, 2022 at 12:00:00 AM”

    Args:
        unix_time (float): Unix time

    Returns:
        str: Human readable time string: “September 01, 2022 at 12:00:00 AM”
    """
    # Convert milliseconds to seconds
    time_seconds = unix_time / 1000
    # Convert Unix epoch time to datetime object
    datetime_stamp = datetime.datetime.fromtimestamp(time_seconds, datetime.timezone.utc)
    # Subtract 7 hours to adjust for time zone difference
    datetime_adj_timestamp = datetime_stamp - datetime.timedelta(hours = 7)
    # Convert to human-interpretable string
    # “September 01, 2022 at 12:00:00 AM”
    return datetime_adj_timestamp.strftime("%B %d, %Y at %I:%M:%S %p")

def export_to_csv(data: list) -> csv:
    """
    Exports the provided earthquake data to a CSV file named 'earthquakes.csv'.

    Args:
        data (list): A list of dictionaries containing earthquake data. Each dictionary should have the following keys:
            - 'time': The time of the earthquake in a human-readable format.
            - 'magnitude': The magnitude of the earthquake.
            - 'latitude': The latitude of the earthquake's location.
            - 'longitude': The longitude of the earthquake's location.
            - 'location': A list containing the county and state of the earthquake's location.

    Returns:
        None
    """
    print('Writing data to to csv file...')
    try:
        with open('./earthquakes.csv', 'w') as file:
            writer = csv.writer(file, lineterminator='\n')
            # Row one of CSV - Column names
            row_1 = ['time', 'magnitude', 'latitude', 'longitude', 'county', 'state']
            writer.writerow(row_1)
            
            # Writing each line of CSV
            for line_number, line in enumerate(data, start=1):
                data = [line["time"], line["magnitude"], line["latitude"], line["longitude"], line["location"][0], line["location"][1]]
                writer.writerow(data)
                # Logging when each line is complete
                print(f'Line {line_number} complete')
        print('Write to csv file complete!')
        
    except Exception as e:
        print(f'An unknown error occurred when trying to export to CSV: {e}')
        print('Please try again.')

# Function to read CSV file to ensure proper format
def read_from_csv(csv_file) -> None:
    """
    Reads data from a specified CSV file and prints each line.

    Args:
        csv_file (str): The path to the CSV file to be read.

    Returns:
        None
    """
    print(f'\nReading from file: {csv_file[2:]}')
    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
                        
            for line in reader:
                print(line)

    except Exception as e:
        print(f'An unknown error occurred when trying to export to CSV: {e}')
        print('Please try again.')

def main():
    """ 
    Main entry point for the program.
    
    """
    # Fetch earthquake data
    earthquake_data = get_earthquake_data()
    
    # List to store final data set
    earthquakes_and_location_data = []

    # Loop through each earthquake entry, convert and format the data, and store it in earthquakes_and_location_data
    for line in earthquake_data:
        mag = line["mag"]
        long = line["coordinates"][0]
        lat = line["coordinates"][1]
        time = convert_time(line["time"])
        location_data = convert_coordinates_to_location(long, lat)
        
        # If location data is 'N/A' display location as 'Ocean'
        if location_data == ["N/A", "N/A"]:
            print(f"Magnitude {mag} earthquake on {time} and located at ({lat}, {long}) in the Ocean.\n")
            description = f"Magnitude {mag} earthquake on {time} and located at ({lat}, {long}) in the Ocean."
        else:
            print(f"Magnitude {mag} earthquake on {time} and located at ({lat}, {long}) in {location_data[0]}, {location_data[1]}.\n")
            description = f"Magnitude {mag} earthquake on {time} and located at ({lat}, {long}) in {location_data[0]}, {location_data[1]}."
        
        # Entry to add to final data list
        entry = {
            "magnitude": mag,
            "longitude": long,
            "latitude": lat, 
            "time": time,
            "location": location_data,
            "description": description
        }

        earthquakes_and_location_data.append(entry)
        
    # Export to CSV
    export_to_csv(earthquakes_and_location_data)
    
    # Read from CSV
    csv_file = './earthquakes.csv'
    read_from_csv(csv_file)
    
    
if __name__ == "__main__":
    main()