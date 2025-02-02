import pandas as pd
import glob
import requests
import time
import os
from shapely import wkt
import google.generativeai as genai
from datetime import timedelta,datetime


def get_addresses_geolife():
    """
    Get the addresses from the Geolife dataset.
    """
    #read the dataset
    df = pd.read_csv('data\\dataSet_geolife.csv')
    #locations unique values
    locations = df['location_id'].unique()
    locs = pd.read_csv('data\\locations_geolife.csv')
    my_api_key = '679b97bfdc523860581209akv3fef67'

    locs_addresses_dict = dict()
    for loc in locations:
        #get the latitude and longitude from the id center of locs but is a POINT object
        point = wkt.loads(locs[locs['id'] == loc]['center'].values[0])
        lat=point.y
        lon=point.x
        address=get_address_from_coords(lat, lon, my_api_key)
        locs_addresses_dict[loc] = address
    
    locs_addresses_dict_df = pd.DataFrame(locs_addresses_dict.items(), columns=['location_id', 'address'])
    #save the dataframe to a csv file
    locs_addresses_dict_df.to_csv('data\\locs_addresses_geolife.csv', index=False)



def get_address_from_coords(latitude, longitude, api_key):
    """
    Get the address for given coordinates using the geocode.maps.co API.
    
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        api_key (str): Your geocode.maps.co API key.
    
    Returns:
        str: Formatted address or an error message.
    """
    # API endpoint for reverse geocoding
    url = "https://geocode.maps.co/reverse"
    
    # Parameters (lat, lon, API key)
    params = {
        "lat": latitude,
        "lon": longitude,
        "api_key": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise HTTP errors
        time.sleep(1)  # Sleep for 1 second to avoid hitting the rate limit
        data = response.json()
        
        if not data:
            return "No address found for the given coordinates."
        
        # Extract the formatted address
        address = data.get("display_name", "Address not available")
        return address
    
    except requests.exceptions.RequestException as e:
        return f"API request failed: {e}"

def generate_sentiment_output(index,df_new,sep,path_dataset_input_llm,path_dataset_output_llm,model,context_prompt,cond,sentiments):
    if index<=12000:
        return pd.DataFrame(columns=['index','id','start_day', 'end_day','start_time', 'end_time', 'location'])
    if(cond(index,sep)):
        df_new.to_csv(f'{path_dataset_input_llm}\\dataSet_input_llm_geolife_{index}.csv', index=False)
        df_new = pd.DataFrame(columns=['index','id','start_day', 'end_day','start_time', 'end_time', 'location'])
        uploaded_files = genai.upload_file(f'{path_dataset_input_llm}\\dataSet_input_llm_geolife_{index}.csv')
        
        flag=True
        context = [context_prompt,uploaded_files]
        while(flag):
            try:
                response = model.generate_content(context)#,generation_config=generation_config)

                with open(f"""{path_dataset_output_llm}\\dataSet_output_llm_geolife_{index}.csv""", "w", encoding='utf-8', errors='ignore') as f:
                    f.write(response.text)
                
                with open(f"""{path_dataset_output_llm}\\dataSet_output_llm_geolife_{index}.csv""", "r", encoding="utf-8", errors='ignore') as file:
                    lines = file.readlines()

                modified_lines = [line.replace("```csv", "").replace("```", "") for line in lines]

                with open(f"""{path_dataset_output_llm}\\dataSet_output_llm_geolife_{index}.csv""", "w", encoding="utf-8",errors='ignore') as file:
                    file.writelines(modified_lines)

                df1=pd.read_csv(f"""{path_dataset_input_llm}\\dataSet_input_llm_geolife_{index}.csv""")
                df2=pd.read_csv(f"""{path_dataset_output_llm}\\dataSet_output_llm_geolife_{index}.csv""")

                sentiments_set=set(sentiments.split(', '))

                cond_a= list(df1['index'])!=list(df2['index'])
                cond_b= list(df2.columns)!=['index','sentiment','explanation']
                cond_c = not df2['sentiment'].isin(sentiments_set).all() 

                if cond_a:
                    print("The indexes of the output dataset are not the same as the input ones.")
                    continue
                
                if cond_b:
                    print("The columns of the output dataset are not the same as the expected ones.")
                    continue

                if cond_c:
                    print("The sentiments are not the same as the expected ones.")
                    continue

                df2['id']=df1['id'].values

                flag=False
                
            except Exception as e:
                print(e)
        
    return df_new

def get_sentiments_geolife():
    # Load the data from the CSV file into a pandas DataFrame
    df_loc_addresses = pd.read_csv('data\\locs_addresses_geolife.csv')
    df = pd.read_csv('data\\dataSet_geolife.csv')

    genai.configure(api_key="AIzaSyDwkh89PbxBWrs2pq5Zad569-dYVDi0P0o")
    model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp")

    sentiments="fear, hunger, illness, indifference, tiredness"
    default_sentiment= 'indifference'
    weekdays = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    context_prompt= f"""You are a sentiment analysis expert. 
            You will be given a dataset and you will have to create a 
            situational context for each row of this dataset, with the provided information, that it is just the time and location; 
            from this context your main goal is to identify a sentiment from the following list of sentiments: {sentiments}. 
            You will have to return the sentiment that is most prominent in the situational context. 
            If you are unable to identify the sentiment, 
            you will have to return '{default_sentiment}'. You will have to return only the sentiment and a brief explanation of why you chose that sentiment. 
            The Output should be in a structured CSV with columns: index, sentiment, explanation. Provide only CSV-formatted output. 
            The index of the output must be the same of the input. The sentiment must be in lowercase. 
            The explanation should be between double quotes and can't have chinese characters.
            You must provide output for all rows in the input. 
            
            Example:
                Input: 17,968,Friday,Friday,10:35 AM,12:32 PM,"KFC, Chengfu Road, Wudaokou, Dongsheng, Haidian District, Beijing, 100190, China"
                Output: 17,hunger,"Being at KFC during late morning/noon suggests hunger for lunch."

                
            """

    #if folder does not exist, create it
    path_dataset_input_llm= 'data\\dataset_input_llm'
    if not os.path.exists(path_dataset_input_llm):
        os.makedirs(path_dataset_input_llm)

    path_dataset_output_llm= 'data\\dataset_output_llm'
    if not os.path.exists(path_dataset_output_llm):
        os.makedirs(path_dataset_output_llm)
    
    #generation_config = {"temperature": 0.3}
    df_new = pd.DataFrame(columns=['index','id','start_day', 'end_day','start_time', 'end_time', 'location'])
    
    sep=1000 #size of the input for llm 

    # Iterate over each row in the DataFrame and extract the required information
    for index, row in df.iterrows():
        base_start_time = datetime(1900, 1, 1) + timedelta(minutes=row['start_min'])
        formatted_start_time = base_start_time.strftime("%I:%M %p")
        base_end_time = datetime(1900, 1, 1) + timedelta(minutes=row['end_min'])
        formatted_end_time = base_end_time.strftime("%I:%M %p")
        start_day = weekdays[(row['weekday'])%7]
        end_day = weekdays[(row['weekday']+(row['end_day']-row['start_day']))%7]
        
        df_new.at[index, 'index'] = index
        df_new.at[index, 'id'] = int(row['id'])
        df_new.at[index, 'start_time'] = formatted_start_time
        df_new.at[index, 'end_time'] = formatted_end_time
        df_new.at[index, 'start_day'] = start_day
        df_new.at[index, 'end_day'] = end_day
        df_new.at[index, 'location'] = df_loc_addresses[row['location_id']==df_loc_addresses["location_id"]].values[0][1]
        
        
        df_new=generate_sentiment_output(index,df_new,sep,path_dataset_input_llm,path_dataset_output_llm,model,context_prompt,lambda index,sep: index!=0 and index%sep==0,sentiments)

    df_new=generate_sentiment_output(df.shape[0]-1,df_new,sep,path_dataset_input_llm,path_dataset_output_llm,model,context_prompt,lambda index,sep: index%sep!=0,sentiments)


    #merge all output dataset
    # Get all CSV file paths
    csv_files = glob.glob(f"{path_dataset_output_llm}\\*.csv")

    df_combined = pd.concat([pd.read_csv(f) for f in csv_files if f!=f'{path_dataset_output_llm}\\dataSet_output_llm_geolife.csv'],ignore_index=True)
    df_combined = df_combined.sort_values(by="index")
    df_combined.to_csv(f'{path_dataset_output_llm}\\dataSet_output_llm_geolife.csv', index=False)

    #df_final = df.copy()
    #df_final = df_final.merge(df_combined, on='location_id', how='left')
    #df_final.to_csv(f'data\\dataSet_sentiment_geolife.csv', index=False)

        
#get_addresses_geolife()
get_sentiments_geolife()