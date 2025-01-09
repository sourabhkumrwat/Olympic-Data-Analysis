from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def fetch_medal_tally(df , year , country):
    medal_df = df.drop_duplicates(subset=['Team','region','Games','Year','City','Sport','Event','Medal']) 
    flag = 0
    if year == 'Overall' and country == 'Overall' :
        temp_df = medal_df
    if year == 'Overall' and country != 'Overall' :
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    if year != 'Overall' and country == 'Overall' :
        temp_df = medal_df[medal_df['Year'] == int(year)]
    if year != 'Overall' and country != 'Overall' :
        temp_df = medal_df[(medal_df['Year'] == year) & (medal_df['region'] == country)]
     
    if flag ==1 :   
         x = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()  
    else:
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()  
           
    x ['total'] = x['Gold'] + x['Silver'] + x['Bronze'] 
    
    
    x['Gold'] = x['Gold'].astype('int')
    x['Silver'] = x['Silver'].astype('int')
    x['Bronze'] = x['Bronze'].astype('int')
    x['total'] = x['total'].astype('int')
    
    return x    


def medal_tally(df):
    
    medal_tally = df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal']) 
    
    medal_tally = medal_tally.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()
    
    medal_tally ['total'] = medal_tally['Gold'] + medal_tally['Silver'] + medal_tally['Bronze'] 
    
    medal_tally['Gold'] = medal_tally['Gold'].astype('int')
    medal_tally['Silver'] = medal_tally['Silver'].astype('int')
    medal_tally['Bronze'] = medal_tally['Bronze'].astype('int')
    medal_tally['total'] = medal_tally['total'].astype('int')
    
    return medal_tally

def country_Year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0,'Overall')
    
    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0,'Overall')
    
    return years , country
    
    
def data_over_time(df, col):
    
    
 nations_over_time = df.drop_duplicates(['Year', col])['Year'].value_counts().reset_index().sort_values('Year')
 nations_over_time.rename(columns={'count': col },inplace=True)
 return nations_over_time


def most_successful(df,sport):
    temp_df = df.dropna(subset=['Medal'])
    
    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]
        
    x = temp_df['Name'].value_counts().reset_index().head(15).merge(df,left_on='Name',right_on='count',how='left')
    x.rename(columns={'count':'Medals'}, inplace=True)
    return x

def most_successful(df, sport):
    temp_df = df.dropna(subset=['Medal'])

    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]
    
    # Calculate value counts and reset index, renaming columns properly
    name_counts = temp_df['Name'].value_counts().reset_index()
    name_counts.columns = ['Name', 'Medals']  # Rename columns for clarity

    # Merge with the original DataFrame
    result = name_counts.merge(df, left_on='Name', right_on='Name', how='left')
    
    # Select required columns and drop duplicates
    result = result[['Name', 'Medals', 'Sport', 'region']].drop_duplicates('Name').head(15)

    return result


def yearwise_medal_tally(df,country):
  temp_df = df.dropna(subset=['Medal'])
  temp_df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'],inplace=True)


  new_df = temp_df[temp_df['region'] == country]
  final_df = new_df.groupby('Year').count()['Medal'].reset_index()
  
  return final_df


def country_event_heatmap(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'],inplace=True)
    
    new_df = temp_df[temp_df['region'] == country]
    
    pt = new_df.pivot_table(index='Sport',columns='Year',values='Medal',aggfunc='count').fillna(0)
    
    return pt      


def most_successful_countrywise(df,country):
    temp_df = df.dropna(subset=['Medal'])
    
    
    temp_df = temp_df[temp_df['region'] == country]
        
    x = temp_df['Name'].value_counts().reset_index().head(15).merge(df,left_on='Name',right_on='count',how='left')
    x.rename(columns={'count':'Medals'}, inplace=True)
    return x

def most_successful_countrywise(df, country):
    temp_df = df.dropna(subset=['Medal'])

    
    temp_df = temp_df[temp_df['region'] == country]
    
    # Calculate value counts and reset index, renaming columns properly
    name_counts = temp_df['Name'].value_counts().reset_index()
    name_counts.columns = ['Name', 'Medals']  # Rename columns for clarity

    # Merge with the original DataFrame
    result = name_counts.merge(df, left_on='Name', right_on='Name', how='left')
    
    # Select required columns and drop duplicates
    result = result[['Name', 'Medals', 'Sport']].drop_duplicates('Name').head(15)

    return result

def weight_v_height(df,sport):
    athlete_df = df.drop_duplicates(subset=['Name','region'])
    athlete_df['Medal'].fillna('No Medal',inplace=True)
    if sport != 'Overall' :
       temp_df = athlete_df[athlete_df['Sport'] == sport]
       return temp_df
    else:
        return athlete_df
    
def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name','region'])
    
    men = athlete_df [athlete_df ['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df [athlete_df['Sex'] == 'F'].groupby('Year').count() ['Name'].reset_index()
    
    final= men.merge(women, on="Year",how = 'left')
    final.rename(columns={'Name_x': 'Male', 'Name_y':'Female'}, inplace=True)
    
    final.fillna(0,inplace=True)
    
    return final





# for predictions of 2020 and 2024



def predict_medals(df, years_to_predict=[2020, 2024]):

    # Group data to get total medals by country and year
    medal_df = df.groupby(['Year', 'region']).sum()[['Gold', 'Silver', 'Bronze']].reset_index()
    medal_df['Total'] = medal_df['Gold'] + medal_df['Silver'] + medal_df['Bronze']

    # Initialize prediction dictionary
    predictions = {year: [] for year in years_to_predict}

    # Unique countries
    countries = medal_df['region'].unique()

    for country in countries:
        # Get medal trends for the country
        country_data = medal_df[medal_df['region'] == country]
        X = country_data['Year'].values.reshape(-1, 1)
        y = country_data['Total'].values

        # Ensure there's enough data for prediction
        if len(X) > 1:
            model = LinearRegression()
            model.fit(X, y)

            # Predict medals for each year
            for year in years_to_predict:
                pred = model.predict([[year]])[0]
                predictions[year].append({
                    'Country': country,
                    'Predicted Medals': max(0, int(pred))  # Non-negative prediction
                })

    # Convert predictions to DataFrame for easier visualization
    prediction_dfs = {
        year: pd.DataFrame(predictions[year]).sort_values(by='Predicted Medals', ascending=False).head(10)
        for year in years_to_predict
    }

    return prediction_dfs



                           