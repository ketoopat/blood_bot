import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from aiogram import Bot, types
from datetime import datetime, timedelta
import asyncio
import os

bot_token = '6583308519:AAHM7QRqg34taX2tDgaOK5ihWM8qKn8QmYk'
chat_id = '-1002004078431'


# Fetch .CSVs from url
url_donations_facility = 'https://github.com/MoH-Malaysia/data-darah-public/raw/main/donations_facility.csv'

url_donations_state = 'https://github.com/MoH-Malaysia/data-darah-public/raw/main/donations_state.csv'

url_newdonors_facility = 'https://github.com/MoH-Malaysia/data-darah-public/raw/main/newdonors_facility.csv'

url_newdonors_state = 'https://github.com/MoH-Malaysia/data-darah-public/raw/main/newdonors_state.csv'

url_blood_donation = 'https://dub.sh/ds-data-granular'


# convert to dataframe
df_donations_facility = pd.read_csv(url_donations_facility)

df_donations_state = pd.read_csv(url_donations_state)

df_newdonors_facility = pd.read_csv(url_newdonors_facility)

df_newdonors_state = pd.read_csv(url_newdonors_state)

df_blood_donation = pd.read_parquet(url_blood_donation, engine='fastparquet') # parquet file

# Function to save chart to a file
def save_chart(plt, filename, chart_file_paths):
    file_path = f'{filename}.png'
    plt.savefig(file_path)
    plt.close()
    chart_file_paths.append(file_path)

# Function to process data and generate charts
def process_data_and_generate_charts():
    chart_file_paths = []

    ## [Question 1.1] Script for 1st Chart (Malaysia Blood Donation Stats. as of Yesterday (updated EoD))

    df_donations_state['date'] = pd.to_datetime(df_donations_state['date'])

    latest_day = df_donations_state['date'].max()
    df_daily = df_donations_state[df_donations_state['date'] == latest_day]

    df_bloodrate = df_daily[['date','state','daily','blood_a','blood_b','blood_o','blood_ab']]

    blood_type_totals  = df_bloodrate.groupby('state')[['blood_a', 'blood_b', 'blood_o', 'blood_ab']].sum()

    blood_type_totals['total'] = blood_type_totals.sum(axis=1)

    for blood_type in ['blood_a', 'blood_b', 'blood_o', 'blood_ab']:
        blood_type_totals[f'perc_{blood_type}'] = (blood_type_totals[blood_type] / blood_type_totals['total']) * 100

    blood_type_totals = blood_type_totals.reset_index()

    # Original DataFrame

    country_df = blood_type_totals[blood_type_totals['state'] == 'Malaysia'].melt()

    country_df = country_df[~country_df['variable'].isin(['total', 'state'])].reset_index(drop=True)



    # Splitting the DataFrame into counts and percentages
    df_counts = country_df.iloc[:4, :].rename(columns={'value': 'count'}).reset_index(drop=True)
    df_percents = country_df.iloc[4:, :].rename(columns={'value': 'percentage'}).reset_index(drop=True)

    # Cleaning the 'variable' columns to match
    df_counts['variable'] = df_counts['variable'].str.replace('blood_', '')
    df_percents['variable'] = df_percents['variable'].str.replace('perc_blood_', '')

    # Merging the two DataFrames on the 'variable' column
    df_merged = pd.merge(df_percents, df_counts, on='variable')

    # Reordering the columns to match the desired output
    df_final = df_merged[['variable', 'percentage', 'count']]

    df_final['variable'] = df_final['variable'].str.upper()

    # Convert the 'count' column to string type and format with comma separators
    df_final['count'] = df_final['count'].apply(lambda x: f'{x:,}')

    # store latest date, to be used in chart title
    latest_date = latest_day.strftime('%d %B %Y')

    # store total donors
    # total_donors = (df_final['count'].astype('int64')).sum()
    total_donors  = (df_final['count'].str.replace(',', '').astype(int)).sum()


    # Sorting the DataFrame by the percentage in descending order for better visualization
    df_final.sort_values('percentage', ascending=False, inplace=True)

    # Create a horizontal bar plot using seaborn

    current_palette = sns.color_palette("Reds", n_colors=4)
    reversed_palette = current_palette[::-1]

    # name 1st Chart as fig1
    fig1, ax = plt.subplots(figsize=(10, 5)) 
    bar_plot = sns.barplot(
        x='percentage', 
        y='variable', 
        data=df_final, 
        palette=reversed_palette
    )

    # Add the percentage and count labels in the specified format at the end of the bars
    for index, p in enumerate(bar_plot.patches):
        width = p.get_width()  # get bar length
        percentage = df_final.iloc[index]['percentage']  # get the percentage for the blood type
        count = df_final.iloc[index]['count']  # get the donor count for the blood type
        label_text = f'{percentage:.1f}% | {count} Donors'  # formatted text label
        plt.text(width + 0.5,  # x-coordinate position of text, outside the bar
                p.get_y() + p.get_height() / 2,  # y-coordinate position of text, to be at the middle height of the bar
                label_text,  # the text label
                va='center', 
                ha='left', 
                color='black')

    # Remove the spines and the axes labels
    sns.despine(left=True, bottom=True)
    plt.ylabel('')  # Remove the y-axis label
    plt.xlabel('')  # Remove the x-axis label
    plt.xticks([])  # Remove the x-axis tick labels
    plt.yticks(range(len(df_final['variable'])), df_final['variable'])  # Show the y-axis tick labels which are the blood types
    plt.gca().tick_params(axis='y', length=0)  # Remove the y-axis ticks

    # Add title
    title = 'Malaysia Blood Donation Figures for Yesterday, {}'.format(latest_date)
    plt.title(title)

    # Place the 'Total Donors' text at the bottom middle of the chart
    plt.text(0.5, 0.001,  # x, y coordinates (in axes fraction, which puts the text at bottom middle)
            'Total Donors = {:,}'.format(total_donors),  # text to display
            ha='center',  # horizontal alignment is center
            va='top',  # vertical alignment is top
            transform=bar_plot.transAxes)  # this transforms the coordinates to be axes fraction

    plt.xlim(0, 100)  # Set the x-axis limits to 0-100 for proportionate scaling
    plt.tight_layout()
    
    # Save 1st chart
    save_chart(plt, 'chart_1', chart_file_paths)

    ## [Question 1.2] Script for 2nd Chart (Total Blood Donations by State as of Yesterday (updated EoD))

    state_df = blood_type_totals[blood_type_totals['state'] != 'Malaysia']

    state_df = state_df[['state','total']]

    (state_df.sort_values('total', ascending=False, inplace=True))

    state_df.reset_index(drop=True,inplace=True) 

    title = 'Total Blood Donations by State on {} (Yesterday\'s Data)'.format(latest_date)

    # name 2nd Chart as fig2
    fig2, ax = plt.subplots(figsize=(12, 8))
    bar_plot = sns.barplot(x="state",
                        y="total",
                        data=state_df,
                        palette="hls",
                        hue='state',
                        legend=False)

    # Adding the text on top of each bar without decimals
    for p in bar_plot.patches:
        bar_plot.annotate(format(int(p.get_height())), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')

    plt.title(title)
    plt.xlabel("State")
    plt.ylabel("Total")
    plt.xticks(rotation=45)
    plt.tight_layout() # check

    # Save 2nd chart
    save_chart(plt, 'chart_2', chart_file_paths)

    ## [Question 2.1] Script for 3rd Chart (Average Interval Between Visits (days) by Age Group)

    # Redefining age_bins and age_labels
    age_bins = [17, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100]
    age_labels = ['17-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65+']

    # Convert 'visit_date' to datetime
    df_blood_donation['visit_date'] = pd.to_datetime(df_blood_donation['visit_date'], format='%Y-%m-%d', dayfirst=True)

    # Calculate 'visit_year' and 'age_at_visit' (assuming birth_date is in datetime format)
    df_blood_donation['visit_year'] = df_blood_donation['visit_date'].dt.year
    df_blood_donation['age_at_visit'] = datetime.now().year - df_blood_donation['birth_date']

    # Define age groups
    df_blood_donation['age_group'] = pd.cut(df_blood_donation['age_at_visit'], bins=age_bins, labels=age_labels, right=False)

    # Exclude single-time donors
    donor_visit_counts = df_blood_donation['donor_id'].value_counts()
    multiple_donors = donor_visit_counts[donor_visit_counts > 1].index
    blood_donation_data_filtered = df_blood_donation[df_blood_donation['donor_id'].isin(multiple_donors)]
    blood_donation_data_filtered = blood_donation_data_filtered.dropna(subset=['age_group']) # drop age_group not in age_labels


    # Calculate the days between visits for each donor in the filtered dataset
    blood_donation_data_filtered = blood_donation_data_filtered.sort_values(['donor_id', 'visit_date'])
    blood_donation_data_filtered['prev_visit_date'] = blood_donation_data_filtered.groupby('donor_id')['visit_date'].shift(1)
    blood_donation_data_filtered['days_between_visits'] = (blood_donation_data_filtered['visit_date'] - blood_donation_data_filtered['prev_visit_date']).dt.days

    # Average interval between visits for each donor in the filtered dataset
    avg_interval_per_donor_filtered = blood_donation_data_filtered.groupby('donor_id')['days_between_visits'].mean()

    # Merge the average interval with the main dataset
    merged_data_filtered = blood_donation_data_filtered.merge(avg_interval_per_donor_filtered.rename('avg_interval'), on='donor_id')


    # Group by age group to get the average interval for each age group in the filtered dataset
    avg_interval_per_age_group_filtered = merged_data_filtered.groupby('age_group')['avg_interval'].mean()


    avg_interval_table_filtered = avg_interval_per_age_group_filtered.reset_index()
    avg_interval_table_filtered.columns = ['Age Group', 'Average Interval Between Visits (days)']

    # Data from your DataFrame

    # Specify a palette with a different color for each age group
    colors = sns.color_palette('tab10', n_colors=len(avg_interval_table_filtered['Age Group'].unique()))

    # name 3rd Chart as fig3
    fig3, ax = plt.subplots(figsize=(10, 6))
    bar_plot = sns.barplot(x='Age Group', y='Average Interval Between Visits (days)', 
                        data=avg_interval_table_filtered, palette=colors)

    # Iterate over the bars
    for p in bar_plot.patches:
        # Get the height of each bar
        height = p.get_height()
        # Place the text on top of each bar
        bar_plot.text(p.get_x() + p.get_width() / 2., height + 2.5, 
                    f'{height:.0f}', ha='center', va='bottom')

    plt.xlabel('Age Group')
    plt.ylabel('Average Gap Between Visits (days)')
    plt.title('Donor Gap: Average Days Between Blood Donations')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()


    # Save 3rd chart
    save_chart(plt, 'chart_3', chart_file_paths)

    ## [Question 2.2] Script for 4th Chart (Percentage of Blood Donor Visit Types in Malaysia)

    df_sorted = df_blood_donation.sort_values(by=['donor_id', 'visit_date'])

    # Convert 'visit_date' to datetime
    df_blood_donation['visit_date'] = pd.to_datetime(df_blood_donation['visit_date'])

    # Create a new column for the shifted visit dates
    df_sorted['shifted_visit_date'] = df_sorted.groupby('donor_id')['visit_date'].shift(-1)

    # Filter out the rows where shifted_visit_date is NaN (these are the last visits)
    df_second_last = df_sorted[df_sorted['shifted_visit_date'].notna()]

    # Get most recent donation date for each donor
    df_last_visit = (df_blood_donation.groupby('donor_id')['visit_date'].max().reset_index())
    df_last_visit.rename(columns={'visit_date': 'last_visit_date'}, inplace=True)

    # Get the second last visit date for each donor
    df_second_last = df_second_last.groupby('donor_id')['visit_date'].max().reset_index()
    df_second_last.rename(columns={'visit_date': 'second_last_visit_date'}, inplace=True)

    # Merge with the df_last_visit DataFrame
    df_combined = pd.merge(df_last_visit, df_second_last, on='donor_id', how='left')

    # Convert 'last_visit_date', 'second_last_visit_date' to datetime
    df_combined['last_visit_date'] = pd.to_datetime(df_combined['last_visit_date'])
    df_combined['second_last_visit_date'] = pd.to_datetime(df_combined['second_last_visit_date'])
    
    return chart_file_paths

# Function to send charts through the bot
async def send_charts(bot_token, chat_id, chart_file_paths):
    bot = Bot(token=bot_token)
    for file_path in chart_file_paths:
        with open(file_path, 'rb') as photo:
            await bot.send_photo(chat_id=chat_id, photo=photo)

def send_to_tele():
    try:
        # Your bot token and chat_id (Consider retrieving these from environment variables or a secure source)
        # bot_token = os.environ.get('BOT_TOKEN')
        # chat_id = os.environ.get('CHAT_ID')

        # Process data and generate charts
        chart_file_paths = process_data_and_generate_charts()

        # Send charts through the bot
        loop = asyncio.get_event_loop()
        loop.run_until_complete(send_charts(bot_token, chat_id, chart_file_paths))

        return {
            'statusCode': 200,
            'body': 'Process completed successfully.'
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'An error occurred: {str(e)}'
        }

# For local testing (uncomment when testing locally)
if __name__ == "__main__":
    send_to_tele()