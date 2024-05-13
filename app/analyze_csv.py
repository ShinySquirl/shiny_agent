import pandas as pd

def load_data(filename):
    """Load LinkedIn data from a CSV file."""
    return pd.read_csv(filename)

def find_potential_cofounders(data, target_company, target_position):
    """Find connections working at a specific company and position."""
    # Filter by company and position
    filtered_data = data[(data['Company'].str.contains(target_company, case=False, na=False)) & 
                         (data['Position'].str.contains(target_position, case=False, na=False))]
    return filtered_data

# Load data
linkedin_data = load_data('/Users/nicbaird/Desktop/agent_hack/Connections.csv')

# Define your target company and position
target_company = 'Software'
target_position = 'Founder'

# Find potential cofounders
potential_cofounders = find_potential_cofounders(linkedin_data, target_company, target_position)

# Print potential cofounders
print(potential_cofounders[['First Name', 'Last Name', 'Company', 'Position', 'URL (LinkedIn Profile)']])

#tabling this for now because I don't have an contact info in the LI download, so it doesn't work to send messages. Instead, I'm going to focus on the agent framework to send the messages.