import psycopg2
import pandas as pd

# PostgreSQL connection details replace with your own
host = "***********"
database = "***********"
user = "***********"
password = "************"

# Connect to PostgreSQL
try:
    connection = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    print("Connection to PostgreSQL established.")
    
    # Create a cursor object
    cursor = connection.cursor()
    
    # Define the views to import
    views = ['vw_churndata', 'vw_joindata']
    
    for view in views:
        # Execute SELECT query
        query = f"SELECT * FROM {view};"
        cursor.execute(query)
        
        # Fetch all data
        data = cursor.fetchall()
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Create a DataFrame
        df = pd.DataFrame(data, columns=column_names)
        
        # Save to CSV
        df.to_csv(f"{view}.csv", index=False)
        print(f"Data from {view} saved to {view}.csv")
    
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Close the cursor and connection
    if cursor:
        cursor.close()
    if connection:
        connection.close()
    print("PostgreSQL connection closed.")
