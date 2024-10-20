import pandas as pd
from sqlalchemy import create_engine

# Connect to the database
engine = create_engine('sqlite:///attendance.db')

def analyze_attendance():
    df = pd.read_sql("SELECT name, timestamp FROM attendance", con=engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Count how many times each employee was present
    report = df.groupby('name').size().reset_index(name='Attendance Count')
    print(report)

analyze_attendance()
