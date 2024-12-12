from datetime import datetime

# Define the time ranges
time_ranges = [
    ("5pm Fri 20th Dec", "12pm Tues 24th Dec"),
    ("12pm Tues 24th Dec", "9am Sun 29th Dec"),
    ("9am Sun 29th Dec", "12pm Thurs 2nd Jan"),
    ("12pm Thurs 2nd Jan", "9am Mon 6th Jan")
]

# Define a helper function to parse times and calculate the difference in hours
def calculate_hours(start, end):
    time_format = "%I%p %a %dth %b"  # Example format: "5pm Fri 20th Dec"
    start_dt = datetime.strptime(start, time_format)
    end_dt = datetime.strptime(end, time_format)
    delta = end_dt - start_dt
    return delta.total_seconds() / 3600  # Convert seconds to hours

# Calculate hours for each range
hours = [calculate_hours(start, end) for start, end in time_ranges]
hours
