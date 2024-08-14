# Open the text file in read mode
with open('short_reviews/positive.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

    # Count the number of rows (lines) in the file
    num_rows = len(lines)

    # Split the first line to determine the number of columns
    # Assuming the columns are separated by a specific delimiter, such as a comma (',') or tab ('\t')
    # For example, if columns are separated by commas, use: num_columns = len(lines[0].strip().split(','))
    num_columns = len(lines[0].strip().split('\t'))  # Assuming columns are separated by tabs
    
# Print the number of rows and columns
print(f'Number of rows: {num_rows}')
print(f'Number of columns: {num_columns}')
