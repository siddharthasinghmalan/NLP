
file_path = "short_reviews/negative.txt"

try:
    with open(file_path, 'r') as file:
        file_content = file.read()
        words = file_content.split()
        num_words = len(words)
        print("Number of words in the file:", num_words)
except FileNotFoundError:
    print("The specified file was not found.")
except Exception as e:
    print("An error occurred:", str(e))

eighty =  num_words * 0.8
print(eighty)