import os

def update_first_number(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.readline().strip().split()

    # Check if the file contains exactly 5 numbers
    if len(content) != 5:
        print(f"Skipping file {file_path}. It does not contain 5 numbers.")
        return

    # Update the first number (index 0)
    # new_first_number = 0
    # if bool(content[0]):
    #     new_first_number = 1
    new_height = content[3]
    new_width = content[4]
    #new_first_number = int(bool(content[0]))  # You can modify this line based on your requirement
    #content[0] = str(new_first_number)
    content[3] = new_width
    content[4] = new_height

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(' '.join(content))
        
         
folder_path = "E:/LUNA/Autolabel/1126/yolo_origin_aug/Train/labels/"  # Replace with the path to your folder

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if the file is a text file
    if filename.endswith(".txt"):
        update_first_number(file_path)

print("Modification complete.")


