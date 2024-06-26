import os

# Define the directories
images_dir = 'dataset/valid/images'
labels_dir = 'dataset/valid/labels'

# Ensure the labels directory exists
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

# Iterate over each image file in the images directory
for img_file in os.listdir(images_dir):
    if os.path.isfile(os.path.join(images_dir, img_file)):
        # Extract the alphabet character from the image filename
        alphabet = img_file[0].upper()
        
        # Create the corresponding label filename
        label_file = os.path.splitext(img_file)[0] + '.txt'
        
        # Write the alphabet to the label file
        with open(os.path.join(labels_dir, label_file), 'w') as file:
            file.write(alphabet)

print('Label files have been created successfully.')
