import pandas as pd
import os

# Process obj files in target directory to remove the colour point coordinates
def preprocess_obj(directory_path):
    for file in os.listdir(directory_path):
        vertices = []
        faces = []
        filename, filetype = os.path.splitext(file)
        path = os.path.join(directory_path, file)
        # Check if the file has already been processed
        processed_name = filename + '_processed' + filetype
        processed_path = os.path.join(directory_path, processed_name)
        if file.endswith('.obj') and not os.path.isfile(processed_path):
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('v'):
                        elements = line.split()
                        vertices.append(elements[1:4])
                    if line.startswith('f'):
                        elements = line.split()
                        faces.append(elements[1:])
            
            # Write to obj file
            write_to_obj(processed_path, vertices, faces)

            
def write_to_obj(file_path, vertices, faces):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            file.write(f"f {' '.join(face)}\n")

if __name__ == '__main__':
    preprocess_obj('/homes/rqg23/individualproject/object_autocompletion/data')

        

    