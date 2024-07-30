import os


# Process obj files in target directory to remove the colour point coordinates
def preprocess_obj(directory_path, target_directory):

    directory_path = os.path.expanduser(directory_path)
    target_directory = os.path.expanduser(target_directory)

    os.makedirs(target_directory, exist_ok=True)
    for file in os.listdir(directory_path):
        vertices = []
        faces = []
        source_path = os.path.join(directory_path, file)
        path = os.path.join(directory_path, file)
        # Check if the file has already been processed
        processed_path = os.path.join(target_directory, file)
        if file.endswith('.obj') and not os.path.isfile(processed_path):
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        elements = line.split()
                        vertices.append(elements[1:4])
                    if line.startswith('f'):
                        elements = line.split()
                        faces.append([int(index.split('/')[0]) for index in elements[1:]])
            
            # Write to obj file
            write_to_obj(processed_path, vertices, faces)
            print(f'Processed {file}')

            
def write_to_obj(file_path, vertices, faces):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            face_indices = ' '.join(map(str, face))
            file.write(f"f {face_indices}\n")


if __name__ == '__main__':
    source_dir = '~/../../vol/bitbucket/rqg23/ground_truth_faceCompletion'
    target_dir = '~/../../vol/bitbucket/rqg23/ground_truth_obj_processed'
    # source_dir = '/Users/raymondguo/Desktop/IndividualProject/object_autocompletion/data'
    # target_dir = '/Users/raymondguo/Desktop/IndividualProject/object_autocompletion'
    preprocess_obj(source_dir, target_dir)
        

    