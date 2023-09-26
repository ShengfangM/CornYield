import os

def get_files_with_matching_word(file_path: str, matching_word: str) -> list:
    files_list = []

    # Check all files in the specified directory
    for filename in os.listdir(file_path):
        # Check if the word is in the file name (case-insensitive)
        if matching_word.lower() in filename.lower():
            # If the word is found, add the file to the list
            files_list.append(os.path.join(file_path,filename))

    return files_list

def get_subdirectories(path):
    # in_path = os.path.dirname(img_path)
    subdirectories = []
    for root, directories, files in os.walk(path):
        for directory in directories:
            subdirectories.append(os.path.join(root, directory))
    return subdirectories


def get_files_by_suffix(path, suffix:str) ->list:
    file_list = []
    for filename in os.listdir(path):
        file_name, file_extension = os.path.splitext(filename)
        if suffix in file_extension:
            file_list.append(os.path.join(path,filename))
    return file_list