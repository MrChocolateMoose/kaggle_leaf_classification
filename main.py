from tqdm import tqdm
import requests
import os.path
import getpass
import hashlib

def get_md5(data_file):
    md5 = None
    with open(data_file, "rb") as handle:
        # read contents of the file
        data = handle.read()
        # pipe contents of the file through
        md5 = hashlib.md5(data).hexdigest()

    return md5

def ensure_data():

    md5s = [
        "af0a4da25fd4dac59f32f5cf106ea1fb",
        "61fd54fbd207757eaa63434ee04293ee",
        "fa4407496de3ee6454a49774a8575461",
        "9ffeb9fa5cb12a1133233f904d77160c",
    ]

    data_subdirectory = "data"
    data_base_files = [
        "sample_submission.csv.zip",
        "train.csv.zip",
        "test.csv.zip",
        "images.zip"
    ]
    assert(len(md5s) == len(data_base_files))

    data_base_url = "https://www.kaggle.com/c/leaf-classification/download/"
    data_urls = [data_base_url + data_file for data_file in data_base_files]

    data_files = [os.path.join(data_subdirectory, data_base_file) for data_base_file in data_base_files]

    data_files_urls_md5s = zip(data_files, data_urls, md5s)

    has_entered_credentials = False
    kaggle_login_info = None


    for data_file, data_url, md5 in data_files_urls_md5s:

        if not os.path.isfile(data_file):
            print("file missing: " + data_file)

            if not has_entered_credentials:
                has_entered_credentials = True
                user_name = input("Kaggle User Name: ")
                password = input("Kaggle Password: ") # getpass.getpass("Kaggle Password: ")
                kaggle_login_info = {'UserName': user_name, 'Password': password}

            response = requests.get(data_url)
            response = requests.post(response.url, data=kaggle_login_info)

            with open(data_file, "wb") as handle:
                for chunk in response.iter_content(chunk_size=512 * 1024):  # Reads 512KB at a time into memory
                    if chunk:  # filter out keep-alive new chunks
                        handle.write(chunk)
        else:
            print("file found: " +  data_file)

        computed_md5 = get_md5(data_file)

        if md5 != computed_md5:
            print("md5 does not match for: " + data_file)
        else:
            print("md5 matches for: " + data_file)

ensure_data()

