"""
Contains some utils functions for io operations.
"""

import json
import os
import pickle
import csv
import requests


def path_exits(path):
    return os.path.exists(path)


def mkdir(path):
    if not path_exits(path):
        os.mkdir(path)


def makedirs(path):
    if not path_exits(path):
        os.makedirs(path)


def list_files_in_dir(dir):
    return [file for file in os.listdir(dir) if is_file(join(dir, file))]


def list_directories(dir):
    return [subdir for subdir in os.listdir(dir) if os.path.isdir(join(dir, subdir))]


def is_file(path):
    return os.path.isfile(path)


def is_dir(path):
    return os.path.isdir(path)


def join(path1, path2):
    return os.path.join(path1, path2)


def write_json(path, dict, indent=2):
    with open(path, 'w') as outfile:
        if indent == 0:
            json.dump(dict, outfile)
        else:
            json.dump(dict, outfile, indent=indent)
    outfile.close()


def read_json(path):
    with open(path, "r") as infile:
        data = json.load(infile)
    infile.close()
    return data


def read_file_into_list(input_file):
    lines = []
    with open(input_file, "r") as infile_fp:
        for line in infile_fp.readlines():
            lines.append(line.strip())
    infile_fp.close()
    return lines


def write_list_to_file(output_file, list):
    with open(output_file, "w") as outfile_fp:
        for line in list:
            outfile_fp.write(line + "\r\n")
    outfile_fp.close()


def write_text_to_file(output_file, text):
    with open(output_file, "w") as output_fp:
        output_fp.write(text)
    output_fp.close()


def write_pickle(data, file_path):
    pickle.dump(data, open(file_path, "wb"))


def read_pickle(file_path):
    return pickle.load(open(file_path, 'rb'))


def write_to_csv(filepath, header, rows):
    with open(filepath, 'w', encoding='UTF8', newline='') as output_fp:
        writer = csv.writer(output_fp)
        writer.writerow(header)
        writer.writerows(rows)
    output_fp.close()

def get_request_content(end_point, params):
    get_response = requests.get(end_point, params=params)
    return get_response.json()


def download_file(url, download_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
    }
    r = requests.get(url, stream=True, allow_redirects=True, headers=headers)
    with open(download_path, 'wb') as f:
        for ch in r:
            f.write(ch)
