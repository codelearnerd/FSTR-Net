import os

def replace_path(src_name, last_path, ext):

    dirname, filename = os.path.split(src_name)
    base, file_extension = os.path.splitext(filename)


    parts = dirname.split(os.path.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = last_path
            break


    new_dirname = os.path.sep.join(parts)


    new_filename = base + ext


    dst_name = os.path.join(new_dirname, new_filename)

    return dst_name

def replace_last_path(path,new_folder_name):

    dirname, basename = os.path.split(path)


    new_path = os.path.join(dirname, new_folder_name)
    return new_path


def add_suffix_to_filename(path, suffix, ext2=None):

    dir_name, base_name = os.path.split(path)

    file_name, ext = os.path.splitext(base_name)

    new_file_name = f"{file_name}{suffix}{ext}" if ext2==None else f"{file_name}{suffix}{ext2}"

    new_path = os.path.join(dir_name, new_file_name)
    return new_path

def is_valid_file(file_path):


    return os.path.exists(file_path) and os.path.isfile(file_path)

def remove_suffix(filename,suffix):

    name, ext = filename.rsplit('.', 1)

    new_name = name.replace(suffix, "") + '.' + ext
    return new_name

from pathlib import Path

def replace_path_part(json_path):
    path = Path(json_path)

    parts = list(path.parts)

    parts[-3] = "leftImg8bit"

    new_path = str(Path(*parts))
    return new_path
