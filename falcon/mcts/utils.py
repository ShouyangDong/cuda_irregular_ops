def open_file(file_name):
    with open(file_name, "r") as f:
        code = f.read()
        f.close()
    return code
