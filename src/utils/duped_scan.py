import os

dir = "/home/leprieto/tfm/resources/dataset/casting_data"

def find_duplicate_files(directory):
    """Find and delete duplicate files based on byte-by-byte comparison."""
    file_hashes = {}
    files_to_remove = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            with open(file_path, 'rb') as f:
                file_content = f.read()

            file_hash = hash(file_content)
            if file_hash in file_hashes:
                relpath1 = os.path.relpath(file_hashes[file_hash], dir)
                relpath2 = os.path.relpath(file_path, dir)
                files_to_remove.append(relpath2)
                print(f"Duplicated: {relpath1} <===> {relpath2}")
            else:
                file_hashes[file_hash] = file_path

    return files_to_remove

def delete_files(files):
    """Delete the duplicate files."""
    count = 0
    for file in files:
        os.remove(file)
        print(f"Removed: {file}")
        count += 1
    return count

files_to_remove = find_duplicate_files(dir)
count = delete_files(files_to_remove)

print(f"Files removed: {count}")