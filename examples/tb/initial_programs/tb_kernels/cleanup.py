from glob import glob
from shutil import copyfile

files = glob('./*.py')

for iF, f in enumerate(files):
    if 'cleanup.py' not in f and '__init__.py' not in f:
        print(f"{iF/len(files)}: cleaning file: {f}")
        try:
            copyfile(f, f+'.bak')
            f_data = open(f, 'r').readlines()
            index = f_data.index("#"*146 + "\n")
            f_data = f_data[:index]
            open(f, 'w').writelines(f_data)
        except Exception as e:
            print(f"Error processing file {f}: {e}")
