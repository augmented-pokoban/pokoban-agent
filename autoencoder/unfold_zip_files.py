import os
from zipfile import ZipFile, ZIP_DEFLATED

path = 'zipped/'

files = filter(lambda f: '2017-12-23_17_34_32.066026.zip' in f, os.listdir(path))

for file in files:
    print(file)
    if '2017-12-23_17_34_32.066026.zip' in file:

        # read zip with compression
        with ZipFile(path + file, 'r') as zip_file:
            file_content = zip_file.open(zip_file.namelist()[0])

        # write back without compression
        unzipped_filename = path + 'unzip_' + file
        with open(unzipped_filename, 'wb') as unzipped:
            unzipped.write(file_content.read())

        # read zipped file without compression
        with ZipFile(unzipped_filename, 'r') as uncompressed_file:
            file_content = uncompressed_file.open(uncompressed_file.namelist()[0])

        # write back as unzipped file
        uncompressed_filename = path + 'unc_' + file
        with open(uncompressed_filename, 'wb') as uncompressed_file:
            uncompressed_file.write(file_content.read())

        # now it is pickle, zip again
        with ZipFile(path + file, 'w', ZIP_DEFLATED) as zip_file:
            zip_file.write(uncompressed_filename)
            zip_file.close()

        os.remove(unzipped_filename)
        os.remove(uncompressed_filename)




