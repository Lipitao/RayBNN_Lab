import os
import subprocess
import time
import sys
import pathlib
import shutil


def main():
	for root, dirs, files in os.walk(os.path.abspath(sys.argv[1])):
		for filename in files:
			filename = os.path.abspath(os.path.join(root, filename))
			if (".rs" in filename) and ("_f64" in filename):
				newfilename = filename.replace("_f64","_f32")
				shutil.copyfile(filename,newfilename)
				print(newfilename)
				print(filename)


				reading_file = open(newfilename, "r")
				new_file_content = ""
				for line in reading_file:
					stripped_line = line
					new_line = stripped_line.replace("f64", "f32")
					new_file_content += new_line
				reading_file.close()

				writing_file = open(newfilename, "w")
				writing_file.write(new_file_content)
				writing_file.close()




if __name__ == '__main__':
    main()
