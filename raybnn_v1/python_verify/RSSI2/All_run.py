import sys
import os
import subprocess
import time

def main():
	filename = sys.argv[1]
	fold = 10
	input_size_list = [6,8,12,14,16,32,50,75,100,125,162]
	for q in range(fold):
		for i in range(len(input_size_list)):

			proc = subprocess.Popen(["python3",  filename, str(input_size_list[i]), str(q) ])
			proc.wait()
			time.sleep(5)




if __name__ == '__main__':
    main()
