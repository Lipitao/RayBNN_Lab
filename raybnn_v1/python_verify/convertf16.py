import os
import subprocess
import time
import sys
import pathlib
import shutil
import re


def split_by_char(instr,tchar):
	idx = -1
	for ii in range(len(instr)):
		cur_char = instr[ii]
		if cur_char == tchar:
			idx = ii
			break
	if idx == -1:
		return [instr]
	return [instr[:idx],instr[idx:] ]



def parseArrConst(instr):
	find_idx = 0

	targetstr = instr
	start_token = "arrayfire::constant::<"
	token_len = len(start_token)
	idx = targetstr.find(start_token, find_idx)
	while idx > -1:
		firststr = targetstr[:(idx+token_len)]
		laststr = targetstr[(idx+token_len):]

		find_idx = idx+token_len

		arr = split_by_char(laststr,'(')
		firststr = firststr + arr[0] + "("
		laststr = arr[1][1:]

		arr = split_by_char(laststr,',')
		if ("." in arr[0]) and (not (".clone" in arr[0])):
			arr[0] = arr[0].replace("f64", "")
			arr[0] = arr[0].replace("f32", "")
			arr[0] = arr[0].replace(" ", "")
			arr[0] = arr[0].replace("\t", "")
			arr[0] = "f16::from_f32_const(" + arr[0] + "f32)"
		
		targetstr = firststr + arr[0] + arr[1]

		idx = targetstr.find(start_token, find_idx)
	return targetstr




def split_by_char_reverse(instr,tchar):
	idx = -1
	for ii in reversed(range(len(instr))):
		cur_char = instr[ii]
		if cur_char == tchar:
			idx = ii
			break
	if idx == -1:
		return [instr]
	idx = idx + 1
	return [instr[:idx],instr[idx:] ]




def split_by_alpha_reverse(instr):
	idx = -1
	for ii in reversed(range(len(instr))):
		cur_char = instr[ii]
		if not (cur_char.isalnum() or cur_char == '_'):
			idx = ii
			break
	if idx == -1:
		return [instr]
	idx = idx + 1
	return [instr[:idx],instr[idx:] ]


def split_by_char_reverse_until(instr,tchar,count):
	idx = -1
	cx = 0
	for ii in reversed(range(len(instr))):
		cur_char = instr[ii]
		if cur_char == tchar:
			cx = cx + 1
			if cx == count:
				idx = ii
				break
	if idx == -1:
		return [instr]
	#idx = idx + 1
	return [instr[:idx],instr[idx:] ]





def parseAsCast_single(instr):
	find_idx = 0

	targetstr = instr
	start_token = " as "
	token_len = len(start_token)

	end_token = "f64"
	token_len2 = len(end_token)
	idx = targetstr.find(start_token, find_idx)
	if idx > -1:
		firststr = targetstr[:(idx+token_len)]
		midstr = targetstr[(idx+token_len):]

		

		idx2 = midstr.find(end_token, find_idx)
		if idx2 == -1:
			return targetstr

		laststr = midstr[(idx2+token_len2):]
		midstr = midstr[:(idx2+token_len2)]

		if midstr.replace(" ","") != end_token:
			return targetstr


		arr = split_by_char_reverse(firststr,'(')
		arg_token = ""
		before_token = ""

		if len(arr) == 1:
			newlen = len(firststr) - token_len
			firststr = firststr[:newlen]

			arr = split_by_alpha_reverse(firststr)

		count2 = arr[1].count(')')

		arr[1] = arr[1].replace(" as ", "")
		
		if count2 == 0:
			arg_token = arr[1]
			before_token = arr[0]
		else:
			arr = split_by_char_reverse_until(arr[0]+arr[1],'(',count2)
			arg_token = arr[1]
			before_token = arr[0]
		

		targetstr =  before_token + "f16::from_f32_const(" + arg_token  + " as f32)" 

		#find_idx = len(targetstr)
		targetstr = targetstr + laststr
		#idx = targetstr.find(start_token, find_idx)
	return targetstr



def parseAsCast(instr):
	find_idx = 0

	targetstr = instr
	start_token = " as "
	token_len = len(start_token)

	end_token = "f64"
	token_len2 = len(end_token)
	idx = targetstr.find(start_token, find_idx)
	while idx > -1:
		firststr = targetstr[:(idx+token_len)]
		midstr = targetstr[(idx+token_len):]

		idx2 = midstr.find(end_token)
		if idx2 == -1:
			break

		laststr = midstr[(idx2+token_len2):]
		midstr = midstr[:(idx2+token_len2)]

		

		if midstr.replace(" ","") != end_token:
			find_idx = idx + token_len
			idx = targetstr.find(start_token, find_idx)
			continue
		
		
		targetstr = firststr+midstr
		tempstr = targetstr[find_idx:]
		targetstr = targetstr[:find_idx]
		
		targetstr = targetstr + parseAsCast_single(tempstr)

		find_idx = len(targetstr)
		
		targetstr = targetstr + laststr
		idx = targetstr.find(start_token, find_idx)
		
	return targetstr



def parseInto(instr):
	find_idx = 0

	targetstr = instr
	start_token = " as "
	token_len = len(start_token)

	end_token = ["bool","i8","i16","i32","i64","i128","isize","u8","u16","u32","u64","u128","usize","f32","f64"]
	
	idx = targetstr.find(start_token, find_idx)
	prev_idx = idx
	while idx > -1:
		firststr = targetstr[:(idx+token_len)]
		midstr2 = targetstr[(idx+token_len):]


		for cur_token in end_token:
			idx2 = midstr2.find(cur_token)
			token_len2 = len(cur_token)
			if idx2 == -1:
				continue

			laststr = midstr2[(idx2+token_len2):]
			midstr = midstr2[:(idx2+token_len2)]

			if midstr.replace(" ","") != cur_token:
				continue
			
			
			firststr = targetstr[:(idx)]
			targetstr = firststr + ".into()"

			find_idx = len(targetstr)
		
			targetstr = targetstr + laststr
			idx = targetstr.find(start_token, find_idx)
			break
		if idx == prev_idx:
			break
		else:
			prev_idx = idx
		
	return targetstr





def main():
	for root, dirs, files in os.walk(os.path.abspath(sys.argv[1])):
		for filename in files:
			filename = os.path.abspath(os.path.join(root, filename))
			if (".rs" in filename) and ("_f64" in filename):
				newfilename = filename.replace("_f64","_f16")
				shutil.copyfile(filename,newfilename)
				print(newfilename)
				print(filename)


				
				reading_file = open(newfilename, "r")
				new_file_content = "use half;\r\n"
				new_file_content += "use half::f16;\r\n"
				new_file_content += "use num::traits::real::Real;\r\n"
				for line in reading_file:
					stripped_line = line.replace("\t", "   ")
					if ("const" ==  stripped_line.replace(" ", "")[:5] ) and ("f64"  in  stripped_line):
						new_line = stripped_line.replace("f64", "f16")
						arr = new_line.split(" ")

						new_line = ""
						state = 0
						for token in arr:
							new_token = token
							if state == 1:
								new_token = token.replace(";", "")
								new_token = new_token.replace("\n", "")
								new_token = new_token.replace("\r", "")
								new_token = new_token.replace("f64", "")
								new_token = new_token.replace("f32", "")
								new_token = "f16::from_f32_const(" + new_token + "f32)"
								
								state = 2
							if token == "=":
								state = 1
							new_line += (new_token + " ")
						if not (";" in new_line):
							new_line += ";\r\n" 
					else:

						if ("arrayfire::constant::<f64>" in stripped_line.replace(" ", "")):
							stripped_line = parseArrConst(stripped_line)
						
						if (" as " in stripped_line)  and ("asf64" in stripped_line.replace(" ", "") ):
							stripped_line = parseAsCast(stripped_line)
						
						if (" as " in stripped_line) :
							stripped_line = parseInto(stripped_line)

						new_line = stripped_line.replace("f64", "f16")
						#for token in const_arr:
						#	new_line = re.sub("\\b"+token+"\\b", "half::f16::from_f64("+token+")", new_line)

					new_file_content += new_line
				reading_file.close()

				writing_file = open(newfilename, "w")
				writing_file.write(new_file_content)
				writing_file.close()


				




if __name__ == '__main__':
	main()
