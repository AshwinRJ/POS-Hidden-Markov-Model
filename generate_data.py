#################################################################################
#																				#
# Generates a single text file from the sequence of files and cleans the data	#
#																				#
#################################################################################

import os

# Some useful variables
data_dir = './Data/'
file_names = ['book1.txt', 'book2.txt', 'book3.txt', 'book4.txt', 'book5.txt']
output_file = 'data.txt'
lower_case = True
remove_numbers = True
replace_numbers = False
sentence_per_line = True

accepted_characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
numbers = '1234567890'

# Read the data
data = ''
for filename in file_names:
	f = open(os.path.join(data_dir, filename))
	text = f.read().strip().replace('\t', '').replace('\n', '')
	
	if lower_case:
		text = text.lower()
	
	if remove_numbers:
		text = text.replace('1', '').replace('2', '').replace('3', '')
		text = text.replace('4', '').replace('5', '').replace('6', '')
		text = text.replace('7', '').replace('8', '').replace('9', '')
		text = text.replace('*', '').replace('0', '').replace('#', '')
	
	n = len(text)
	i = 0
	while i < n:
		if text[i] == '.':	# New line
			if sentence_per_line:
				data += '\n'
			else:
				data += ' '
			i += 1
			while  i<n and (text[i] not in numbers and text[i] not in accepted_characters):
				i += 1
		
		elif text[i] in numbers:
			if replace_numbers:
				j = i+1
				while j+1 < n:
					if text[j+1] in numbers:
						j += 1
					else:
						break
				i = j + 1
				data += '__NUMBER__'
			else:
				data += text[i]
				i += 1
		
		elif text[i] in accepted_characters:
			data += text[i]
			i += 1
		
		elif text[i] == ' ':
			data += ' '
			i += 1
			while  i<n and text[i]==' ':
				i += 1
		
		else:
			i += 1
	
	if sentence_per_line:
		data += '\n'
	
	f.close()

		
data = data[:-1]
f = open(os.path.join(data_dir, output_file), 'w')
f.write(data)
f.close()
