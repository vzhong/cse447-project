# Cleans the files found at https://databus.dbpedia.org/dbpedia/collections/latest-core,
# in the "Replaced IRIs of DBpedia long-abstracts" section

import re
import argparse

def corpuscleaner(original_file_path, new_file_path):
	number = 0
	with open(original_file_path, "r+") as original_file, open(new_file_path, "w+") as clean_file:
		for line in original_file:
			clean = re.sub("[\(\[\<].*?[\>\)\]]", "", line) # Delete anything between angle brackets, brackets, and parenthesis
			
			# get rid of weird spaces (\u200b)
			split = clean.split()
			for i in range(len(split)):
				split[i] = split[i].strip(u'\u200b')
			clean = " ".join(split)

			clean = re.sub(r'\s+([?.!,])', r'\1', clean) #the join from above will mess up commas and punctutation.
														 # ie, it will turn "arjun, singla" into "arjun , singla".
														 # This line undoes that. 

			#Get rid of the first quotations and the trailing language tag
			clean = clean[1:(len(clean) - 6)]

			clean_file.write(clean)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()                                               
	parser.add_argument("--file", "-f", type=str, required=True)
	args = parser.parse_args()
	src_file = args.file
	dest_file = "clean_"+src_file
	corpuscleaner(src_file, dest_file)