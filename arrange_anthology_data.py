import os
import argparse
import glob
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
import argparse
import json

def innertext(elt):
	# return (elt.text or '')
	return (elt.text or '') +(''.join(innertext(e)+(e.tail or '') for e in elt) or '')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--json_save_path", type=str, default="data/acl_anthology.json", help=".json path where the file is to saved",)
	args = parser.parse_args()
	json_save_path = args.json_save_path
	split_path = os.path.split(json_save_path)
	if not os.path.exists(split_path[0]):
		os.makedirs(split_path[0]) 

	cmd = "git clone {}".format("https://github.com/acl-org/acl-anthology.git")
	print("Running " + cmd)
	os.system(cmd)

	# find the paths all XML files
	lst_of_xml_files = glob.glob("acl-anthology/data/xml/*.xml")

	all_conf = {}

	for path in tqdm(lst_of_xml_files):
		with open(path) as xml_file:
			xml = xml_file.read()
			if xml.find("<abstract>") == -1:
					continue
			tree = ET.fromstring(xml)

			for vol_node in tree.iter('volume'):
			  
				publisher = vol_node.find(".//publisher")
				if publisher is None:
					continue
				else:
					publisher = innertext(publisher)
				
				if publisher not in all_conf:
					all_conf[publisher] = {}
				
				try:
					year = innertext(vol_node.find(".//year"))
				except:
					continue
				if year not in all_conf[publisher]:
					all_conf[publisher][year] = {}
				
				booktitle = vol_node.find(".//booktitle")
				if booktitle is not None:
					all_conf[publisher][year]["booktitle"] = innertext(booktitle)
				else:
					all_conf[publisher][year]["booktitle"] = None
				
				month = vol_node.find(".//month")
				if month is not None:
					all_conf[publisher][year]["month"] = innertext(month)
				else:
					all_conf[publisher][year]["month"] = None

				url = vol_node.find(".//url")
				if url is not None:
					all_conf[publisher][year]["url"] = innertext(url)
				else:
					all_conf[publisher][year]["url"] = None
				
				
				all_conf[publisher][year]["papers"] = []

				for node in vol_node.iter('paper'):
					paper_dict = {}
					paper_dict["authors"] = []
					author_name = ""
					for elem in node.iter():
						if not elem.tag==node.tag:
							if elem.tag == "author":
								continue
							elif elem.tag == "first":
								if elem.text is not None:
									author_name = elem.text
							elif elem.tag == "last":
								if elem.text is not None:
									author_name += " " + elem.text
								paper_dict["authors"].append(author_name)
							else:
								paper_dict[elem.tag] = innertext(elem)
					
					all_conf[publisher][year]["papers"].append(paper_dict)

	with open(json_save_path, "w") as f:
		json.dump(all_conf, f)
if __name__ == '__main__':
	main()