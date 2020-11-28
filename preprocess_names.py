import os
import re
import cxxfilt
import json
import multiprocessing as mp
from multiprocessing import Value

discarded = Value('i', 0)
no_name = Value('i', 0)

def demangle_name(name):
	demangled_name = None
	try:
		demangled_name = cxxfilt.demangle(name)
	except:
		demangled_name = None
	return demangled_name



def find_parens(s, par1, par2):
	toret = {}
	pstack = []
	for i, c in enumerate(s):
		if c == par1:
			pstack.append(i)
		elif c == par2:
			if len(pstack) == 0:
				continue
			toret[pstack.pop()] = i
	return toret



def camel_to_snake(name):
	_underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
	_underscorer2 = re.compile('([a-z0-9])([A-Z])')
	subbed = _underscorer1.sub(r'\1_\2', name)
	return _underscorer2.sub(r'\1_\2', subbed).lower()


def filter_name(x):
	global discarded
	global no_name
	info_file = x[0]
	walk_file = x[1]
	names_dir = x[2]
	name_file = x[3]
	z_encoding = x[4]
	name = None
	demangled_name = None
	with open(info_file, 'r') as f:
		function_info = json.load(f)
	if type(function_info) == type(dict()):
		try:
			name = function_info['name']
		except:
			with no_name.get_lock():
				no_name.value += 1
			# name not retrieved, must use the one on file
			if name_file[:-10] == '.json.json':
				name = name_file[:-10]
			else:
				name = name_file[:-5] # leave out .json
	else:
		for x in function_info:
			if x[0] == 'name':
				name = x[1]
		if name == None:
			with no_name.get_lock():
				no_name.value += 1
			# name not retrieved, must use the one on file
			if name_file[:-10] == '.json.json':
				name = name_file[:-10]
			else:
				name = name_file[:-5] # leave out .json
	if name != None:
		demangled_name = demangle_name(name)
		if demangled_name != None:
			matching_par1 = find_parens(demangled_name, "<", ">")
			if demangled_name == name and not matching_par1:
				# do none, demangler has no effect: this function has already been mangled by Ghidra,
				# or it's a simple c/cpp function
				pass
			else:
				# remove useless information, i.e. everithing inside <> or inside ()
				for k, v in matching_par1.items():
					demangled_name = demangled_name[:k + 1] + " " * (v - k - 1) + demangled_name[v:]
				matching_par2 = find_parens(demangled_name, "(", ")")
				for k, v in matching_par2.items():
					demangled_name = demangled_name[:k + 1] + " " * (v - k - 1) + demangled_name[v:]
				demangled_name = ' '.join(demangled_name.split())
				if "::" in demangled_name.split("(")[0]:
				# take the substring between :: and (
					res = re.findall(r'(::((?!::).)*?\()', demangled_name, re.S)
					if len(res) > 0:
						demangled_name = res[len(res) - 1][0].replace("::", "").replace("(", "").replace("~", "")
				else:
					# take the substring before (
					demangled_name = demangled_name.split("(")[0]
				# remove matching <> if they are present in the function name
				final_matching_par = find_parens(demangled_name, "<", ">")
				if final_matching_par:
					demangled_name = demangled_name.replace("<", "", 1).replace(">", "", 1)
			demangled_name = camel_to_snake(demangled_name)
			# now filter pecial characters
			# replace special characters with '_'
			demangled_name = demangled_name.replace(".", "_").replace(":", "_").replace(
						"$", "_").replace("(", "_").replace(")", "_")
			# also, filter out numbers, strings inside brackets and
			# multiple withe spaces
			demangled_name = re.sub(r"[0-9]", "_", demangled_name)
			demangled_name = re.sub(r'\[.*?\]', '_', demangled_name)
			demangled_name = re.sub(r'\s+', '_', demangled_name).strip()
			# eliminate function encoded with z-encoding which are written in haskell
			function_name_tokens = set(demangled_name.split(" "))
			if function_name_tokens.intersection(z_encoding) or "haskell" in demangled_name:
				with discarded.get_lock():
					discarded.value += 1
				demangled_name = None
			# eliminate java functions, pyx functions and Go functions (GO functions have name starting with github)
			if demangled_name.startswith("java") or demangled_name.startswith("github") or demangled_name.startswith("pyx") or demangled_name.startswith("caml"):
				with discarded.get_lock():
					discarded.value += 1
				demangled_name = None
		else:
			with discarded.get_lock():
				discarded.value +=1
	else:
		with discarded.get_lock():
			discarded.value += 1
	# save routine
	#if demangled_name != None:
	#	with open(names_dir+name_file, 'w') as f:
	#		json.dump(demangled_name, f)
	return


def main():
	chunks = 10
	pool = mp.Pool(4)
	fun_dir = '/home/valerio/valerio/dataset/multiverse/' # main/universe
	z_encoding = {'za', 'zb', 'zc', 'zd', 'ze', 'zg', 'zh', 'zi', 'zl', 'zm', 'zn', 'zp', 'zq', 'zr', 'zs',
					'zt', 'zu', 'zv'}
	to_filter = []
	for binary in os.listdir(fun_dir):
		binary_dir = fun_dir+binary
		info_dir = binary_dir+'/functions_info/'
		walk_dir = binary_dir+'/randwalks_text_pro/'
		names_dir = binary_dir+'/names/'
		if os.path.exists(walk_dir) and os.path.exists(info_dir):
			#if not os.path.exists(names_dir):
			#	os.mkdir(names_dir)
			for name_file in os.listdir(walk_dir):
				info_file = info_dir+name_file
				walk_file = walk_dir+name_file
				to_filter.append((info_file, walk_file, names_dir, name_file, z_encoding))
	pool.map(filter_name, to_filter, chunks)
	pool.close()
	pool.join()
	print('\nTotal discarded: {}'.format(discarded.value))
	print('\tTotal no name: {}\n\n'.format(no_name.value))


if __name__ == '__main__':
	main()