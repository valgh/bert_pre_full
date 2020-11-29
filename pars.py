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


def remove_paren(s, par):
	pos_par = None
	for i, c in enumerate(s):
		if c == par:
			pos_par = i
			break
	return pos_par


trial1 = 'this_is_a_trial_<for_patenthesis_filtering>'
trial2 = 'this_is_a_trial_<for_patenthesis_filtering'

#### trial 1
trial1_ret = find_parens(trial1, '<', '>')
# remove useless information, i.e. everithing inside <> or inside ()
if trial1_ret:
	for k, v in trial1_ret.items():
		after_match_parens = trial1[:k + 1] + " " * (v - k - 1) + trial1[v:]
else:
	after_match_parens = trial1
print(after_match_parens)
if after_match_parens == trial1:
	trial1_ret2 = remove_paren(trial1, '<')
	if trial1_ret2 == None:
		pass
	else:
		after_match_parens = after_match_parens[:trial1_ret2]
trial1 = after_match_parens.replace("<", "", 1).replace(">", "", 1)
print(trial1)


#### trial2
trial2_ret = find_parens(trial2, '<', '>')
# remove useless information, i.e. everithing inside <> or inside ()
if trial2_ret:
	for k, v in trial2_ret.items():
		after_match_parens_2 = trial2[:k + 1] + " " * (v - k - 1) + trial2[v:]
else:
	after_match_parens_2 = trial2
print(after_match_parens_2)
if after_match_parens_2 == trial2:
	trial2_ret2 = remove_paren(after_match_parens_2, '<')
	if trial2_ret2 == None:
		pass
	else:
		after_match_parens_2 = after_match_parens_2[:trial2_ret2]
trial2 = after_match_parens_2.replace("<", "", 1).replace(">", "", 1)
print(trial2)