# utilty functions developed my Juhyun Kim, and editted by HyeongChan Jo

def correct_FIPS(fips):
    # We need to add back the leading zero to some of the FIPS codes
    # since they were removed when loaded them as floats
    fips = str(int(fips))
    if len(fips) == 4:
        fips = '0' + fips
    return fips

def get_homedir(verbose=False):
    # Simply get the root directory
    import git

    homedir = git.Repo("./", search_parent_directories=True).working_dir
    if verbose:
        print(homedir)
    return homedir

def get_FIPS(path='/exploratory_HJo/FIPS_mapping.txt', reduced=False):
    """
    Load FIPS mapping in dictionary format of the form {alias fips:genuine fips}.
    Designed to have two usage:
    (reduced=True) Set the reference FIPS for training, convert data FIPS into training FIPS
    (reduced=False) Convert prediction data into submission format
    Returns:
      FIPS_mapping: 
        dictionary with key=FIPS to be replaced, value=replaced FIPS.
      FIPS_full:
        list of full FIPS.
        Same as list(FIPS_mapping.values())
    """
    homedir = get_homedir()
    with open(f'{homedir}'+path, 'r') as f:
        dic = eval(f.read())
    # Take inverse of the dictionary
    dic_inv = {}
    for key, value in dic.items():
        dic_inv[value] = dic_inv.setdefault(value, set())
        dic_inv[value].add(key)
    # Disregard identity part of dic_inv
    alias = {}
    for fips, value in dic_inv.items():
        if len(value)>1:
            alias[fips] = value.difference({fips})
    
    FIPS_mapping = {}
    for fips, alias_fips in alias.items():
        for dummy in alias_fips:
            FIPS_mapping[dummy] = fips
    FIPS_full = set(dic.keys())
    if reduced:
        for fips, alias_fips in alias.items():
            if fips[:2]=='02' and len(alias_fips)==1:
                dummy ,= alias_fips
                FIPS_full.discard(dummy)
            elif fips[:2]=='02' and len(alias_fips)>1:
                FIPS_full.discard(fips)
                for dummy in alias_fips:
                    FIPS_mapping.pop(dummy)
            elif fips=='46113':
                dummy ,= alias_fips
                FIPS_full.discard(dummy)
            else:
                for dummy in alias_fips:
                    FIPS_mapping.pop(dummy)
    else:
        compl = set().union(*alias.values())
        FIPS_full = FIPS_full.difference(compl)
    return FIPS_mapping, sorted(FIPS_full)

def fix_FIPS(df, fipslabel=None, datelabel=None, **kwargs):
    """
    Fix FIPS in a dataframe using get_FIPS function.
    Accept both wide and tidy formats.
    CAVEAT: may reindex the dataframe.
            always put date and FIPS columns on the left.
            It returns a dataframe, not done in place.
    
    Parameters:
      df:
        A dataframe with FIPS column.
        Can be wide(having one column per date) or tidy(having single column of dates).
      fipslabel:
        Name of the FIPS column.
      datelabel:
        Name of the date column, only necessary when df is tidy.
    Returns:
        Modified dataframe.
    """
    df_modified = df
    FIPS_mapping, FIPS_full = get_FIPS(**kwargs)
    if fipslabel is not None:
        df_modified.replace({fipslabel:FIPS_mapping}, inplace=True)
        if datelabel is None:
            return df_modified.groupby(fipslabel, as_index=False).sum(min_count=1)
            # return df_modified.groupby(fipslabel, as_index=False).apply(lambda x: x.sum(min_count=1))
        else:
            return df_modified.groupby([datelabel, fipslabel], as_index=False).sum(min_count=1)
            # return df_modified.groupby([datelabel, fipslabel], as_index=False).apply(lambda x: x.sum(min_count=1))
    else:
        return df