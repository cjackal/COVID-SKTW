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