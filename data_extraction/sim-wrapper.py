#!/usr/bin/env python3
import sys
import os
from glob import glob
import argparse
import subprocess
from pathlib import Path

"""
TO DO:
    -Clean up outputs, specfically order the outputs in ascending order first (should be easy?)
        -Also when getting the version clear out uneccessary folders (like rootfiles and stuff)
        -This also means make the outputs more consistent (ex: EVERY listing has a endl afterwards)
    -Add in the whole pass-parameter methods to sim-maker.py and sim-converter.py (Definitely important)
    -Add more comments
    -Make the code cleaner
"""

def updatePath(currPath, newArg):
    return(os.path.join(currPath, newArg))

class Back(Exception):
    """Raised when the user backs out of a prompt; redo the selection pass."""


# One picker for every prompt: numbered list, answer by number(s), name(s),
# 'a' = all, 'b' = back. Names still accepted so old habits keep working.
_autoall = False  # 'aa' seen: answer every remaining prompt with all options


def pickMulti(label, options, aliases=None):
    global _autoall
    options = list(options)
    print(f"Available {label}:")
    for i, opt in enumerate(options, 1):
        print(f"[{i}] {opt}")
    while(True):
        if(_autoall):
            return options
        raw = input(f"Pick {label} (numbers/names, space-separated, 'a'=all, 'b'=back, 'aa'=all from here on): ").strip()
        if(raw.lower() in ("b", "back")):
            raise Back()
        if(raw.lower() in ("a", "all")):
            return options
        if(raw.lower() == "aa"):
            _autoall = True
            return options
        picks = []
        for tok in raw.replace(",", " ").split():
            tok = tok.strip()
            if(tok.isdigit() and 1 <= int(tok) <= len(options)):
                picks.append(options[int(tok) - 1])
            else:
                match = [o for o in options if o.lower() == tok.lower()]
                if not match and aliases:
                    match = [o for o, al in aliases.items() if tok.lower() in [a.lower() for a in al] and o in options]
                picks.append(match[0] if len(match) == 1 else None)
                if len(match) != 1:
                    picks = None
                    break
        if(picks):
            return(list(dict.fromkeys(picks)))
        print("Invalid selection was made.")

#getYear function, gets a selection of all the valid years and gets the user input on the wanted year
#Only list folders that actually hold cosmic-ray primaries; avoids dead ends
#like 2016 (GCD only), gamma-only years, oldstructure and temp dirs.
def hasPrim(path):
    return any(os.path.isdir(os.path.join(path, prim)) for prim in ('p', 'He', 'O', 'Fe'))

def getYear(path):
    availableYears = [folder for folder in os.listdir(path)
                      if folder.startswith("IC86.") and any(hasPrim(updatePath(updatePath(path, folder), m))
                                                            for m in os.listdir(updatePath(path, folder)) if not "GCD" in m)]
    picked = pickMulti("years", availableYears)
    # tolerate years typed without the IC86. prefix
    return([y if "IC86." in y else "IC86." + y for y in picked])

#Gets the interaction model(s)
def getInteractionModel(path):
    availableModels = [model for model in os.listdir(path)
                       if not("GCD" in model) and hasPrim(updatePath(path, model))]
    return(pickMulti("interaction models", availableModels))

#Gets the composition
def getComposition(path):
    availableComps = [comp for comp in os.listdir(path)
                      if comp in ("p", "He", "O", "Fe") and os.path.isdir(updatePath(path, comp))]

    #The oldstructure folder doesn't have any compositions like this, so this can be ignored if that path was chosen
    if(len(availableComps) == 0):
        return([])

    picked = pickMulti("compositions", availableComps,
                       aliases={"p": ["proton"], "He": ["helium"], "O": ["oxygen"], "Fe": ["iron"]})
    # common long names map to the folder names
    alias = {"proton": "p", "helium": "He", "oxygen": "O", "iron": "Fe"}
    return([alias.get(pv.lower(), pv) for pv in picked])

#Gets the simulation set(s). Each simset may nest a version level; versions
#are chosen per simset (multi-select works there too). Returns [path, name] pairs.
def getSimSet(path):
    allSims = sorted(os.listdir(path))
    picked = []

    # 'a' at any version prompt means: stop asking, take every version of every
    # remaining simset too (one 'a' covers the whole level instead of N prompts).
    sims = pickMulti("simulation sets", allSims)
    for wantedSim in sims:
        newPath = updatePath(path, wantedSim)
        newPathContents = sorted(os.listdir(newPath))
        if(".i3" in newPathContents[0]):
            picked.append([newPath, wantedSim])
            continue
        for ver in pickMulti(f"versions of {wantedSim}", newPathContents):
            picked.append([updatePath(newPath, ver), wantedSim])
    return picked
            

#Gets the path to the GCD file
def getGCD(path, sim):
    allGCD = os.listdir(path)

    #If the simulation set needs any parsing, it's done here
    if("_" in sim):
        sim = sim[0:sim.find("_")]

    #Gets the GCD file
    for GCD in allGCD:
        if(sim in GCD):
            return(updatePath(path, GCD))

    # Newer productions (IC86.2018+) have a single year-wide GeoCalib file instead
    # of per-simset Level3_<simset>_GCD files; use it when it is the only choice.
    geo = [g for g in allGCD if g.endswith(".i3.gz")]
    if(len(geo) == 1):
        return(updatePath(path, geo[0]))

    #Needs better error handling, but this should NEVER occur
    return("error uh oh LOL")

def defineOutput():
    folderName = input("Extraction folder? (Will be located in /data/user/<yourlogin>/sim/<yourfolder>) ")
    out_dir = Path(f"/data/user/{os.getlogin()}/sim/{folderName}")
    while(out_dir.exists()):
        print("ALERT! Folder already exists, data has a high chance of being overwritten!")
        new_folder = input("Enter new folder name, or leave blank to stick with current path.")
        if(len(new_folder) == 0):
            break
        else:
            out_dir = Path(f"/data/user/{os.getlogin()}/sim/{new_folder}")
    return out_dir
    
def main(args):
    outPath = defineOutput()
    
    if(args.test == True):
        print("Testing mode enabled, running on the default path of .../IC86.2012/oldstructure/12360/")
        
        data_path = os.path.join('/', 'data', 'ana', 'CosmicRay', 'IceTop_level3', 'sim', 'IC86.2012','oldstructure', '12360')
        GCD_path = os.path.join('/', 'data', 'ana', 'CosmicRay', 'IceTop_level3', 'sim', 'IC86.2012','GCD', 'Level3_12360_GCD.i3.gz')

        #Run sim-maker.py
        subprocess.run([f"{os.getcwd()}/./sim-maker.py", f"-g {GCD_path}", f"-i {data_path}", f"-o {outPath}", "--test"])
    else:
        data_path = os.path.join('/','data', 'ana', 'CosmicRay', 'IceTop_level3', 'sim')
        compFlags = {"p": "proton", "He": "helium", "O": "oxygen", "Fe": "iron"}

        # Every step accepts numbers, names, several picks, or 'a'; 'b' backs out
        # and redoes the whole selection pass (safe: nothing runs until you
        # confirm the plan). sim-maker numbers batches per out dir and the
        # converter globs by simset ID, so all runs merge into one dataset.
        while(True):
            plan = []
            try:
                for year in getYear(data_path):
                    year_path = updatePath(data_path, year)
                    gcd_base = updatePath(year_path, "GCD")

                    for model in getInteractionModel(year_path):
                        model_path = updatePath(year_path, model)

                        comps = getComposition(model_path)
                        if(not comps):
                            print(f"oldstructure/gamma-only year detected at {model_path}; nothing extractable, skipping.")
                            continue

                        for comp in comps:
                            for simInfo in getSimSet(updatePath(model_path, comp)):
                                data_sel, sim_name = simInfo
                                GCD_path = getGCD(gcd_base, sim_name)
                                # ponytail: -n 5000 keeps condor job count sane
                                # (~70 jobs/year-comp instead of ~340); extractor
                                # RAM scales with batch events, still well under 8 GB.
                                cmd = [f"/home/{os.getlogin()}/icetop-cnn/data_extraction/sim-maker.py",
                                       "-g", GCD_path, "-i", data_sel, "-o", str(outPath), "-n", "5000"]
                                if(comp in compFlags):
                                    cmd += ["-c", compFlags[comp]]
                                plan.append(cmd)
            except Back:
                print("Backed out - start over.")
                continue
            except AllRest:
                pass  # _autoall flag set; loop replans with every pick auto-answered


            if(not plan):
                print("Nothing selected.")
                continue

            # rough condor job estimate: -n 5000 -> one job per ~5000 files.
            # one find over all -i roots (ceph walk, takes a moment on big trees)
            _roots = " ".join(f"'{cmd[cmd.index('-i') + 1]}'" for cmd in plan)
            _n_files = int(subprocess.run(
                f"find {_roots} -type f 2>/dev/null | wc -l",
                shell=True, capture_output=True, text=True).stdout.strip() or 0)
            _jobs = max(1, -(-_n_files // 5000))
            print(f"\n{len(plan)} extractions planned (~{_jobs} condor jobs from ~{_n_files} files):")
            for cmd in plan:
                print(" ", " ".join(cmd))
            if(args.dryrun or input("Run them? [y/N] ").strip().lower() == "y"):
                for cmd in plan:
                    if(not args.dryrun):
                        subprocess.run(cmd)
            break

    #Now run sim-converter.py if that argument was passed
    if(args.convert == True):
        print("Now converting data:")
        subprocess.run([f"/home/{os.getlogin()}/icetop-cnn/data_extraction/sim-converter.py", f"-s {outPath}"])
    
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description='sim wrapper')
    p.add_argument('--test', dest='test', default=False, action='store_true',
                  help="Run in testing mode")
    
    p.add_argument('-c', '--convert', dest="convert", default=False, action='store_true',
                  help="Run sim-converter.py after extraction")
    p.add_argument('-d', '--dryrun', dest="dryrun", default=False, action='store_true',
                  help="Print the sim-maker commands without running them")
    
    main(p.parse_args())
