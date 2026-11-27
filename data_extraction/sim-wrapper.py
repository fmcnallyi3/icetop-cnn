#!/usr/bin/env python3
import sys
import os
from glob import glob
import getpass
import json
import argparse
import subprocess
import math
import random
import re
from pathlib import Path
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    out_dir = Path(f"/data/user/{getpass.getuser()}/icetop-cnn/sim/{folderName}")
    while(out_dir.exists()):
        print("ALERT! Folder already exists, data has a high chance of being overwritten!")
        new_folder = input("Enter new folder name, or leave blank to stick with current path.")
        if(len(new_folder) == 0):
            break
        else:
            out_dir = Path(f"/data/user/{getpass.getuser()}/icetop-cnn/sim/{new_folder}")
    return out_dir
    
# Converter sizing, calibrated 2026-11-26: one raw npy peaks at ~17.8 kB/event
# RSS (measured 19.3 GB for 1,086,241 events from a 1.97 GB file). Rounded to
# 20 kB for headroom. sim-maker caps a batch at -n 20000 i3 files, and a batch
# yields at least 55 events per file (1.09M events in 20000 files = 54.3), so
# peak RAM depends on the batch, not on dataset size.
CONVERT_KB_PER_EVENT = 20
EVENTS_PER_I3 = 55
BATCH_FILES = 20000

def convert_mem_gb(n_files):
    events = min(n_files or BATCH_FILES, BATCH_FILES) * EVENTS_PER_I3
    gb = events * CONVERT_KB_PER_EVENT / 1e6  # kB -> GB
    return max(12, math.ceil(gb / 4) * 4)

def converter_python():
    # Condor jobs get no login env; the converter needs the venv numpy, and the
    # venv is what every working converter job has used.
    venv = f"/data/user/{getpass.getuser()}/.venv/bin/python"
    return venv if os.path.exists(venv) else sys.executable

def run_plan(plan, args):
    # rough condor job estimate: one job per ~20000 files; ceph find takes a moment
    _roots = " ".join(f"'{cmd[cmd.index('-i') + 1]}'" for cmd in plan)
    _n_files = int(subprocess.run(
        f"find {_roots} -type f ! -type l 2>/dev/null | wc -l",
        shell=True, capture_output=True, text=True).stdout.strip() or 0)
    _jobs = max(1, -(-_n_files // BATCH_FILES))
    print(f"\n{len(plan)} extractions planned (~{_jobs} condor jobs from ~{_n_files} files):")
    for cmd in plan:
        print(" ", " ".join(cmd))
    job_ids = []
    if(args.dryrun):
        print(f"\nConverter would request {convert_mem_gb(_n_files)} GB, "
              f"submitted by a watcher once every extractor job exits.")
        return job_ids, _n_files
    if(input("Run them? [y/N] ").strip().lower() != "y"):
        return job_ids, _n_files
    for cmd in plan:
        # Capture so the per-job ids can be parsed; pysubmit prints one per batch.
        out = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(out.stdout)
        sys.stderr.write(out.stderr)
        job_ids += re.findall(r"npx4-\d{6}", out.stdout)
    return job_ids, _n_files

def submit_convert(outPath, job_ids, n_files):
    # A plain submit file cannot depend on another cluster (DAGMan is the
    # supported way, and sim-maker submits its own jobs), and queue order alone
    # let the converter race 21 h ahead of the extraction on 2026-11-19. So a
    # detached watcher condor_waits each extractor log, then submits the job.
    if(not job_ids):
        print("No extractor job ids captured; NOT submitting the converter.")
        print(f"Run it yourself once the queue drains: "
              f"{converter_python()} {os.path.dirname(os.path.abspath(__file__))}"
              f"/sim-converter.py -s {outPath}")
        return
    # sim-maker submits with outdir='./condor', so mirror that exact location.
    condor_dir = Path.cwd().resolve() / "condor"
    mem = convert_mem_gb(n_files)
    conv_id = f"npx4-{random.randint(0, 999999):06d}"
    exe = condor_dir / "npx4-execs" / f"{conv_id}.sh"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("#!/bin/bash\n"
                   f"{converter_python()} {os.path.dirname(os.path.abspath(__file__))}"
                   f"/sim-converter.py -s {outPath}\n")
    exe.chmod(0o755)
    sub = condor_dir / f"{conv_id}.sub"
    sub.write_text(f"""universe = vanilla
getenv = True
executable = {exe}
log = {condor_dir}/npx4-logs/{conv_id}.log
output = {condor_dir}/npx4-out/{conv_id}.out
error = {condor_dir}/npx4-error/{conv_id}.error
notification = never
request_memory = {mem} GB
queue
""")
    log = condor_dir / f"npx4-logs/{conv_id}.watch.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    watcher = condor_dir / f"convert-watch-{outPath.name}-{conv_id}.sh"
    watcher.write_text("#!/bin/bash\n" + "".join(
        f"condor_wait {condor_dir}/npx4-logs/{j}.log "
        f"|| {{ echo 'WAIT FAILED {j}' >> {log}; exit 1; }}\n" for j in job_ids)
        + f"condor_submit {sub} >> {log} 2>&1\n"
        + f"echo \"converter {conv_id} submitted $(date)\" >> {log}\n")
    watcher.chmod(0o755)
    with open(log, "a") as fh:
        subprocess.Popen(["setsid", "bash", str(watcher)],
                         stdout=fh, stderr=fh, start_new_session=True)
    print(f"Watcher armed on {len(job_ids)} extractor job(s); converter {conv_id} "
          f"({mem} GB) submits when they exit. Progress: {log}")


def main(args):
    outPath = defineOutput()
    
    if(args.he == True):
        # HE productions live outside the regular trees (no comp level): year/
        # simset dirs; comp + energy range come from the production json. GCDs
        # stay in the regular year GCD dirs, so pass -g per simset. Only sets
        # reaching past 10^8.0 GeV count as high energy; bad_* is quarantine.
        he_base = os.path.join('/', 'data', 'ana', 'CosmicRay', 'IceTop_level3', 'sim', 'NewL3_MC')
        info = json.load(open(os.path.join(os.path.dirname(he_base), 'dataset_info_v0.json')))
        primaries = {'PPlus': 'proton', 'He4Nucleus': 'helium', 'O16Nucleus': 'oxygen', 'Fe56Nucleus': 'iron'}
        plan = []
        for year in sorted(y for y in os.listdir(he_base) if y.startswith('IC86.')):
            year_path = updatePath(he_base, year)
            gcd_dir = updatePath(updatePath(os.path.dirname(he_base), year), 'GCD')
            for simset in sorted(os.listdir(year_path)):
                meta = info.get(simset, {})
                if(simset.startswith('bad_') or float(meta.get('ebin_last', 0)) <= 8.0):
                    continue
                gcd_file = updatePath(gcd_dir, f'Level3_{simset}_GCD.i3.gz')
                if(not os.path.exists(gcd_file)):
                    print(f'No GCD for {simset}; skipping.')
                    continue
                cmd = [f"/home/{getpass.getuser()}/icetop-cnn/data_extraction/sim-maker.py",
                       "-g", gcd_file, "-i", updatePath(year_path, simset),
                       "-o", str(outPath), "-n", "20000"]
                prim = primaries.get(meta.get('primary'))
                if(prim):
                    cmd += ["-c", prim]
                plan.append(cmd)
        if(not plan):
            print("No extractable high-energy sets found.")
            return
        job_ids, n_files = run_plan(plan, args)
        if(args.convert == True):
            submit_convert(outPath, job_ids, n_files)
        return

    job_ids, n_files = [], 0
    if(args.test == True):
        print("Testing mode enabled, running on the default path of .../IC86.2012/oldstructure/12360/")
        
        data_path = os.path.join('/', 'data', 'ana', 'CosmicRay', 'IceTop_level3', 'sim', 'IC86.2012','oldstructure', '12360')
        GCD_path = os.path.join('/', 'data', 'ana', 'CosmicRay', 'IceTop_level3', 'sim', 'IC86.2012','GCD', 'Level3_12360_GCD.i3.gz')

        #Run sim-maker.py
        subprocess.run([os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim-maker.py"),
                        "-g", GCD_path, "-i", data_path, "-o", str(outPath), "--test"])
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
                            # whole tree and resolves the per-simset GCD per batch,
                            # so ~5000 files per condor job actually happens
                            # (724 tiny jobs -> ~8 per year-comp pair).
                            cmd = [f"/home/{getpass.getuser()}/icetop-cnn/data_extraction/sim-maker.py",
                                   "-i", updatePath(model_path, comp), "-o", str(outPath), "-n", "20000"]
                            if(comp in compFlags):
                                cmd += ["-c", compFlags[comp]]
                            plan.append(cmd)
            except Back:
                print("Backed out - start over.")
                continue


            if(not plan):
                print("Nothing selected.")
                continue
            job_ids, n_files = run_plan(plan, args)
            break

    #Run sim-converter on the cluster once extraction finishes: a watcher waits
    #on every extractor job, then submits it. Queue order alone is not enough.
    if(args.convert == True):
        submit_convert(outPath, job_ids, n_files)
    
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description='sim wrapper')
    p.add_argument('--test', dest='test', default=False, action='store_true',
                  help="Run in testing mode")
    
    p.add_argument('-c', '--convert', dest="convert", default=False, action='store_true',
                  help="Run sim-converter.py after extraction")
    p.add_argument('-d', '--dryrun', dest="dryrun", default=False, action='store_true',
                  help="Print the sim-maker commands without running them")
    p.add_argument('-e', '--he', dest="he", default=False, action='store_true',
                  help="Extract high-energy sets from NewL3_MC (log10E/GeV up to 9.5)")
    
    main(p.parse_args())
