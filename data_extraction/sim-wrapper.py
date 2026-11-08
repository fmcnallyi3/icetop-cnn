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

#getYear function, gets a selection of all the valid years and gets the user input on the wanted year
#Only list folders that actually hold cosmic-ray primaries; avoids dead ends
#like 2016 (GCD only), gamma-only years, oldstructure and temp dirs.
def hasPrim(path):
    return any(os.path.isdir(os.path.join(path, prim)) for prim in ('p', 'He', 'O', 'Fe'))

def getYear(path):
    allData = os.listdir(path)

    #Tells the user what years are available
    print("Here are the available years:")
    availableYears = []
    for folder in allData:
        if(folder.startswith("IC86.") and any(hasPrim(updatePath(updatePath(path, folder), m))
                                              for m in os.listdir(updatePath(path, folder)) if not "GCD" in m)):
            availableYears.append(folder)
            print(folder)

    #Checks the user's input to get a valid year
    wantedYear = ""
    while(True):
        wantedYear = input("Please enter the year that you want to get data from: ")
        if not("IC86." in wantedYear):
            print("IC86. has been added to your input! If prompted to input selection again, please include IC86. at the front of your response.")
            wantedYear = "IC86." + wantedYear
        if(wantedYear in availableYears):
              break
        else:
            print("Invalid selection was made.")
    print()
    return(wantedYear)

#Gets the interaction model
def getInteractionModel(path):
    allModels = os.listdir(path)

    #Tells the user what interaction models are available
    availableModels = []
    for model in allModels:
        if not("GCD" in model) and hasPrim(updatePath(path, model)):
            print(model)
            availableModels.append(model)

    #Checks the user's input to get a valid model
    wantedModel = ""
    while(True):
        wantedModel = input("Please enter the model you want to get data from: ")
        if(wantedModel in availableModels):
            break
        else:
            print("Invalid selection was made.")
            
    print()
    return(wantedModel)

#Gets the composition
def getComposition(path):
    allCompositions = os.listdir(path)

    availableComps = []
    for comp in allCompositions:
        if comp in ("p", "He", "O", "Fe") and os.path.isdir(updatePath(path, comp)):
            print(comp)
            availableComps.append(comp)

    #The oldstructure folder doesn't have any compositions like this, so this can be ignored if that path was chosen
    if(len(availableComps) == 0):
        return("no")
        
    wantedComp = ""
    while(True):
        wantedComp = input("Please enter the composition you want to get data from (or 'all' for every primary listed): ")
        if(wantedComp in availableComps):
            break
        # 'all' expands to every primary present; main() makes one selection per primary
        if(wantedComp.lower() == "all" and availableComps):
            return("all")
        print("Invalid selection was made.")
    return(wantedComp)

#Gets the simulation set
def getSimSet(path):
    allSims = os.listdir(path)

    #All simulation sets will be valid, so no parsing array needs to be made
    for sim in allSims:
        print(sim)

    wantedSim = ""
    while(True):
        wantedSim = input("Please enter the simulation set you want to get data from: ")
        if(wantedSim in allSims):
            break
        else:
            print("Invalid selection was made.")
    newPath = updatePath(path, wantedSim)
    newPathContents = os.listdir(newPath)
    if not(".i3" in newPathContents[0]):
        for ver in newPathContents:
            print(ver)
            
        wantedVer = ""
        while(True):
            wantedVer = input("Please enter the version you want to select: ")
            if(wantedVer in newPathContents):
                break
            else:
                print("Invalid selection was made.")
        newPath = updatePath(newPath, wantedVer)
    return [newPath, wantedSim]
            

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

        # One dataset can hold many selections (every primary of a year, several
        # years, ...). sim-maker numbers batches per out dir and the converter
        # globs by simset ID, so repeated runs into the same folder merge cleanly.
        while(True):
            #Year
            year = getYear(data_path)
            sel_path = updatePath(data_path, year)
            gcd_base = updatePath(sel_path, "GCD")

            #Model
            model = getInteractionModel(sel_path)
            sel_path = updatePath(sel_path, model)

            #Composition, needs the extra protection if the oldstructure path was chosen
            composition = getComposition(sel_path)
            comps = ["p", "He", "O", "Fe"] if composition == "all" else [composition]
            if(composition == "no"):
                print("oldstructure detected?!?!")
                comps = []

            for comp in comps:
                comp_path = updatePath(sel_path, comp)

                #Finalizes the paths
                simInfo = getSimSet(comp_path)
                data_sel = simInfo[0]

                GCD_path = getGCD(gcd_base, simInfo[1])

                #Run sim-maker.py (pass the composition so output files get the right simset ID)
                cmd = [f"/home/{os.getlogin()}/icetop-cnn/data_extraction/sim-maker.py", f"-g {GCD_path}", f"-i {data_sel}", f"-o {outPath}"]
                if(comp in compFlags):
                    cmd.append(f"-c {compFlags[comp]}")
                print("Running:", " ".join(cmd))
                if(not args.dryrun):
                    subprocess.run(cmd)

            again = input("Add another selection to this dataset? [y/N] ")
            if(again.strip().lower() != "y"):
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
