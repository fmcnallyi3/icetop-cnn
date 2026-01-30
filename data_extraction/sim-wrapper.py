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
    -Fix the general problem with this whole menagerie not working on Condor (Important!!!!)
        -Seems like something to do with pathing issues?
    -Add in the whole pass-parameter methods to sim-maker.py and sim-converter.py (Definitely important)
    -Add more comments
    -Make the code cleaner
"""

def updatePath(currPath, newArg):
    return(os.path.join(currPath, newArg))

#getYear function, gets a selection of all the valid years and gets the user input on the wanted year
def getYear(path):
    allData = os.listdir(path)

    #Tells the user what years are available
    print("Here are the available years:")
    availableYears = []
    for folder in allData:
        if(folder[:4] == "IC86"):
            if(folder[-1] == "2" or folder[-1] == "5"): #Hardcoded to avoid getting any weird edge cases
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
        if not("GCD" in model):
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
        if (len(comp) <= 2): #This causes gamma folder to be ignored, irrelevant?
            print(comp)
            availableComps.append(comp)

    #The oldstructure folder doesn't have any compositions like this, so this can be ignored if that path was chosen
    if(len(availableComps) == 0):
        return("no")
        
    wantedComp = ""
    while(True):
        wantedComp = input("Please enter the composition you want to get data from: ")
        if(wantedComp in availableComps):
            break
        else:
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
    
        #Year
        year = getYear(data_path)
        data_path = updatePath(data_path, year)
        GCD_path = updatePath(data_path, "GCD")
    
        #Model
        model = getInteractionModel(data_path)
        data_path = updatePath(data_path, model)
    
        #Composition, needs the extra protection if the oldstructure path was chosen
        composition = getComposition(data_path)
        if(composition == "no"):
            print("oldstructure detected?!?!")
        else:
            data_path = updatePath(data_path, composition)
    
        #Finalizes the paths 
        simInfo = getSimSet(data_path)
        data_path = simInfo[0]
         
        #print(data_path)
        GCD_path = getGCD(GCD_path, simInfo[1])
    
        #Run sim-maker.py
        subprocess.run([f"{os.getcwd()}/./sim-maker.py", f"-g {GCD_path}", f"-i {data_path}", f"-o {outPath}", "--test"])

    #Now run sim-converter.py if that argument was passed
    if(args.convert == True):
        print("Now converting data:")
        subprocess.run([f"{os.getcwd()}/./sim-converter.py", f"-s {outPath}"])
    
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description='sim wrapper')
    p.add_argument('--test', dest='test', default=False, action='store_true',
                  help="Run in testing mode")
    
    p.add_argument('-c', '--convert', dest="convert", default=False, action='store_true',
                  help="Run sim-converter.py after extraction")
    
    main(p.parse_args())