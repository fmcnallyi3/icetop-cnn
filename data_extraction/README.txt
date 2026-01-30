SIMULATION DATA EXTRACTION

How to use:

1. Connect to the submitter node by running the command 'ssh submitter'.
  
2. Ensure 'pysubmit.py' is present in the 'npx4' folder in your home directory.
   2.a) If not possible, change the import statement at the beginning of 'sim-maker.py'.

3. Make sure that the following files are in the same directory (assumes home directory by default):
  - 'sim-maker.py'
  - 'sim-extractor.py'
  3.a) If not possible, change the variable 'exe' file path at the beginning of 'sim-maker.py'.

4. Run 'sim-maker.py' with any combination of the following arguments:
   * -c  --composition  ('proton', 'helium', 'oxygen', 'iron', 'all')  Specifies compositon(s) to extract                        default: 'proton'
   * -n  --n            (int)                                          Specifies number of files in a batch                      default: 1000
   * -o  --out          (str)                                          Output directory                                          default: '/data/user/LOGIN_NAME/sim'
   *     --mc                                                          Extract information from Monte Carlo generated sim files  default: False
   *     --test                                                        Runs the extraction off of the condor cluster             default: False
   NOTE: This will take some time. Check the progress of your submissions with the command 'condor_q'.

5. Make sure that the following files are in the same directory:
   - 'sim-converter.py'
   - 'sim_utils.py'
   5.a) If not possible, change the import statement at the beginning of 'sim-converter.py'.

6. Run 'sim-converter.py' with any combination of the following arguments:
   * -i  --infill                                 Option to include the infill array of detectors                                 default: False
   * -o  --output     ('array', 'param', 'both')  Specifies whether to convert detector data ('array') or primary info ('param')  default: 'both'
   *     --overwrite                              Option to overwrite existing preprocessed files                                 default: False
