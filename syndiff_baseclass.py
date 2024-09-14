#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:44:17 2024

@author: arest
"""

import argparse,os,sys,re,copy,shutil
#from datetime import datetime, timedelta
from subprocess import Popen, PIPE, STDOUT

from astropy.time import Time
from datetime import datetime

import pytz

import yaml

def rmfile(filename,raiseError=1,gzip=False):
    " if file exists, remove it "
    if os.path.lexists(filename):
        os.remove(filename)
        if os.path.isfile(filename):
            if raiseError == 1:
                raise RuntimeError('ERROR: Cannot remove %s' % filename)
            else:
                return(1)
    return(0)


def executecommand(cmd,execution_finished_word=None,cmd_logfilename=None,
                   overwrite=True,raiseRuntimeError=False,verbose=1):

    execution_start_date = Time(datetime.now())
    startlines=[f'\n################## {execution_start_date.to_value("fits")}',
                f'###### executing:\n{cmd}',
                '##################\n']
    
    if verbose: print('\n'.join(startlines))

    # execute the command and capture the output
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, 
              close_fds=True,text=True,universal_newlines=True)
    cmd_in,cmd_out = p.stdin,p.stdout
    output = cmd_out.readlines()

    execution_end_date = Time(datetime.now())

    
    # print the output if verbose>=2
    if verbose>=2: print(''.join(output))

    # Check for errors!
    errorflag=0
    if execution_finished_word is not None:
        if (len(output)<1):
            errorflag=1
        else:
            line = output[-1]
            #if sys.version_info[0] >= 3: line = line.decode('utf-8')
            if re.search(execution_finished_word,line) is None:
                print(f'ERROR: cannot find "{execution_finished_word}" in the last line of the output of cmd {cmd}!')
                errorflag = 2
    
    # result strings: Error or finished?
    delta_time = execution_end_date - execution_start_date
    endlines = [f'\n!!!!!!!!!!!!!!!!!!! {delta_time.sec/60.0:.3f} min execution time: {execution_start_date.to_value("fits")} to {execution_end_date.to_value("fits")}']
    if errorflag:
        endlines.append( f'!!! ERROR executing {cmd}')
    else:
        endlines.append( f'!!! FINISHED executing {cmd}')
    endlines.append('!!!!!!!!!!!!!!!!!!!')

    # print the result strings if verbose
    if verbose: print('\n'.join(endlines))
    
    # save to log file
    if (cmd_logfilename is not None):
        if verbose: print(f'Saving output to {cmd_logfilename}')
        if os.path.isfile(cmd_logfilename):
            if overwrite:
                rmfile(cmd_logfilename)
                buff = open(cmd_logfilename, 'w')
            else:
                buff = open(cmd_logfilename, 'a')
        else:
            buff = open(cmd_logfilename, 'w')
            
        buff.write('\n'.join(startlines)+'\n')
        buff.write(''.join(output))
        buff.write('\n'.join(endlines))

        buff.close()
        
    return errorflag, execution_start_date, execution_end_date, output

def save2file(filename,lines,verbose=0,append=False):
    if type(lines) is str:#types.StringType:
        lines = [lines,]
    if os.path.isfile(filename):
        if append:
            buff = open(filename, 'a')
        else:
            rmfile(filename)
            buff = open(filename, 'w')
    else:
        buff = open(filename, 'w')
    r=re.compile('\n$')
    for line in lines:
        if not r.search(line):
            buff.write(line+'\n')
            if verbose: print((line+'\n'))
        else:
            buff.write(line)
            if verbose: print(line)
    buff.close()

def makepath(path,raiseError=1):
    if path == '':
        return(0)
    if not os.path.isdir(path):
        os.makedirs(path)
        if not os.path.isdir(path):
            if raiseError == 1:
                raise RuntimeError('ERROR: Cannot create directory %s' % path)
            else:
                return(1)
    return(0)

def makepath4file(filename,raiseError=1):
    path = os.path.dirname(filename)
    if not os.path.isdir(path):
        return(makepath(path,raiseError=raiseError))
    else:
        return(0)


class syndiff_baseclass:

    def __init__(self):
        self.verbose=0

        self.reset()

    def reset(self):
        # self.params will be populated with the arguments
        self.params = {}
        
        # main directories
        self.outdir = None
        self.prepdir = None
        self.diffimdir = None

    def define_optional_arguments(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)

        # default for config file, if available
        if 'SYNDIFF_DEFAULT_CFG_FILE' in os.environ and os.environ['SYNDIFF_DEFAULT_CFG_FILE'] != '':
            cfgfilename = os.environ['SYNDIFF_DEFAULT_CFG_FILE']
        else:
            cfgfilename = None

        parser.add_argument('-s','--sector', type=int, default=None, help='TESS Sector.')
        parser.add_argument('--ccd', type=int, default=None, choices=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],help='TESS CCD.')
        parser.add_argument('-o','--outsubdir', type=str, default=None,help='subdir that gets added to the output directory')

        parser.add_argument('-v','--verbose', default=0, action='count')
        parser.add_argument('-c','--primary_configfile', type=str, default=cfgfilename, help=f'primary config file. default is set to $SYNDIFF_DEFAULT_CFG_FILE={os.environ["SYNDIFF_DEFAULT_CFG_FILE"]}. Use -vvv to see the full list of all set parameters.')
        
        return(parser)

    def get_arguments(self, args, configfile = None):
        '''

        Parameters
        ----------
        args : list
            pass the command line arguments to self.params.
        configfile : string, optional
            Config filename. The default is None. If None, then
            $SYNDIFF_CFGFILE is used if exists.

        Returns
        -------
        None.

        '''

        def subenvvarplaceholder(paramsdict):
            """ Loop through all string parameters and substitute environment variables. environment variables have the form $XYZ """
            envvarpattern=re.compile('\$(\w+)')

            for param in paramsdict:
                print(param)
                if isinstance(paramsdict[param], str):
                    envvarnames=envvarpattern.findall(paramsdict[param])
                    if envvarnames:
                        for name in envvarnames:
                            if not (name in os.environ):
                                raise RuntimeError("environment variable %s used in config file, but not set!" % name)
                            else:
                                envval=os.environ[name]
                                subpattern='\$%s' % (name)
                                paramsdict[param] = re.sub(subpattern,envval,paramsdict[param])
                elif isinstance(paramsdict[param], dict):
                #elif type(dict[param]) is types.DictType:
                    # recursive: sub environment variables down the dictiionary
                    subenvvarplaceholder(paramsdict[param])
            return(0)


        # get the parameters from the config file
        if args.primary_configfile is not None:
            #cfgparams = yaml.load_file(args.configfile)
            if not os.path.isfile(args.primary_configfile):
                raise RuntimeError('config file %s does not exist!' % (args.primary_configfile))
            print(f'Loading config file {args.primary_configfile}')
            cfgparams = yaml.load(open(args.primary_configfile,'r'), Loader=yaml.FullLoader)
            self.params.update(cfgparams)

            subenvvarplaceholder(self.params)

            if args.verbose>2:
                print('\n### CONFIG FILE PARAMETERS:')
                for p in cfgparams:
                    print('config file args: setting %s to' % (p),cfgparams[p])

        # Go through optional parameters.
        # 'None' does not overwrite previously set parameters (either default or config file)
        if args.verbose>2:
            print('\n### OPTIONAL COMMAND LINE PARAMETERS (shown because verbose>2):')

        argsdict = vars(args)
        for arg in argsdict:

            # skip config file
            if arg=='configfile': continue

            if argsdict[arg] is not None:
                if args.verbose>2:
                    print('optional args: setting %s to %s' % (arg,argsdict[arg]))
                self.params[arg]=argsdict[arg]
            else:
                if arg not in  self.params:
                    self.params[arg]=None

        if 'verbose' in self.params:
            self.verbose = self.params['verbose']

        if self.verbose>2:
            print('\n### FINAL PARAMETERS (shown because verbose>2):')
            for p in self.params:
                print(p,self.params[p])

        return(0)
    
    def set_outdir(self,outrootdir=None,outsubdir=None):
        if outrootdir is None:
            outdir = f'{os.path.abspath(self.params["outrootdir"])}'
        else:
            outdir = outrootdir
        if outdir is None:
            raise RuntimeError('outdir is not defined!')
        
        if outsubdir is None:
            outsubdir = self.params['outsubdir']
        if outsubdir is not None:
            outdir += os.path.join(outdir,outsubdir)
            
        outdir = f'{outdir}/{self.params["sector"]:03d}/{self.params["ccd"]:02d}'
        self.outdir = outdir
        return(outdir)

    def set_prepdir(self):
        self.prepdir = f'{self.outdir}/prepdata'
        return(self.prepdir)

    def set_diffimdir(self):
        self.diffimdir = f'{self.outdir}/diffim'
        return(self.diffimdir)

    def set_maindirs(self,outrootdir=None,outsubdir=None):
        self.set_outdir(outrootdir=outrootdir,outsubdir=outsubdir)
        self.set_prepdir()
        self.set_diffimdir()
        if self.verbose:
            print(f'\n# outdir: {self.outdir}')
            print(f'# prepdir: {self.prepdir}')
            print(f'# diffimdir: {self.diffimdir}')

    def get_logfiledir(self):
        logfiledir = f'{self.outdir}/logs'
        return(logfiledir)

    def get_logfilename(self,step,imagename=None,resultflag=False):
        logfilename = f'{self.get_logfiledir()}/{step}'
        if imagename is not None:
            logfilename += f'/{os.path.basename(imagename)}'
        if resultflag:
            logfilename += '.result.txt'
        else:
            logfilename += '.log.txt'
        return(logfilename)
        
        

if __name__ == '__main__':

    test = syndiff_baseclass()
    parser = test.define_optional_arguments()
    args = parser.parse_args()

    # the arguments are saved in query.params
    test.get_arguments(args)
    
    test.set_maindirs()
    print(f'\n# Outrootdir: {test.outrootdir()}')
