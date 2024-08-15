#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:44:17 2024

@author: arest
"""

import argparse,os,sys,re,copy,shutil
import yaml


class syndiff_baseclass:

    def __init__(self):
        self.verbose=0

        self.reset()

    def reset(self):
        # self.params will be populated with the arguments
        self.params = {}

    def define_optional_parameters(self,parser=None,usage=None,conflict_handler='resolve'):
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
        parser.add_argument('-c','--configfile', type=str, default=cfgfilename, help='config file. default is set to $SYNDIFF_DEFAULT_CFG_FILE. Use -vvv to see the full list of all set parameters.')
        
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
        if args.configfile is not None:
            #cfgparams = yaml.load_file(args.configfile)
            if not os.path.isfile(args.configfile):
                raise RuntimeError('config file %s does not exist!' % (args.configfile))
            print(f'Loading config file {args.configfile}')
            cfgparams = yaml.load(open(args.configfile,'r'), Loader=yaml.FullLoader)
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
    
    def outrootdir(self):
        outdir = f'{os.path.abspath(self.params["output_rootdir"])}'
        if self.params['outsubdir'] is not None:
            outdir += os.path.join(outdir,self.params['outsubdir'])
        return(outdir)

if __name__ == '__main__':

    test = syndiff_baseclass()
    parser = test.define_optional_parameters()
    args = parser.parse_args()

    # the arguments are saved in query.params
    test.get_arguments(args)
    
    print(f'\n# Outrootdir: {test.outrootdir()}')
