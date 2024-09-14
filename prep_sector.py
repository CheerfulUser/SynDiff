#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:33:14 2024

@author: arest
"""

import argparse,re,sys,os,random
from pdastro import AnotB,pdastroclass
from syndiff_baseclass import syndiff_baseclass,executecommand,makepath4file
import pandas as pd

allowed_steps = ['setTESSref','combinePS1']


class prep_sectorclass(syndiff_baseclass):

    def __init__(self):
       syndiff_baseclass.__init__(self)
       self.steps = None
       self.stepcmds = {}

       # assign for each step the function to execute
       self.stepcmds['setTESSref']=self.execute_setTESSref
       self.stepcmds['combinePS1']=self.execute_combinePS1

    def define_arguments(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)
        parser.add_argument('steps', type=str, nargs='*', help='list of steps to do')
        return(parser)
    
    def optional_arguments_setTESSref(self,parser):
        group = parser.add_argument_group('setTESSref')

        group.add_argument('--setTESSref_test1',default=None,help="test optional params 1 ")
        group.add_argument('--setTESSref_test2',default=None,help="")
        
        return(group)

    def optional_arguments_combinePS1(self,parser):
        group = parser.add_argument_group('combinePS1')

        group.add_argument('--combinePS1_test1')
        group.add_argument('--combinePS1_test2')
        
        return(group)

    def optional_arguments_setTESSref_old(self,parser):
        parser.add_argument('--setTESSref_test1',default=4,help="test optional params 1 ")
        parser.add_argument('--setTESSref_test2',default=None,help="")

        return(parser)

    def optional_arguments_combinePS1_old(self,parser):
        parser.add_argument('--combinePS1_test1',default=4,help="test optional params 1 ")
        parser.add_argument('--combinePS1_test2',default="hello",help="whatever")

        return(parser)

    def define_optional_arguments(self,parser=None,usage=None,conflict_handler='resolve'):
        parser = syndiff_baseclass.define_optional_arguments(self,parser=parser,usage=usage,conflict_handler=conflict_handler)
        #subparsers = parser.add_subparsers(help='commands')
        
        #setTESSref_subparser = subparsers.add_parser('setTESSref', help='set the TESS reference file')
        #self.optional_arguments_setTESSref(setTESSref_subparser)

        #combinePS1_subparser = subparsers.add_parser('combinePS1', help='combine the PS1 filter images and convolve them with the TESS PSF')
        #self.optional_arguments_combinePS1(combinePS1_subparser)

        group_setTESSref = self.optional_arguments_setTESSref(parser)
        group_combinePS1 = self.optional_arguments_combinePS1(parser)

        return(parser)
    
    def get_steps(self,steps=None):
        if (steps is None) or (steps==[]):
            self.steps = allowed_steps
        else:
            self.steps = steps
            
        badsteps = AnotB(self.steps,allowed_steps)
        if len(badsteps)>0:
            raise RuntimeError(f'Unknown steps: {",".join(badsteps)}')

        for step in steps:
            if step not in self.stepcmds:
                raise RuntimeError(f'step {step} has not yet have a function assigned!')
                
            
        #if self.verbose>1: print(f'### steps: {self.steps}')
        
    def get_logfilenames4step(self,step):
        cmd_logfilename = step_resultfilename = None
        
        if self.params['save_logfiles']:
            cmd_logfilename = self.get_logfilename(step,resultflag=False)
            makepath4file(cmd_logfilename)

        step_resultfilename = self.get_logfilename(step,resultflag=True)

        return(cmd_logfilename,step_resultfilename)
    
    def save_results(self,step_resultfilename,step,cmd,errorflag,execution_start_date,execution_end_date):
        results = pdastroclass()
        if os.path.isfile(step_resultfilename):
            if self.verbose: print(f'Loading {step_resultfilename}')
            results.t = pd.read_csv(step_resultfilename)
            
        if errorflag==0:
            errorstring='-'
        elif errorflag==1:
            errorstring='ERR_OUT'
        elif errorflag==2:
            errorstring='ERR_FINISHED'
        else:
            raise RuntimeError(f'Unknown errorflag {erroflag}!!!')
        
        delta_time = execution_end_date - execution_start_date

        ix = results.newrow({'step':step,
                             'error':errorstring,
                             'errorflag':errorflag,
                             'duration_min':delta_time.sec/60.0,
                             'date_start':execution_start_date.to_value('fits'),
                             'date_end':execution_end_date.to_value('fits'),
                             'cmd':cmd})
        
        results.default_formatters['duration_min']='{:.3f}'.format
                             
        results.write()
        if self.verbose: print(f'Saving {step_resultfilename}')
        results.t.to_csv(step_resultfilename,index=False)
        
    def execute_step(self,step,cmd):
            
        # get logfilenames. Depends on config file!
        cmd_logfilename,step_resultfilename = self.get_logfilenames4step(step)
            
        #print('ggg',self.params['timezone'])
        #sys.exit(0)
        
        errorflag,  execution_start_date, execution_end_date, output = executecommand(cmd,f'STEP {step} FINISHED',cmd_logfilename=cmd_logfilename,
                                                                                      overwrite=self.params['overwrite_logfiles'],
                                                                                      verbose=self.verbose)
        self.save_results(step_resultfilename,step,cmd,errorflag,execution_start_date,execution_end_date)

        if errorflag and self.params['raise_RuntimeError']:
            raise RuntimeError(f'ERROR in step {step} executing cmd: {cmd}')
            

        return(errorflag)

    def execute_setTESSref(self,step):
        # construct the command
        cmd = 'setTESSref.py'
        if self.params['setTESSref_test1'] is not None:
            cmd += f' --test1 {self.params["setTESSref_test1"]}'
       
        errorflag = self.execute_step(step,cmd)
        
        return(errorflag)   
        
    
        
    def execute_combinePS1(self,step):
        print('HELLO combinePS1')
        
    #def execute_cmds(self,steps=None):
    #    print('HELLO')
            
    def execute_steps(self,steps=None):
        # if no steps are passed, execute the steps in self.steps
        if steps is None:
            steps = self.steps
            
        for step in steps:
            print(f'###\n### Preparing to execute step {step}!\n###')
            self.stepcmds[step](step)
    
if __name__ == '__main__':
    """
    from astropy.time import Time
    import pytz
    from datetime import datetime
    
    timezone='US/Eastern'
    execution_start_date = datetime.now()
    if timezone is not None:
        print('jjj')
        execution_start_date=Time(pytz.utc.localize(execution_start_date).astimezone(pytz.timezone(timezone)))
    execution_end_date = datetime.now()
    if timezone is not None:
        execution_end_date=Time(pytz.utc.localize(execution_end_date).astimezone(pytz.timezone(timezone)))
    delta_time = execution_end_date - execution_start_date
    print(delta_time,'hhhhh',execution_start_date)
    print(delta_time.sec,'hhhhh')
    #sys.exit(0)
    
    #print('Timezones')
    #for timeZone in pytz.all_timezones:
    #    print(timeZone)
    
    t1 = Time.now()
    t3=pytz.utc.localize(t1.datetime).astimezone(pytz.timezone('US/Eastern'))
    t2 = Time("2024-09-14T15:59:00.200")
    dt = t2-t1
    print(t1,t2,t3)
    print(dt.sec/60)
    #sys.exit(0)
    
    #import pandas as pd
    #test=pd.DataFrame()
    #test = pd.read_csv('/Users/arest/data/syndiff/output/Users/arest/data/syndiff/output/v1/020/01/logs/setTESSref.result.txt')
    #print(test.columns)
    #print(test)
    #sys.exit(0)
    """
        
    prep = prep_sectorclass()

    # get the arguments
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser = prep.define_arguments(parser=parser)
    parser = prep.define_optional_arguments(parser=parser)
    args = parser.parse_args()
    
    # the arguments are saved in prep.params
    prep.get_arguments(args)
    prep.set_maindirs()
    prep.get_steps(args.steps)
    
    prep.execute_steps()
    #prep.execute_cmds()
    
