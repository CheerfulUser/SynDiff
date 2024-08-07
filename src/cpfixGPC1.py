#!/usr/bin/env python
import sys, os, re, math, optparse, types, copy, shutil, glob
import scipy
import numpy as np
sys.path.append(os.path.join(os.environ['PIPE_PYTHONSCRIPTS'],'tools'))
from tools import rmfiles,makepath4file,deg2sex,hex2int
from astropy import wcs
import astropy.io.fits as pyfits

class cpfixGPC1class:
    def __init__(self):

        # output format, hardcoded for right now, could be changed to optional
        self.bitpix = 'int16'
        self.bscale = 1.0
        self.bzero = 32768.0

        self.noisebitpix = 'int16'
        self.noisebscale = 0.1
        self.noisebzero = 3276.80

        self.IPP_DETECTOR = 1
        self.IPP_FLAT =  2
        self.IPP_DARK   =4
        self.IPP_BLANK  =8
        self.IPP_CTE   = 16
        self.IPP_SAT  =  32
        self.IPP_LOW  =  64
        self.IPP_SUSPECT = 128
        self.IPP_BURNTOOL = 128
        self.IPP_CR    =  256
        self.IPP_SPIKE   = 512
        self.IPP_GHOST  =  1024
        self.IPP_STREAK  = 2048
        self.IPP_STARCORE = 4096
        self.IPP_CONVBAD = 8192
        self.IPP_CONVPOOR = 16384
        self.IPP_MASKVALUE =  8575
        self.IPP_MARKVALUE =  32768

        self.stackflag = False

        #print('VVVVV 0x%x' % self.IPP_MASKVALUE)
        #sys.exit(0)

    def add_options(self, parser=None, usage=None):
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler='resolve')
        parser.add_option('-v', '--verbose', action='count', dest='verbose',default=0)

        parser.add_option('--minnoise', default=10.0 , type="float",
                          help='If the median noise is below minnoise, then scale so that the median noise is minnoise (default=%default)')
        parser.add_option('--maxnoise', default=15.0 , type="float",
                          help='If the median noise is above maxnoise, then scale so that the median noise is maxnoise (default=%default)')
        parser.add_option('--nopedestal', default=False, action='store_true', 
                          help='skip adding a pedestal, which is added so that the smallest good data value is 1.0. limits can be set with --minpedestal and --maxpedestal (default=%default)')
        parser.add_option('--minpedestal',default=200.0, type='float', 
                          help='lower limit on pedestal that is added to the image. (default=%default)')
        parser.add_option('--maxpedestal',default=1000.0, type='float', 
                          help='upper limit on pedestal that is added to the image. This can be used to make sure that --autopedestal does not go wild... (default=%default)')
        parser.add_option('--specialmask',default=(None,None,None,None), nargs=4, type='int', 
                          help='pass x,y,boxsize,maskval, where maskval is the masking to be removed within the box around x,y')


        return(parser)

    def removespecialmask4area(self,x,y,boxsize,maskval):
        print('### Removing masking of 0x%x around a box of %d pixel around x,y=%d,%d' % (maskval,boxsize,x,y))
        
        x =  float(x)
        y =  float(y)
        boxsize =  int(boxsize)
        maskval =  hex2int(maskval)

        xmin  = np.amax([int(x-0.5*boxsize),0])
        xmax0 = int(x+0.5*boxsize)
        ymin  = np.amax([int(y-0.5*boxsize),0])
        ymax0 = int(y+0.5*boxsize)

        xmax = np.amin([int(self.header['NAXIS1']),xmax0])
        ymax = np.amin([int(self.header['NAXIS2']),ymax0])
         
        specialmask4area = np.zeros(self.maskdata.shape, dtype=np.int16)
        specialmask4area[ymin:ymax,xmin:xmax] = np.where((self.maskdata[ymin:ymax,xmin:xmax] & maskval)>0,self.IPP_BLANK,0)

        invmaskval = ~maskval
        self.maskdata[ymin:ymax,xmin:xmax]&=invmaskval

        return(specialmask4area)

    def mk_warps_photpipe_compatible(self,subdir,photcode):

        if self.options.verbose:
            print('massaging warp images in order to make them photpipe compatible')

        # Remove IPP_CONVBAD and IPP_CONVPOOR. These are unconvolved stacks, and these should not be set.
        IPP_BADMASK = 0x1fff

        if self.options.specialmask[0]!=None:
        #if self.options.specialmask[0]!=None and (not self.stackflag):
            specialmask4area = self.removespecialmask4area(self.options.specialmask[0],self.options.specialmask[1],
                                                           self.options.specialmask[2],self.options.specialmask[3])

        self.maskdata &= IPP_BADMASK

        # make sure all NaN values are masked.
        #nanpixels = np.logical_or(np.isnan(self.data),np.isnan(self.noisedata))
        nanpixels = np.isnan(self.data)
        self.maskdata[nanpixels]|= self.IPP_LOW
        self.data[nanpixels] = 0.0
        del nanpixels

        nanpixels = np.isnan(self.noisedata)
        self.maskdata[nanpixels]|= self.IPP_LOW
        self.noisedata[nanpixels] = 0.0
        del nanpixels

        goodpixels =  np.logical_and(self.noisedata>=0.0,np.equal(self.maskdata,0))

        if 'STK_TYPE' in self.header:
            saturation = 0.0 
        else:
            saturation = self.header['SATURATE']


        #pyfits.writeto('/Users/arest/delme/test1.fits',self.noisedata,overwrite=True)   
        self.noisedata = scipy.where(self.maskdata<1,np.sqrt(self.noisedata),self.noisedata)
        #pyfits.writeto('/Users/arest/delme/test2.fits',self.noisedata,overwrite=True)   
        
        # median sky and sky noise
        skysig = scipy.median(self.noisedata[goodpixels])
        skyadu = scipy.median(self.data[goodpixels])
        print('median sky:', skyadu)
        print('median sky noise:', skysig)

        # scale the image?
        scale4image=1.0
        if self.options.minnoise!=None:
            if skysig*scale4image<self.options.minnoise:
                if self.options.verbose:
                    print('the median noise=%f < %f, thus scaling by %f' % (skysig*scale4image,self.options.minnoise,self.options.minnoise/skysig))
                scale4image=self.options.minnoise/skysig

        if self.options.maxnoise!=None:
            if skysig*scale4image>self.options.maxnoise:
                if self.options.verbose:
                    print('the median noise=%f > %f, thus scaling by %f' % (skysig*scale4image,self.options.maxnoise,self.options.maxnoise/skysig))
                scale4image=self.options.maxnoise/skysig



        if scale4image!=1.0:
            self.noisedata *= scale4image
            self.data *= scale4image
            skysig*=scale4image
            saturation *= scale4image

        self.header.set('CPSCALE',scale4image,'IPP image got scaled by this factor')

        # add pedestal?
        pedestal = 0.0
        if not self.options.nopedestal:
            minpixval = scipy.amin(self.data[goodpixels])
            if self.options.verbose:
                print('lowest good pix value: ',minpixval)
            if minpixval<0.0:
                pedestal = -minpixval+1
                if self.options.verbose:
                    print('Setting pedestal to: ',pedestal)
            if self.options.minpedestal!=None and pedestal<self.options.minpedestal:
                pedestal = self.options.minpedestal
                if self.options.verbose:
                    print('min: setting pedestal to: ',pedestal)
            if self.options.maxpedestal!=None and pedestal>self.options.maxpedestal:
                pedestal = self.options.maxpedestal
                if self.options.verbose:
                    print('max: setting pedestal to: ',pedestal)

        if pedestal!=0.0:
            self.data+=pedestal
            skyadu+=pedestal
            saturation+=pedestal
        self.header.set('PEDESTAL',pedestal)
            
        self.header.set('SKYADU',skyadu,'median sky')
        self.header.set('SKYSIG',skysig,'median sky noise')
        if 'STK_TYPE' in self.header:
            self.header.set('SATURATE',65535)
        else:
            self.header.set('SATURATE',saturation)

        print('Pedestal: %f\nImage Scaling: %f\nSKYADU: %.2f\nSKYSIG: %.2f' % (pedestal,scale4image,skyadu,skysig))

        badpixels = np.nonzero(self.maskdata)
        # just make sure the 0x80 bit is set...
        self.maskdata[badpixels] |= 0x80

        #if self.options.specialmask[0]!=None and (not self.stackflag):
        if self.options.specialmask[0]!=None:
            # add back in the masks, but with mask values that won't trigger masking
            # in the pipeline
            self.maskdata |= specialmask4area

        # set all bad pixels to sky
        self.data[badpixels] = skyadu
        self.header.set('BADPVAL',skyadu,'Bad pixel value in warped image')
        self.maskheader.set('IPPBAD',IPP_BADMASK,'IPP mask bits that are deemed bad pixels')

        self.header.set('NOISEIM', 1)
        self.header.set('MASKIM', 1)
        self.header.set('SOFTNAME', "PSWarp")
        self.header.set('PIXSCALE', 0.25)
        self.header.set('SW_PLTSC', 0.25)

        self.header.set('PHOTCODE','0x%06x'% (hex2int(photcode)))
        self.header.set('SUBDIR',subdir)
        self.header.set('FITSNAME',os.path.basename(self.filename))

        Nmask = len(badpixels[0])
        Pmask = 100.0*Nmask/(int(self.header['NAXIS1'])*int(self.header['NAXIS2']))
        if self.options.verbose:
            print('%d out of %d pixels masked, %.2f percent' % (Nmask,(int(self.header['NAXIS1'])*int(self.header['NAXIS2'])),Pmask))
        self.header.set('MASKN',Nmask)
        self.header.set('MASKP',float('%.2f' % Pmask))

        # Get the RA,DEC of the center of image
        # make a dummy header with only relevant keywords. PS1 header
        # is non-conform, and gives lots of errors (HISTORY keywords)
        # and warnings
        dummyhdr=pyfits.Header()
        for k in ['CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2','CDELT1','CDELT2']:
            dummyhdr[k]=self.header[k]
        w = wcs.WCS(dummyhdr)
        radec = w.wcs_pix2world(np.array([[0.5*int(self.header['NAXIS1']),0.5*int(self.header['NAXIS2'])]]), 1)
        ra = deg2sex(radec[0][0],ra=True)
        dec =  deg2sex(radec[0][1])

        self.header.set('RA',ra)
        self.header.set('DEC',dec)
        self.header.set('SW_RA',ra)
        self.header.set('SW_DEC',dec)
        self.noiseheader.set('RA',ra)
        self.noiseheader.set('DEC',dec)
        self.noiseheader.set('SW_RA',ra)
        self.noiseheader.set('SW_DEC',dec)
        
        if 'BLANK' in self.header: del self.header['BLANK']
        if 'ZBLANK' in self.header: del self.header['ZBLANK']
        if 'BLANK' in self.noiseheader: del self.noiseheader['BLANK']
        if 'ZBLANK' in self.noiseheader: del self.noiseheader['ZBLANK']

        del goodpixels,badpixels
        
        return(0)

    def savefitsfiles(self,outputfilename):
        maskfilename = re.sub('fits','mask.fits',outputfilename)
        noisefilename = re.sub('fits','noise.fits',outputfilename)
        if maskfilename==outputfilename or noisefilename==outputfilename:
            raise RuntimeError('ERROR: could nto determine mask or wt file: %s, %s, %s' % (outputfilename,noisefilename,maskfilename))

        # overflow checking data
        if self.bitpix == 'int16':
            maxval = 32767.0 * self.bscale + self.bzero
            minval = -32768.0 * self.bscale + self.bzero

            toohigh = scipy.where(self.data>maxval)
            if self.options.verbose: print('%d values too high (>%f)' % (len(toohigh[0]),maxval))
            self.maskdata[toohigh] |= 0x80
            self.data[toohigh] = maxval

            toolow = scipy.where(self.data<minval)
            if self.options.verbose: print('%d values too low (<%f)' % (len(toolow[0]),minval))
            self.maskdata[toolow] |= 0x80
            self.data[toolow] = minval
        
        # overflow checking noise data
        if self.noisebitpix == 'int16':
            maxval = 32767.0 * self.noisebscale + self.noisebzero
            minval = -32768.0 * self.noisebscale + self.noisebzero

            # overflow checking noise
            toohigh = scipy.where(self.noisedata>maxval)
            if self.options.verbose: print('%d values too high in noise image (>%f)' % (len(toohigh[0]),maxval))
            self.maskdata[toohigh] |= 0x80
            self.noisedata[toohigh] = maxval

            toolow = scipy.where(self.noisedata<minval)
            if self.options.verbose: print('%d values too low in noise image (<%f)' % (len(toolow[0]),minval))
            self.maskdata[toolow] |= 0x80
            self.noisedata[toolow] = minval



        # make sure path exists
        makepath4file(outputfilename)
        # delete file. If there are any issues with premission, an error will be thrown
        rmfiles([outputfilename,maskfilename,noisefilename],gzip=True)

        print('Saving %s' % outputfilename)
        hdu_im = pyfits.PrimaryHDU(self.data,header=self.header)
        hdu_im.scale(self.bitpix, bscale=self.bscale,bzero=self.bzero)
        hdu_im.writeto(outputfilename,output_verify='fix',overwrite=True)
        os.chmod(outputfilename,0o664)

        print('Saving %s' % noisefilename)
        hdu_im = pyfits.PrimaryHDU(self.noisedata,header=self.noiseheader)
        hdu_im.scale(self.noisebitpix, bscale=self.noisebscale,bzero=self.noisebzero)
#        hdu_im.scale('float')
        hdu_im.writeto(noisefilename,output_verify='fix',overwrite=True)
        os.chmod(noisefilename,0o664)

        print('Saving %s' % maskfilename)
        hdu_im = pyfits.PrimaryHDU(self.maskdata,header=self.maskheader)
        hdu_im.scale('int16', bscale=1.0,bzero=32768.0)
        hdu_im.writeto(maskfilename,output_verify='fix',overwrite=True)
        os.chmod(maskfilename,0o664)

        return(0)
        
    def loadfitsfile(self,filename,maskflag=False):
        try:
            if self.options.verbose:
                print('Loading %s' % filename)
            this_data,this_header = pyfits.getdata(filename,header=True)
            if maskflag:
                #this_data = np.array(this_data,dtype=np.int16)
                print(this_data.dtype)
                this_data = np.array(this_data).astype(np.int16)
            return(0,this_header,this_data)

        except Exception as e:
            print("ERROR: Could not load fits file %s! error message %s" % (filename,e))
            return(2,None,None)

    def loadfitsfiles(self,filename): 
        # load data
        (errorflag1,self.header,self.data) = self.loadfitsfile(filename)
        if (errorflag1):
            raise RuntimeError('Error loading fits files %s' % (filename))

        if 'STK_TYPE' in self.header:
            maskfilename = re.sub('fits','mk.fits',filename)
        else:
            maskfilename = re.sub('fits','mask.fits',filename)
        wtfilename = re.sub('fits','wt.fits',filename)
        if maskfilename==filename or wtfilename==filename:
            raise RuntimeError('ERROR: could nto determine mask or wt file: %s, %s, %s' % (filename,wtfilename,maskfilename))

        (errorflag2,self.noiseheader,self.noisedata) = self.loadfitsfile(wtfilename)
        (errorflag3,self.maskheader,self.maskdata) = self.loadfitsfile(maskfilename,maskflag=True)
        errorflag = errorflag1 | errorflag2 | errorflag3
        if (errorflag1 | errorflag2 | errorflag3):
            raise RuntimeError('Error loading fits files %s (%d) and/or %s (%d) and/or %s (%d)' % (filename,errorflag1,wtfilename,errorflag2,maskfilename,errorflag3))

        if 'BOFFSET' in self.header:
            print '# Image file %s compressed, decompressing with BOFFSET=%f and BSOFTEN=%f ...' % (filename,self.header['BOFFSET'],self.header['BSOFTEN'])
            self.data = self.header['BOFFSET'] + self.header['BSOFTEN']*(np.exp(self.data/(2.5*np.log10(np.e))) - np.exp(-self.data/(2.5*np.log10(np.e))))
            del self.header['BOFFSET']
            del self.header['BSOFTEN']
        if 'BOFFSET' in self.noiseheader:
            print '# Weight file %s compressed, decompressing with BOFFSET=%f and BSOFTEN=%f ...' % (wtfilename,self.noiseheader['BOFFSET'],self.noiseheader['BSOFTEN'])
            self.noisedata = self.noiseheader['BOFFSET'] + self.noiseheader['BSOFTEN']*(np.exp(self.noisedata/(2.5*np.log10(np.e))) - np.exp(-self.noisedata/(2.5*np.log10(np.e))))
            del self.noiseheader['BOFFSET']
            del self.noiseheader['BSOFTEN']
        
        if self.options.verbose>2:
            print('%s, %s and %s loaded' % (filename,wtfilename,maskfilename))

        self.filename = filename

        return(0)

    def cpfixfile(self, inputfilename, outputfilename,subdir,photcode):
        self.loadfitsfiles(inputfilename)
        if 'STK_ID' in self.header:
            print('STACK IMAGE!!')
            self.stackflag=True
        self.mk_warps_photpipe_compatible(subdir,photcode)            
        self.savefitsfiles(outputfilename)
        print('Success cpfixGPC1.py')
        return(0)
            

if __name__ == "__main__":
    cpfixGPC1 = cpfixGPC1class()
    usagestring='USAGE: cpfixGPC1.py inputimage outputimage'
    parser=cpfixGPC1.add_options(usage=usagestring)
    cpfixGPC1.options, args = parser.parse_args()

    if len(args)!=4:
        parser.parse_args(args=['--help'])

    (inputfile,outputfile,subdir,photcode)=args

    cpfixGPC1.cpfixfile(inputfile,outputfile,subdir,photcode)
