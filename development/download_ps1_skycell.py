import numpy
from astropy.table import Table
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import os

def getimages(ra,dec,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table


def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    
    """Get URL for images in the table
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={format}")
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[numpy.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url


def getcolorim(ra, dec, size=240, output_size=None, filters="grizy", format="jpg"):
    
    """Get color image at a sky position
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """
    
    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra,dec,size=size,filters=filters,output_size=output_size,format=format,color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    return im


def getgrayim(ra, dec, size=240, output_size=None, filter="g", format="jpg"):
    
    """Get grayscale image at a sky position
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filter = string with filter to extract (one of grizy)
    format = data format (options are "jpg", "png")
    Returns the image
    """
    
    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    if filter not in list("grizy"):
        raise ValueError("filter must be one of grizy")
    url = geturl(ra,dec,size=size,filters=filter,output_size=output_size,format=format)
    r = requests.get(url[0])
    im = Image.open(BytesIO(r.content))
    return im


def get_img_lc(ra, dec, fname_list = [],home_dir = './'):
    image_list = getimages(ra, dec)
    image_list = image_list[image_list['filter']!='g']
    fnames = []
#    return image_list['filename']
    for filename in image_list['filename']:
        if filename.split('/')[-1] not in fname_list:
            url = "http://ps1images.stsci.edu/" + filename
            response = requests.get(url)
            if response.status_code == 200:
                outputfilename = home_dir + filename.split('/')[-1]+ ".fz"
                with open(outputfilename, "wb") as file:
                    file.write(response.content)
                os.system('funpack -D '+outputfilename)
                print(filename.split('/')[-1] + ' '+str(ra) + ' '+ str(dec))
                fnames = fnames + [filename.split('/')[-1]]
            else:
                print("Failed to download the file "+filename.split('/')[-1] + 'for coordinates '+ str(ra)+' '+str(dec))
    fname_list = fname_list + fnames
    return fname_list

def bulk_download(rac, decc, radius, home_dir = './'):
    outfiles = []
    for dec in np.arange(decc-radius+0.001, decc+radius+0.2, 0.2):
        r4ra = np.sqrt(radius**2-(decc-dec)**2)
        for ra in np.arange(rac-r4ra/np.cos(dec/180*np.pi), 
                            rac+(r4ra+0.19)/np.cos(dec/180*np.pi),
                            0.2/np.cos(dec/180*np.pi)):
            outfiles = get_img_lc(ra, dec,fname_list = outfiles, home_dir = home_dir)
    return outfiles