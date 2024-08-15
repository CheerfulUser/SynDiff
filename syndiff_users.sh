if [[ $HOSTNAME =~ arminmac* ]]; then
   export SYNDIFF_SRCDIR="/Users/arest/pipes/SynDiff"
   export SYNDIFF_DATADIR="/Users/arest/data/SynDiff"
   export SYNDIFF_BATCH_SYSTEM=None
elif [[ $HOSTNAME =~ plhstproc* ]]; then
   export SYNDIFF_SRCDIR="/astro/armin/pipe/SynDiff"
   export SYNDIFF_DATADIR="/astro/armin/data/SynDiff"
   export SYNDIFF_BATCH_SYSTEM=Condor
fi

source $SYNDIFF_SRCDIR/syndiff.sh

