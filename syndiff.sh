# This scripts expects that the following environment variables are set:
# SYNDIFF_DATADIR
# SYNDIFF_SRCDIR
if [ -z ${SYNDIFF_DATADIR+x} ]; then
   echo "environment variable SYNDIFF_DATADIR is not set! exiting..."
   return 1;
fi
if [ -z ${SYNDIFF_SRCDIR+x} ]; then
   echo "environment variable SYNDIFF_DATADIR is not set! exiting..."
   return 1;
fi

export SYNDIFF_DEFAULT_CFG_FILE="$SYNDIFF_SRCDIR/syndiff_default_cfg.ini"

alias cdsrc='cd $SYNDIFF_SRCDIR'
alias cddata='cd $SYNDIFF_DATADIR'

if [ -z ${SYNDIFF_DEFAULTPATH+x} ]; then
   export SYNDIFF_DEFAULTPATH=$PATH
   # echo "ERROR: The SYNDIFF_DEFAULTPATH is not set! check out the syndiff.sh file in the main config dir"
   # return 1;
else
   export PATH=$SYNDIFF_DEFAULTPATH
fi
export PATH="${PATH}:${SYNDIFF_SRCDIR}:${SYNDIFF_SRCDIR}/src"

export PS1="\u@\h(syndiff)% "
