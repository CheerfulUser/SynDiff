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

export PS1="\u@\h(syndiff)% "
