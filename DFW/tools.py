import sys


def err_exit_mesg(mesg, status = -1):
    sys.stderr.write(mesg)
    sys.exit(status)
