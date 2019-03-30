#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
import sys, os

for year in range(1979, 1980):
    fn_netcdf = "ERA_INTERIM_{0:04d}.nc".format(year)
    print(fn_netcdf)
    if not os.path.isfile(fn_netcdf): 
        print("Trying to download {0:s}".format(fn_netcdf))
        try:
            server = ECMWFDataServer()
            server.retrieve({
                'dataset'   : "interim",
                'class'     : "ei",
                'expver'    : "1",
                'stream'    : "oper",
                'type'      : "an",
                'levtype'   : "pl",
                'levelist'  : "150",
                'grid'      : "0.9983/0.9983",
                'format'    : "netcdf",
                'param'     : "60.128",
                'time'      : "00:00:00",
                'target'    : fn_netcdf,
                'date'      : "{0:04d}-01-01/to/{0:04d}-01-02".format(year),
            })
            print("Everything works for {0}".format(fn_netcdf))

        except Exception as e:
            print(e)
            print("Problems with {0}".format(fn_netcdf))
