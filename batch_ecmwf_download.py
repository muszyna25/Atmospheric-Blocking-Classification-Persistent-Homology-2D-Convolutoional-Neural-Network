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
                'levelist'  : "150/200/250/300/350/400/450/500",
                'grid'      : "0.75/0.75",
                'format'    : "netcdf",
                'param'     : "60.128/129.128/130.128/131.128/132.128",
                'time'      : "00:00:00/06:00:00/12:00:00/18:00:00",
                'target'    : fn_netcdf,
                'date'      : "{0:04d}-01-01/to/{0:04d}-12-31".format(year),
            })
            print("Everything works for {0}".format(fn_netcdf))

        except Exception as e:
            print(e)
            print("Problems with {0}".format(fn_netcdf))
