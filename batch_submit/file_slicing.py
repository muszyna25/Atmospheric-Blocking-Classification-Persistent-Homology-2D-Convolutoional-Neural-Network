#!/usr/bin/env python

import os
from netCDF4 import Dataset
import numpy as np
import sys

#............................
    def read_netcdf_file(fname, varname): #Variables names: e.g., 'lon', 'lat', 'prw'
        fh = Dataset(fname, mode='r')
        var_netcdf = fh.variables[varname][:] #Retrieves a given variable by name.
        fh.close()
        return var_netcdf

if __name__ == "__main__":

    fname = sys.argv[1]
    f_idx = int(sys.argv[2]) # SLURM_ARRAY_TASK_ID
    n_proc = int(sys.argv[3])

    data = read_netcdf_file(fname, 't')

    f_data = [data[int(i + f_idx*32)][0] for i in range(0, n_proc) if int(i + f_idx*32) <= 1459]

    np.savez(fname.split('/')[-1][:-3] + '_' + str(f_idx) + '_' + '.npz', *f_data) # Pattern: ERA_INTERIM_1980_(f_idx)_.npz

