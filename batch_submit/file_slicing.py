#!/usr/bin/env python

from netCDF4 import Dataset
import numpy as np
import sys

#............................
def read_netcdf_file(fname, varname): # Variables names: e.g., 'lon', 'lat', 'prw'
        fh = Dataset(fname, mode='r')
        var_netcdf = fh.variables[varname][:] # Retrieves a given variable by name.
        fh.close()
        return var_netcdf

if __name__ == "__main__":

    fname = sys.argv[1]
    f_idx = int(sys.argv[2]) # SLURM_ARRAY_TASK_ID
    n_proc = int(sys.argv[3])

    print('[+] ==== %s File name: %s File index: %i No processes/frames: %i' %(sys.argv[0], fname, f_idx, n_proc))

    data = read_netcdf_file(fname, 't') # For now fix the variable.

    f_data = [data[int(i + f_idx*32)][0] for i in range(0, n_proc) if int(i + f_idx*32) <= 1459] # Last slice of frames has 20 frames.

    if len(f_data) == 0: # Safely exit if all indices are out of file range.
        print('[-] ==== SAFE EXIT! Indices out of file range.')
        sys.exit(0)

    print('==== Min frame index: %d Max frame index: %d ' %(f_idx*32, (n_proc-1)+f_idx*32))
    print('==== Extracted number of frames: %d ' %len(f_data))
    print('==== Size of one frame: %d %d ' %(f_data[-1].shape[0], f_data[-1].shape[1]))

    new_fname = fname.split('/')[-1][:-3] + '_' + str(f_idx) + '_' + '.npz'
    np.savez(new_fname, *f_data) # Pattern: ERA_INTERIM_1980_(f_idx)_.npz

    print('[+] ==== File: ' + new_fname + ' saved!')

