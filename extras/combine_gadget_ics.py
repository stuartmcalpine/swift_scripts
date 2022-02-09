"""
Combine multiple hdf5 gadget IC files into a single file in parallel.

mpirun -np X combine_ics.py /path/to/hdf5/file 

If two args are provided, second arg is the output folder.

Do not include the '.0.hdf5' at the end of the input string, just the basename.
This basename will be the output basename.

Note that h5py + mpi4py doesn't support hdf5 compression.

Therefore you have to compress the file afterwards using:

h5repack -l
PartType1/Masses,PartType2/Masses,PartType1/ParticleIDs,PartType2/ParticleIDs:CHUNK=10000
-l
PartType1/Coordinates,PartType2/Coordinates,PartType1/Velocities,PartType2/Velocities:CHUNK=10000x3
-v infile.hdf5 tmp.hdf h5repack -f GZIP=9 -f SHUF -f FLET -v tmp.hdf5
outfile.hdf5
"""

import h5py
import numpy as np
import sys
from mpi4py import MPI

comm_rank = MPI.COMM_WORLD.rank
comm_size = MPI.COMM_WORLD.size
comm = MPI.COMM_WORLD

class CombineICs:

    def __init__(self, fname, out_folder):
        self.fname      = fname
        self.sim_name   = self.fname.split('/')[-1]

        # New file.
        self.new_f = h5py.File('%s%s.hdf5'%(out_folder, self.sim_name), 'w', driver='mpio',
                comm=comm)
        self.new_f.atomic = True

        self.copy_header()
        self.create_datasets()
        self.index_particles()
        self.copy_particles()

        # Close file.
        self.new_f.close()


    def copy_header(self):
        """ Copy header information to new file. """

        f = h5py.File(self.fname + '.0.hdf5', 'r', driver='mpio', comm=comm)
        grp = self.new_f.create_group('Header')
        self.attrs = {}
        for att in f['Header'].attrs.keys():
            if att == 'NumPart_ThisFile':
                grp.attrs[att] = f['Header'].attrs['NumPart_Total']
            elif att == 'NumFilesPerSnapshot':
                grp.attrs[att] = 1
            else:
                grp.attrs[att] = f['Header'].attrs[att]
            self.attrs[att] = f['Header'].attrs[att]

        self.attrs['ntot'] = np.zeros(6, dtype=np.uint64)
        for i in range(6):
            self.attrs['ntot'][i] = (self.attrs['NumPart_Total_HighWord'][i] << 32) + \
                self.attrs['NumPart_Total'][i]
        f.close()

    def create_datasets(self):
        """ Set up the datasets for the output file. """
       
        for i in range(6):
            if self.attrs['ntot'][i] > 0:
                if comm_rank == 0: print('Creating datasets, %i particles'%(self.attrs['ntot'][i]))
                grp = self.new_f.create_group('PartType%i'%i)

                if comm_rank == 0: print('Creating coordinates...')
                self.create_set(grp, "Coordinates", self.attrs['ntot'][i], 3, "d")
                if comm_rank == 0: print('Creating velocities...')
                self.create_set(grp, "Velocities", self.attrs['ntot'][i], 3, "f")
                if comm_rank == 0: print('Creating masses...')
                self.create_set(grp, "Masses", self.attrs['ntot'][i], 1, "f")
                if comm_rank == 0: print('Creating particleids...')
                self.create_set(grp, "ParticleIDs", self.attrs['ntot'][i], 1, "l")
                if i == 0:
                    self.create_set(grp, "InternalEnergy", self.attrs['ntot'][i], 1, "f")
                    self.create_set(grp, "SmoothingLength", self.attrs['ntot'][i], 1, "f")

    def copy_particles(self):
        if comm_rank == 0:
            print('Copying %i particles from %i files.\n'\
                    %(np.sum(self.attrs['ntot']), self.attrs['NumFilesPerSnapshot']))
        for i in range(6):
            if self.attrs['ntot'][i] == 0: continue

            if comm_rank == 0:
                print('Copying %i PartType%i particles...'%(self.attrs['ntot'][i], i))

            left = 0
            total_num = 0
            for j in range(self.attrs['NumFilesPerSnapshot']):
                if j % comm_size != comm_rank: continue
                this_f = h5py.File(self.fname + '.%i.hdf5'%j, 'r')
           
                this_num = this_f['Header'].attrs['NumPart_ThisFile'][i]
                total_num += this_num
                if this_num > 0:
                    for att, att_new in zip(['Coordinates', 'Velocities', 'Masses', 'ParticleIDs'],
                            ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs']):
                        tmp_data = this_f['PartType%i/%s'%(i,att)][...]
                        tmp_left = self.offsets[i][comm_rank]+left
                        tmp_right = self.offsets[i][comm_rank]+left+this_num
                        assert tmp_right - tmp_left == len(tmp_data), 'Indexing error'
                        with self.new_f['PartType%i/%s'%(i,att_new)].collective:
                            self.new_f['PartType%i/%s'%(i,att_new)][tmp_left:tmp_right] = tmp_data
                    left += this_num
                this_f.close()
                print('[Rank %i] Done %i/%i'%(comm_rank, j+1, self.attrs['NumFilesPerSnapshot']))

    def index_particles(self):
        if comm_rank == 0:
            print('Indexing %i particles from %i files.\n'\
                %(np.sum(self.attrs['ntot']), self.attrs['NumFilesPerSnapshot']))

        self.offsets = []
        for i in range(6):
            this_rank_num_particles = 0
            if self.attrs['ntot'][i] == 0: 
                self.offsets.append(np.full(comm_size,-1))
                continue

            for j in range(self.attrs['NumFilesPerSnapshot']):
                if j % comm_size != comm_rank: continue
                    
                this_f = h5py.File(self.fname + '.%i.hdf5'%j, 'r')
       
                this_rank_num_particles += this_f['Header'].attrs['NumPart_ThisFile'][i]

                this_f.close()

            total_counts = comm.allreduce(this_rank_num_particles)
            assert total_counts == self.attrs['ntot'][i],\
                'Add up error %i != %i'%(total_counts, self.attrs['ntot'][i])
            gather_counts = comm.allgather(this_rank_num_particles)
            rights = np.cumsum(gather_counts) 
            lefts = rights - gather_counts
            counts = rights - lefts
            assert np.array_equal(gather_counts, counts), 'Not add up'
            self.offsets.append(lefts)

    # Helper function to create the datasets we need
    def create_set(self, grp, name, size, dim, dtype):
        if dim == 1:
            grp.create_dataset(
                name,
                (size,),
                dtype=dtype
            )
        else:
            grp.create_dataset(
                name,
                (size, dim),
                dtype=dtype
            )

if __name__ == '__main__':
    fname = sys.argv[1]
    if len(sys.argv) > 2:
        out_folder = sys.argv[2]
    else:
        out_folder = './ics/'
    x = CombineICs(fname, out_folder)
