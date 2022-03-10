import numpy as np
import os
import re

class ForceChecksData:

    def __init__(self, data_dir):

        # Base directory all the exact force and swift force dat files are stored.
        self.data_dir = data_dir

    def find_steps(self):
        """ Probe the data dir and see what steps have exact forces computed. """

        files = [x for x in os.listdir(self.data_dir) if 'gravity_checks' in x and '.dat' in x \
                and 'exact' not in x]
        steps = [int(x.split('step')[1][:4]) for x  in files]
        orders = [int(x.split('order')[1][0]) for x  in files]
        
        return steps, orders

    def load_swift_step_exact(self, filename, ):
        """ Load exact forces. """

        print('Loading ', filename)
        data_exact = np.loadtxt(filename,
            dtype={'names': ['id', 'x', 'y', 'z', 'ax', 'ay',
            'az', 'pot', 'ax_short', 'ay_short', 'az_short','ax_long', 'ay_long', 'az_long'],
            'formats': ['i8', 'f8', 'f8','f8','f8','f8','f8','f8', 'f8','f8','f8',
            'f8','f8','f8']}, skiprows=7)

        return data_exact

    def load_swift_step(self, filename):
        """ Load swift approximated forces. """

        print('Loading ', filename)
        data = np.loadtxt(filename, dtype={'names': [
            'id', 'x', 'y', 'z', 'ax', 'ay', 'az', 'pot',
            'ax_long', 'ay_long', 'az_long', 'pot_mesh', 'ax_p2p', 'ay_p2p', 'az_p2p',
            'ax_p2m', 'ay_p2m', 'az_p2m', 'ax_m2m', 'ay_m2m', 'az_m2m', 'n_p2p', 'n_p2m',
            'n_m2m', 'n_pm'], 
            'formats': ['i8', 'f8', 'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                'f8','f8','f8','f8','f8', 'f8','f8','f8','i8','i8','i8','i8']}, skiprows=8)

        # Sanity checks.
        ninteractions = data['n_p2p'] + data['n_p2m'] +data['n_m2m'] + data['n_pm']
        #assert len(np.unique(ninteractions)) == 1, 'Some interactions dont add up'
        
        return data

    def load_step(self, step_no, order, periodic):
        """ Load force information for a particular step. """

        # Load swift forces.
        fname = f'{self.data_dir}/gravity_checks_swift_step{step_no:04d}_order{order}.dat' 
        data_swift = self.load_swift_step(fname)

        # Load exact forces.
        if periodic:
            fname = f'{self.data_dir}/gravity_checks_exact_periodic_step{step_no:04d}.dat'
        else:
            fname = f'{self.data_dir}/gravity_checks_exact_step{step_no:04d}.dat'
        data_exact = self.load_swift_step_exact(fname)

        # Make sure all particles matched.
        assert np.array_equal(data_swift['id'], data_exact['id']), 'id error'

        return data_swift, data_exact

    def compute_quantities(self, data_swift, data_exact):
        """ Compute force errors, percentiles, etc..."""

        data = {}

        # Compute relative force error.
        a = np.sqrt(data_swift['ax']**2. + data_swift['ay']**2. + data_swift['az']**2.)
        a_exact = np.sqrt(data_exact['ax']**2. + data_exact['ay']**2. + data_exact['az']**2.)
        data['da'] = np.true_divide(np.abs(a-a_exact), a_exact, dtype='f8')

        # Cumulative relative force error.
        hist, bin_edges = np.histogram(data['da'], bins=10**np.arange(-8,3,0.01))
        data['hist_y'] = np.true_divide(hist, np.max(hist))
        data['hist_x'] = (bin_edges[1:] + bin_edges[:-1]) / 2.
        cumulative = np.cumsum(hist)
        data['cumulative_hist_y'] = np.true_divide(cumulative, cumulative[-1])
        
        # Percentiles.
        for p in [50, 90, 99]:
            data[f'p{p}'] = np.percentile(data['da'], p)

        return data

if __name__ == '__main__':

    dir="/cosma7/data/dp004/rttw52/swift-paper-runs/individual-galaxy-force-checks/o5/geometric_0p9"
    step_no = 0
    order = 5
    periodic = False

    # ---
    x = ForceChecksData(dir)
    data_swift, data_exact = x.load_step(step_no, order, periodic)
    data = x.compute_quantities(data_swift, data_exact)

    import matplotlib.pyplot as plt

    plt.loglog()
    plt.plot(data['hist_x'], data['hist_y'])
    plt.ylabel('Norm Count')
    plt.xlabel(r'$|a - a_{\mathrm{exact}}| / a_{\mathrm{exact}}$')
    plt.axvline(data['p90'], label=f"p99={data['p90']}")
    plt.axvline(data['p99'], label=f"p99={data['p99']}")
    plt.legend()
    plt.tight_layout()
    plt.show()


