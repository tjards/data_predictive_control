# A standardized dataset for DPC

import os
import numpy as np
import h5py

class Dataset:

    def __init__(self, filepath="data/dataset.h5", overwrite = True):
        
        # note: file can have multiple 'groups'
        self.filepath = filepath
        
        # removes existing file at initialization, if flagged
        if overwrite and os.path.exists(self.filepath):
            os.remove(self.filepath)

        # standardized columns (not all have to be stored)
        self.columns = [
            "step",
            "A_hat",
            "B_hat",
            "d_hat",
            "d",
            "target",
            "state",
            "input",
            "plan",
        ]
        
        # pipeline phase (becomes group name in h5 file)
        self.phase = "untitled"

        # initialize attributes
        for col in self.columns:
            setattr(self, col, [])

    # we can add one sample or batch(es) before storing 
    def stage(self, phase="untitled", step=None, A_hat=None, B_hat=None, d_hat=None, d=None, target=None, state=None, input=None, plan=None):

        # can rename group on the fly
        self.phase = phase

        values = {
            "step": step,
            "A_hat": A_hat,
            "B_hat": B_hat,
            "d_hat": d_hat,
            "d": d,
            "target": target,
            "state": state,
            "input": input,
            "plan": plan,
        }

        # append the values (if provided)
        for key, value in values.items():
            if value is not None:
                getattr(self, key).append(value)

    # stores and clears
    def store(self, flush_after=True):

        # make dir if needed
        if os.path.dirname(self.filepath):
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        with h5py.File(self.filepath, "a") as f:

            # create a new group, if needed 
            if self.phase not in f:
                group = f.create_group(self.phase)
            else:
                group = f[self.phase]

            # for each standardized data column
            for key in self.columns:

                # get the staged value(s)
                staged = getattr(self, key)

                # ignore empties
                if len(staged) == 0:
                    continue

                # handle single/multi batches 
                data = self._make_batch(staged)

                # column vectors cause issues 
                if key in ["d_hat", "d", "target", "state", "input"]:
                    if data.ndim >= 3 and data.shape[-1] == 1:
                        data = data.squeeze(-1)

                # expand if new column introduced in (this should be rare)
                if key not in group:
                    group.create_dataset(
                        key,
                        data=data,
                        maxshape=(None,) + data.shape[1:],
                        chunks=True,
                    )
                else:
                    old_len = group[key].shape[0]
                    new_len = old_len + data.shape[0]

                    group[key].resize(new_len, axis=0)
                    group[key][old_len:new_len] = data

        if flush_after:
            for col in self.columns:
                setattr(self, col, [])

    # can read whole phase, or phase + specific key
    def read(self, phase, key=None):

        # initialize dictionary
        data = {}

        with h5py.File(self.filepath, "r") as f:
            
            # pull out the group 
            group = f[phase]

            # if no key specified, return all 
            if key is None:
                for name in group.keys():
                    data[name] = group[name][:]
            # else, just the specified key
            else:
                data[key] = group[key][:]

        return data 



    def _make_batch(self, staged):

        # if one thing staged
        if len(staged) == 1:
            item = staged[0]

            arr = np.asarray(item)

            # if some ints in there
            if arr.ndim == 0:
                return arr.reshape(1)
            
            # if already a batch
            if isinstance(item, list):
                return arr

            return arr[np.newaxis, ...]

        # if multiple things staged 
        return np.asarray(staged)