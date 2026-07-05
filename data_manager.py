import os

class Dataset:
    def __init__(self, filepath="data/dataset.txt"):
        
        # config variables 
        self.filepath = filepath
        self.started = False
        
        # labels
        self.columns = ["step", "A_hat", "B_hat", "d_hat", "d", "target", "state", "input"]
        
        # initiate attributes from column list
        for column in self.columns:
            setattr(self, column, [])

    def start(self):
        if os.path.dirname(self.filepath):
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            
        # write fresh column headers (wipes previous)
        with open(self.filepath, "w") as f:
            f.write(",".join(self.columns) + "\n")

        self.started = True

    def store(self, flush_after=True):
        
        # add headers, of not aleady done
        if not self.started:
            self.start()

        # get columns
        active_lists = [getattr(self, col) for col in self.columns]
        
        # calc batch size
        batch_size = max(len(lst) if isinstance(lst, list) else 0 for lst in active_lists)
        if batch_size == 0:
            return

        with open(self.filepath, "a") as f:
            
            # replace nones/empties with a column of [None]s matching batch size
            normalized_iterables = [
                lst if (isinstance(lst, list) and lst) else [None] * batch_size 
                for lst in active_lists
            ]
            
            # zip horizontally across data, write append 
            for row in zip(*normalized_iterables):
                f.write(",".join(str(item) for item in row) + "\n")

        # clear out batch
        if flush_after:
            for col in self.columns:
                value = getattr(self, col)
                if isinstance(value, list):
                    value.clear()
                else:
                    setattr(self, col, [])




