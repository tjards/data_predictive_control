import os
import re
import numpy as np
import json

class Dataset:

    def __init__(self, filepath="data/dataset.txt"):
        
        # config variables 
        self.filepath = filepath
        self.started = False
        
        # labels
        self.columns = ["step", "A_hat", "B_hat", "d_hat", "d", "target", "state", "input"]
        
        # initiate attributes from column list as empty lists
        for col in self.columns:
            setattr(self, col, [])

    def start(self):
        
        # make the directory, if it doesn't exist
        if os.path.dirname(self.filepath):
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            
        # write fresh column headers (wipes previous data log)
        with open(self.filepath, "w") as f:
            f.write(",".join(self.columns) + "\n")

        # flag so this doesn't have to be done with each batch
        self.started = True

    # loads batches into dataset
    def load(self, step = None, A_hat = None, B_hat = None, d_hat = None, d = None, target = None, state = None, input = None):

        self.step   = step
        self.A_hat  = A_hat
        self.B_hat  = B_hat
        self.d_hat  = d_hat
        self.d      = d
        self.target = target
        self.state  = state
        self.input  = input 

     # store file and flush (complicated because sees some None, empties, and single samples)
     # store file and flush (complicated because sees some None, empties, and single samples)
    def store(self, flush_after=True):
        
        # add columns on first run
        if not self.started:
            self.start()
        
        # calculate the batch size
        batch_size = 1
        # cycle through each column
        for col in self.columns:
            # get the values 
            val = getattr(self, col)
            # if it's a list with samples
            if isinstance(val, list) and len(val) > 0:
                # check sequence length first to prevent index out of range errors
                if len(val) == 1 and not isinstance(val[0], (int, float, str)):
                    continue
                # keep the largest column
                batch_size = max(batch_size, len(val))

        # normalize various scenarios
        normalized_iterables = []
        for col in self.columns:
            val = getattr(self, col)
            # deal with Nones
            if val is None:
                normalized_iterables.append([None] * batch_size)
            # deal with well-behaved list
            elif isinstance(val, list) and len(val) == batch_size and batch_size > 1:
                normalized_iterables.append(val)
            # single point (sometimes happens if something is set at start of batch)
            else:
                normalized_iterables.append([val] * batch_size)

        # append
        with open(self.filepath, "a") as f:
            for row in zip(*normalized_iterables):
                line = []
                for item in row:
                    if item is None:
                        line.append("None")
                        continue
                    
                    import numpy as np
                    import re
                    
                    if isinstance(item, np.ndarray):
                        # np arrays get converted to standard string layouts
                        # separator='; ' keeps data distinct without extra line breaks
                        item_str = np.array2string(item, separator='; ').replace('\n', ' ')
                        item_str = re.sub(r'\s+', ' ', item_str).strip()
                    else:
                        # lists convert directly to strings
                        item_str = str(item).replace("\n", " ")
                        item_str = re.sub(r'\s+', ' ', item_str).strip()
                        item_str = item_str.replace(",", ";")
                        
                    line.append(item_str)
                    
                f.write(",".join(line) + "\n")
            
            f.flush()

        # clear for next batch
        if flush_after:
            for col in self.columns:
                setattr(self, col, [])


    def load_dataset_log(self, filepath="data/dataset.txt"):

        # outputs a dict
        data_out = {}
        
        # read all the lines 
        with open(filepath, "r") as f:
            lines = f.readlines()
            
        # pull out heads 
        headers = [h.strip() for h in lines[0].split(",")]
        for h in headers:
            data_out[h] = []
            
        # cycle through each line
        for line in lines[1:]:
            row_cells = line.strip().split(",")

            # skip incomplete lines 
            if len(row_cells) < len(headers):
                continue 
                
            for idx, cell in enumerate(row_cells):
                
                key = headers[idx]
                cell = cell.strip()
                
                # deal with nones/empties 
                if cell == "None" or cell == "":
                    data_out[key].append(None)
                    
                # semi-colons within cells
                elif "[" in cell:

                    # replace semicolons and consecutive blanks with commas 
                    json_ready = cell.replace(";", ",")
                    json_ready = re.sub(r'(?<=\S)\s+(?=\S)', ', ', json_ready)
                    
                    try:
                        # leverage json to recognize matrices 
                        native_list = json.loads(json_ready)
                        data_out[key].append(np.array(native_list))
                    except Exception:
                        data_out[key].append(cell) 
                        
                else:
                    try:
                        data_out[key].append(float(cell))
                    except ValueError:
                        data_out[key].append(cell) 
                        
        for k, v in data_out.items():
            valid_items = [item for item in v if item is not None]
            if valid_items and isinstance(valid_items[0], np.ndarray):
                try:
                    data_out[k] = np.array(v)
                except ValueError:
                    pass 
                    
        return data_out
