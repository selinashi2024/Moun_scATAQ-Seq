import scanpy as sc
import sys

# Redirect output to both console and file
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("adata_inspection.log")

# Load the processed data
print("Loading processed data...")
adata = sc.read_h5ad("processed_scataq.h5ad")

# Print basic info
print("\nDataset Overview:")
print(adata)

print("\nFirst 10 variable names:")
print(list(adata.var_names[:10]))

print("\nAvailable variables in .var:")
print(list(adata.var.columns))

print("\nObservation annotations:")
print(list(adata.obs.columns))

print("\nShape of data:")
print(f"Number of cells: {adata.n_obs}")
print(f"Number of peaks: {adata.n_vars}")