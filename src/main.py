
from config import *
from models.sta_lta import sta_lta_model
from utilities.utilities import load_single_file

# load data 1. mars / lunar.
# run algo
# plot results

# stream_file[0] refers to first trace in the stream. And traces are number of instruments used to record the data
if __name__ == "__main__":
    data_directory = f'{DATA_DIR}/lunar/training/data/S12_GradeA/'
    test_filename = 'xa.s12.00.mhz.1970-06-26HR00_evid00009'
    mseed_file = f'{data_directory}{test_filename}.mseed'
    stream_file = load_single_file(mseed_file, file_type="mseed")
    tr = stream_file.traces[0].copy()
    tr_times = tr.times()
    tr_data = tr.data

    sta_lta_model(sta_len=120, lta_len=600, stream_file=stream_file)
