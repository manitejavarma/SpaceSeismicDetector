import torch
DATA_DIR = r'C:\Users\Mani Teja Varma\Documents\NASA\space_apps_2024_seismic_detection\data'
WINDOW_SIZE = 6000
STEP_SIZE = WINDOW_SIZE // 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')