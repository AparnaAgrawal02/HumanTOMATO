
from tma.callback import ProgressLogger
from tma.config import parse_args
from tma.data.get_data import get_datasets
from tma.models.get_model import get_model
from tma.utils.logger import create_logger
from tma.utils.metrics import get_metric_statistics, print_table

#Parse arguments from command line
cfg = parse_args()

# Create a logger for logging events during training
logger = create_logger(cfg, phase="test")

text = "Test"
print(text)

#get model
model = get_model(cfg)

#load model
model = model.load_from_checkpoint(cfg.TEST.CHECKPOINT_PATH)
model.eval()

#text to motion 
#get npy motion file from text
motion = model.text2motion(text)
print(motion)

