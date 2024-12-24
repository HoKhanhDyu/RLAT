from collections import namedtuple
import torchvision.transforms as transforms

DQN_TRIENVONG = "model_0_new_1.pth"
CLASSIFIER_ARCH = "vgg16"
CLASSIFIER_TRAINED = "Trained_model\classifier_cifar10.pth"
DATASET = "cifar10_splits"
BATCH_SIZE = 512
EPS = 20
EPSILON = 1
IMG_SIZE = 28 # areas: img_size ^ 2

MAX_ITER = 300
NOISE_SD = 0.005
GAMMA = 0.98
TARGET_UPDATE = 100
NUM_TESTS = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Assuming CIFAR-10 normalization
])

INPUT_DQN_SIZE = 28 * 28 + 32 + 3 + 10 + 10 + 10
OUTPUT_DQN_SIZE = 6 + 1
DQN_TRAINED = "model_0.pth"
TRACK_FOLDER = "view"
TEST_FOLDER = "view_test"
ACTION_TRACK = "actions.json"
LOSS_TRACK = "losses.json"
REWARD_TRACK = "rewards.json"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))