import sys
from train import train
from dataset import TerrainDataset

def main():
    dataset = sys.argv[-1]
    print("Loading", dataset)
    dataset = TerrainDataset(dataset)
    train(dataset, checkpoint_path=None, lr=2e-5, num_epochs=75)

if __name__ == '__main__':
    main()
