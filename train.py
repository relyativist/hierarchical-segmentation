from dataset import setup_dataset
from utils import visualize_dataset_example


def main():
    dataset = setup_dataset()
    #pdb.set_trace()
    print(dataset[1])
    visualize_dataset_example(dataset, index=20)
    #print(dataset[0][0].shape, dataset[0][1].shape, dataset[0][2].shape)
    #spdb.set_trace()

if __name__ == "__main__":
    main()