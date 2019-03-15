import warnings
import argparse
from helpers import process_image
from pathlib import Path
warnings.filterwarnings("error")

def get_dataset(base_path, file, image_size):
    warnings.filterwarnings('error')
    data = [str(f.absolute()) for f in Path(base_path).glob('**/*.jpg')]
    bad = []
    with open(file, "w") as f:
        for item in data:
            try:
                img, _ = process_image(item, 0, image_size)
                assert img.shape[2] == 3, "Invalid channel count"
                # write out good images
                f.write('{}\n'.format(item))
            except Exception as e:
                bad.append(item)
                print('{}\n{}\n'.format(e, item))
            except RuntimeWarning as w:
                bad.append(item)
                print('--------------------------{}\n{}\n'.format(w, item))

        print('{} bad images ({})'.format(len(bad), len(bad) / float(len(data))))
    return bad

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Training for Image Recognition.')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-t', '--target', help='target file to hold good data', default='dataset.txt')
    parser.add_argument('-i', '--img_size', help='target image size to verify', default=160, type=int)
    args = parser.parse_args()

    get_dataset(args.data, args.target, args.img_size)

    # python clean.py -d data/PetImages -t dataset.txt
