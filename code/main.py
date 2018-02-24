import pickle
import pprint

def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct

def main():
	pp = pprint.PrettyPrinter(indent=4)
	data_batch_1 = unpickle('../data/data_batch_1')
	# for key in data_batch_1.keys():
	# 	print(len(data_batch_1[key]))
	print(data_batch_1[b'data'])

main()