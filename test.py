from compression import compression_factory
from evaluation import test_for_parameters, prepare_results_folder


orders = [0.20, 0.30,  0.50]
all_methods = ['dft', 'dct', 'pca', 'svd']
block_shapes = [32,8,  64]
data_folder = 'datasets/kodak/'


for block in block_shapes:
	for order in orders:
		for method in all_methods:

			results_folder = prepare_results_folder(order, block, method)
			compressor = compression_factory(method, order, block)
			print(f'Order: {order}, method: {method}, block: {block}')
			test_for_parameters(compressor, data_folder, results_folder)

