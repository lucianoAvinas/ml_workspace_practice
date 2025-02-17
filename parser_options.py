import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

### General Params ###
experiment_group = parser.add_argument_group('experiment arguments')
experiment_group.add_argument('-n_vis', '--n_visual_exs', type=int, default=4, dest='n_vis',
	                          help='Limited to the current batch size of the training.')
experiment_group.add_argument('--pool_size', type=int, default=50,
	                          help='Replay buffer of previously generated images.')
experiment_group.add_argument('--replay_prob', type=float, default=0.5,
	                          help='Probability previously generated image gets queried\n'
	                               'from image pool.')


### Generator Params ###
generator_group = parser.add_argument_group('generator arguments')
generator_group.add_argument('-g_in_ch', '--gen_input_nc', type=int, default=1, dest='input_nc',
	                         help='Generator in-channels.')
generator_group.add_argument('-g_out_ch', '--gen_output_nc', type=int, default=1, dest='output_nc',
	                         help='Generator out-channels.')
generator_group.add_argument('--num_downs', type=int, default=4, 
	                         help='Number of downsamples.')
generator_group.add_argument('--g_max_exp', type=int, default=3, dest='max_exp',
	                         help='Generator maximum expansion exponent.')
generator_group.add_argument('--ngf', type=int, default=32,
	                         help='Initial number of generator features.')
generator_group.add_argument('--g_norm_layer', default='batch', choices=['batch', 'instance'],
	                         dest='norm_layer', help='Generator normalization layer.')
generator_group.add_argument('--use_dropout', action='store_true', help='Activates 0.2 droupout.')


### Discriminator Params ###
discriminator_group = parser.add_argument_group('discriminator arguments')
discriminator_group.add_argument('-d_in_ch', '--disc_input_nc', type=int, default=1, dest='input_nc',
	                             help='Discriminator in-channels.')
discriminator_group.add_argument('--d_max_exp', type=int, default=3, dest='max_exp',
	                             help='Discriminator maximum expansion exponent.')
discriminator_group.add_argument('--ndf', type=int, default=32,
	                             help='Initial number of discriminator features.')
discriminator_group.add_argument('--n_layers', type=int, default=3,
	                             help='Determines PatchDiscriminator field of view.')
discriminator_group.add_argument('--d_norm_layer', default='batch', choices=['batch', 'instance'],
	                             dest='norm_layer', help='Discriminator normalization layer.')


### Optimizer Learning Params ###
optimizer_group = parser.add_argument_group('optimizer arguments')
optimizer_group.add_argument('-lr', '--learning_rate', type=float, default=2e-4, dest='lr',
	                         help='Optimizer learning rate.')
optimizer_group.add_argument('-decay_frac', '--frac_decay_start', type=float, default=0.8,
	                         help='Epoch fraction of total when to start linear decay.\nValue '
	                              '0.0 starts lr decay process in the beginning,\n1.0 is no lr '
	                              'decay. At end of decay process lr is\nalways equal to 0.')

### Dataset Params ###
dataset_group = parser.add_argument_group('data arguments')
dataset_group.add_argument('--data_dir', required=True)
dataset_group.add_argument('-sz', '--image_size', type=int, default=64,
	                       choices=[64, 128, 256, 512], help='Expected image size.')
dataset_group.add_argument('-max_v', '--max_ct_value', type=int, default=2**12 - 1,
	                       help='Max expected CT value.')
dataset_group.add_argument('--reverse_direction', action='store_true',
	                       help='Learns MV to kV rather than kV to MV.')
