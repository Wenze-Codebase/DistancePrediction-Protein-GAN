import collections

training_protein_nums=6862

EPS = 1e-12

input_chanel_num=130

filters=64

max_length=300

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, gen_loss_GAN, gen_loss_L1, channel_attention, spatial_attention, train")
