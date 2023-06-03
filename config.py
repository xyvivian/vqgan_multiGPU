import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.model = model = ml_collections.ConfigDict()
    model.ch = [64,64,128,128,256]
    model.num_res_blocks = 2
    model.attn_resolutions = []
    model.resamp_with_conv = True
    model.in_channels = 3
    model.resolution = 
    model.z_channels = 256
    model.latent_dim = 256
    model.num_codebook_vectors = 512
    model.ckpt_path = None
    model.disc_n_layers = 3
    model.num_filters_last = 64
    model.disc_loss = "hinge"
    
    config.ignore_keys = []
    config.image_key = "image"
    config.colorize_nlabels = None
    config.monitor = None

    config.lr = lr = ml_collections.ConfigDict()
    lr.min_lr = 1e-5
    lr.max_lr = 1e-4
    lr.max_decay_steps = 5000
    
    config.weight = weight = ml_collections.ConfigDict()
    weight.disc_start = 5000
    weight.codebook_weight = 1.0
    weight.pixelloss_weight = 1.0
    weight.disc_facotr = 1.0
    weight.disc_weight = 0.1
    weight.perceptual_weight = 0.1

    config.data = data = ml_collections.ConfigDict()
    data.batch_size = 128
    data.train = True
    data.num_workers = 12
    data.training_images_list_file = ""
    data.test_images_list_file = ""

    config.resume = False
    config.checkpoint = None

    return config