audio_config = {
    'K': 1,
    'output_dim': 256,
    'use': 'hidden_state',
    'num_label': 7,
    'path': '../KEMDy20/wav/',
    'cuda': 'cuda:0',
    # about 10s of wav files
    'max_length' : 512
}

text_config = {
    'K': 1,
    'output_dim': 256,
    'num_label': 7,
    'max_length': 128,
    'cuda': 'cuda:0',
    'freeze': True
}

multimodal_config = {
    'output_dim': 512,
    'num_labels': 7,
    'dropout': 0.1,
    'cuda': 'cuda:0',
    'use_threeway':False,
    'use_attention':False
}

train_config = {
    'epochs': 30,
    'batch_size': 64,
    'lr': 5e-5,
    'accumulation_steps': 8,
    'cuda': 'cuda:0'
}

mixer_config = {
    'projection_dim' : 256,
    'output_dim' : 512,
    'num_blocks' : 1,
    'dropout' : 0.1,
    'num_labels' : 7,
    'cuda' : 'cuda:0'
}

cross_attention_config = {
    'projection_dim': 768,
    'output_dim': 512,
    'num_labels': 7,
    'dropout': 0.1,
    'cuda': 'cuda:0',
    'num_heads': 8,
    'layers': 1,
    'attn_dropout': 0,
    'relu_dropout': 0,
    'res_dropout': 0,
    'embed_dropout': 0
}

test_config = {
    'batch_size': 64,
    'cuda': 'cuda:0'
}
