{
  dataset_reader: {
    type: "pwc_linker_bert",
    token_indexers: {
      bert: {
        type: "bert-pretrained",
        pretrained_model: std.extVar("BERT_VOCAB"),
        do_lowercase: std.extVar("IS_LOWERCASE"),
        truncate_long_sequences : false
      }
    },
    tokenizer: {
       "word_splitter": "bert-basic"
    }
  },
  train_data_path: std.extVar("TRAIN_PATH"),
  validation_data_path: std.extVar("DEV_PATH"),
  test_data_path: std.extVar("TEST_PATH"),
  model: {
    type: "bert_coreference",
    bert_model: {
        pretrained_model: std.extVar("BERT_WEIGHTS"),
        requires_grad : "pooler,10,11"
    },
    aggregate_feedforward: {
      input_dim: 768,
      num_layers: 2,
      hidden_dims: [200, 2],
      activations: ["relu", "linear"],
      dropout: [0.2, 0.0]
    },
    featured : false,
     initializer: [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*token_embedder_tokens\\._projection.*weight", {"type": "xavier_normal"}]
     ]
   },
  iterator: {
    type: "bucket_sample",
    sorting_keys: [],
    batch_size: 50
  },

  trainer: {
    num_epochs: 20,
    num_serialized_models_to_keep: 1,
    patience: 10,
    cuda_device: std.parseInt(std.extVar("CUDA_DEVICE")),
    grad_norm: 5.0,
    validation_metric: "+f1",
    optimizer: {
      type: "adam",
      lr: 0.001
    }
  }
}