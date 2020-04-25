{
  dataset_reader: {
    type: 'doctaet_reader',
    token_indexers: {
        bert: {
          type: "bert-pretrained",
          pretrained_model: std.extVar("BERT_VOCAB"),
          do_lowercase: std.extVar("IS_LOWERCASE"),
          truncate_long_sequences : true
        }
      },
  },
  train_data_path: std.extVar("TRAIN_PATH"),
  validation_data_path: std.extVar("DEV_PATH"),
  test_data_path: std.extVar("TEST_PATH"),
  model :{
    type: "doctaet",
      bert_model: {
          pretrained_model: std.extVar("BERT_WEIGHTS"),
          requires_grad : "pooler,11"
      },
      aggregate_feedforward: {
        input_dim: 768,
        num_layers: 2,
        hidden_dims: [200, 2],
        activations: ["relu", "linear"],
        dropout: [0.2, 0.0]
      }
  },
  iterator: {
    type: "basic",
    batch_size: 10,
  },
  validation_iterator: {
    type: "basic",
    batch_size: 10,
  },
  trainer: {
    num_epochs: 30,
    grad_norm: 5.0,
    patience : 7,
    cuda_device : std.extVar("CUDA_DEVICE"),
    validation_metric: '+f1',
    learning_rate_scheduler: {
      type: "reduce_on_plateau",
      factor: 0.5,
      mode: "max",
      patience: 20
    },
    optimizer: {
      type: "adam",
      lr: 1e-3,
      parameter_groups :[
        [[".*bert_model.*"], {"lr": 2e-5}], 
      ]
    },
    num_serialized_models_to_keep: 1,
    should_log_learning_rate: true
  },
  evaluate_on_test: true

}

