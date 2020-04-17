// Library that accepts a parameter dict and returns a full config.

function(p) {
  // Storing constants.

  local bert_base_dim = 768,
  local lstm_hidden_size = 200,
  local token_embedding_dim = bert_base_dim,
  local context_encoder_dim = 2 * lstm_hidden_size,
  local endpoint_span_embedding_dim = 2 * context_encoder_dim,
  local attended_span_embedding_dim = context_encoder_dim,
  local span_embedding_dim = endpoint_span_embedding_dim + attended_span_embedding_dim,
  local n_features = 1 + 4 + 5,
  local featured_embedding_dim = span_embedding_dim + n_features,

  ////////////////////////////////////////////////////////////////////////////////

  // Function definitions
  local lstm_context_encoder = {
    type: "lstm",
    bidirectional: true,
    input_size: token_embedding_dim,
    hidden_size: lstm_hidden_size
  },

  local make_feedforward(input_dim) = {
    input_dim: input_dim,
    num_layers: 2,
    hidden_dims: 150,
    activations: "relu",
    dropout: 0.2
  },

  local token_indexers = {
    bert: {
      type: "bert-pretrained",
      pretrained_model: std.extVar("BERT_VOCAB"),
      do_lowercase: std.extVar("IS_LOWERCASE"),
      use_starting_offsets: true,
      truncate_long_sequences : false
    }
  },
  local text_field_embedder = {
      allow_unmatched_keys: true,
      embedder_to_indexer_map: {
        bert: ["bert", "bert-offsets"],
      },
      token_embedders : {
        bert: {
            type: "bert-pretrained-modified",
            pretrained_model: std.extVar("BERT_WEIGHTS"),
            requires_grad: p.bert_fine_tune,
        },
    }
  },

  ////////////////////////////////////////////////////////////////////////////////

  // The model

  dataset_reader: {
    type: 'scirex_full_reader',
    token_indexers: token_indexers,
    document_filter_type: p.document_filter,
    filter_to_salient: p.filter_to_salient,
  },
  train_data_path: std.extVar("TRAIN_PATH"),
  validation_data_path: std.extVar("DEV_PATH"),
  test_data_path: std.extVar("TEST_PATH"),

  model: {
    type: "scirex_model",
    text_field_embedder: text_field_embedder,
    loss_weights: p.loss_weights,
    lexical_dropout: 0.2,
    display_metrics: ["validation_metric"],
    context_layer: lstm_context_encoder,
    modules: {
      coref: {
        antecedent_feedforward: make_feedforward(featured_embedding_dim),
      },
      ner: {
        mention_feedforward: make_feedforward(context_encoder_dim),
        label_namespace: 'ner_entity_labels',
        label_encoding: 'BIOUL',
      },
      relation: {
        antecedent_feedforward: make_feedforward(3*featured_embedding_dim),
      },
      saliency_classifier: {
        mention_feedforward: make_feedforward(featured_embedding_dim),
        label_namespace: "span_saliency_labels",
        n_features: n_features
      },
      n_ary_relation: {
        antecedent_feedforward: make_feedforward(4*featured_embedding_dim),
	      relation_cardinality: p.relation_cardinality
      },
      cluster_classifier: {
        antecedent_feedforward: make_feedforward(featured_embedding_dim),
      },
    }
  },
  iterator: {
    type: "ie_batch",
    batch_size: 50,
  },
  validation_iterator: {
    type: "ie_batch",
    batch_size: 50,
  },
  trainer: {
    num_epochs: 30,
    grad_norm: 5.0,
    patience : 7,
    cuda_device : std.extVar("CUDA_DEVICE"),
    validation_metric: '+validation_metric',
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
  }
}
